"""Distribution of the training data: SIC on day k and the residual k+1 - k, one PDF each.

Both over water pixels of the training slice, from every `stride`-th window, as in
CARRA2Forecast.residual_stats (neighbouring windows are near-duplicates). The y-axis is
logarithmic: most water pixels are open water (SIC = 0) and many do not change in a month
(residual = 0), so on a linear axis those spikes would hide everything else. The dashed lines
on the residual mark +-sigma_r, the measured residual std the models divide by; boxes give the
mean and std of each histogram.

Each is also drawn standardized as the models see it: SIC z-scored, (x_k - mu) / sigma (the UNet's
input scaling), and the residual divided by sigma_r without subtracting its mean (what both models
predict). Both are linear maps, so the shapes are unchanged and the counts are reused on
transformed bin edges.

Writes four PDFs to figures/, each a third of the text width so they can be set side by side:
data_distribution_sic, data_distribution_residual, and the same with _standardized. All four share
one y-range, so side-by-side panels line up.

Run from the repo root:  python -m plotting.data_distribution [--area CARRA_CROPPED] [--stride 5]
"""
import argparse

import dask
import dask.array as dsa
import matplotlib.pyplot as plt
import numpy as np

from config import TRAINING_SLICE
from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MUTED, REF_LINESTYLE, muted
from utils.dataloader import CARRA2Forecast, load_residual_stats

SIC_BINS = np.linspace(0.0, 1.0, 51)
RESIDUAL_BINS = np.linspace(-1.0, 1.0, 81)


def histograms(area: DatasetChoice, time_slice: slice, stride: int) -> dict:
    """Counts of SIC (day k) and residual (k+1 - k) over water pixels, plus summary numbers."""
    ds = CARRA2Forecast("siconc", "cpu", time_slice=time_slice, area=area, batch_dim=False)
    w = ds._windows[::stride]
    k = ds.da.isel(time=w[:, 1]).data
    kp1 = ds.da.isel(time=w[:, 2]).data
    r = kp1 - k  # dask arrays, so no time-coordinate alignment between the two
    k_water, r_water = k[dsa.isfinite(k)], r[dsa.isfinite(r)]
    (sic_counts, _), (res_counts, _), sic_zero, res_zero, sic_mean, sic_std, res_mean, res_std = dask.compute(
        dsa.histogram(k_water, bins=SIC_BINS), dsa.histogram(r_water, bins=RESIDUAL_BINS),
        (k_water == 0).mean(), (r_water == 0).mean(),
        k_water.mean(), k_water.std(), r_water.mean(), r_water.std(),
    )
    return {"sic": sic_counts, "residual": res_counts, "sic_zero": float(sic_zero),
            "residual_zero": float(res_zero), "sic_mean": float(sic_mean), "sic_std": float(sic_std),
            "residual_mean": float(res_mean), "residual_std": float(res_std), "n_windows": len(w)}


def _stats_box(ax, x: float, ha: str, text: str):
    """Square-cornered box with summary statistics, near the top of the axes."""
    ax.text(x, 0.95, text, transform=ax.transAxes, ha=ha, va="top",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=MUTED, linewidth=0.6))


def _ylim(h: dict) -> tuple[float, float]:
    """One y-range for all four PDFs, so they line up when set side by side. The top leaves
    ~1.4 decades above the tallest bar for the top-right box (SIC has a spike right under it)."""
    counts = np.concatenate([h["sic"], h["residual"]])
    return 0.5 * counts[counts > 0].min(), 25 * counts.max()


def _plot_single(h: dict, out_dir: str, name: str, counts, bins, color, xlabel: str,
                 stats_text: str, ref: float | None = None):
    """One histogram, a third of the text width. ref: draw dashed lines at +-ref."""
    fig, ax = plt.subplots()
    ax.stairs(counts, bins, fill=True, color=color, linewidth=0)
    ax.set_yscale("log")
    ax.set_ylim(*_ylim(h))
    if ref is not None:
        for s in (-ref, ref):
            ax.axvline(s, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)
    _stats_box(ax, 0.97, "right", stats_text)
    ax.set_xlabel(xlabel)
    # Counts over the sampled windows only (every `stride`-th), so they scale with 1/stride.
    ax.set_ylabel("pixel count")

    savefig(fig, out_dir, name, width=TEXTWIDTH_IN / 3, height=2.4)
    plt.close(fig)
    print(f"Wrote {out_dir}/{name}.pdf")


def _stats(mu_name: str, sigma_name: str, mu: float, sigma: float) -> str:
    # +0.0 turns -0.000 into 0.000.
    return rf"${mu_name} = {mu + 0.0:.3f}$" "\n" rf"${sigma_name} = {sigma:.3f}$"


def plot_data_distribution(h: dict, out_dir: str = "figures"):
    """Raw SIC and residual, in teal and orange."""
    mu, sigma, mu_r, sigma_r = h["sic_mean"], h["sic_std"], h["residual_mean"], h["residual_std"]
    _plot_single(h, out_dir, "data_distribution_sic", h["sic"], SIC_BINS, muted(COLORS[0]),
                 r"SIC on day $k$", _stats(r"\mu", r"\sigma", mu, sigma))
    _plot_single(h, out_dir, "data_distribution_residual", h["residual"], RESIDUAL_BINS, muted(COLORS[1]),
                 r"residual $x_{k+1} - x_k$", _stats(r"\mu_r", r"\sigma_r", mu_r, sigma_r), ref=sigma_r)


def plot_standardized_distribution(h: dict, out_dir: str = "figures"):
    """Standardized SIC and residual, in steel blue and coral to set them apart from the raw ones.
    Linear maps: the counts stay, only the bin edges (and the statistics) move."""
    mu, sigma, mu_r, sigma_r = h["sic_mean"], h["sic_std"], h["residual_mean"], h["residual_std"]
    _plot_single(h, out_dir, "data_distribution_sic_standardized", h["sic"], (SIC_BINS - mu) / sigma,
                 muted(COLORS[3]), r"$(x_k - \mu)/\sigma$", _stats(r"\mu", r"\sigma", 0.0, 1.0))
    _plot_single(h, out_dir, "data_distribution_residual_standardized", h["residual"], RESIDUAL_BINS / sigma_r,
                 muted(COLORS[2]), r"$(x_{k+1} - x_k)/\sigma_r$",
                 _stats(r"\mu_r", r"\sigma_r", mu_r / sigma_r, 1.0), ref=1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--area", default=DatasetChoice.CARRA_CROPPED.value,
                        choices=[a.value for a in DatasetChoice if a != DatasetChoice.FASHION])
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    area = DatasetChoice(args.area)
    h = histograms(area, TRAINING_SLICE, args.stride)
    print(f"{h['n_windows']} windows | SIC exactly 0: {h['sic_zero']:.1%} | "
          f"residual exactly 0: {h['residual_zero']:.1%} | residual std: {h['residual_std']:.4f}")
    # The boxed value is this run's estimate; it should match what training divides by.
    cached = load_residual_stats(area, TRAINING_SLICE).get("std")
    if cached is not None:
        print(f"sigma_r in residual_stats.json: {cached:.4f}")
    plot_data_distribution(h)
    plot_standardized_distribution(h)
