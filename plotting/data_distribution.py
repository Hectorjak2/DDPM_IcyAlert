"""Distribution of the training data: SIC on day k and the residual k+1 - k, side by side.

Both over water pixels of the training slice, from every `stride`-th window, as in
CARRA2Forecast.residual_stats (neighbouring windows are near-duplicates). The shared y-axis is
logarithmic: most water pixels are open water (SIC = 0) and many do not change in a month
(residual = 0), so on a linear axis those spikes would hide everything else. The dashed lines
on the residual mark +-sigma_r, the measured residual std the models divide by; boxes give the
mean and std of each histogram.

Writes figures/data_distribution.pdf at the full text width.

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
    """Shared y-range. The top leaves ~1.4 decades above the tallest bar for the top-right box
    (SIC has a spike right under it)."""
    counts = np.concatenate([h["sic"], h["residual"]])
    return 0.5 * counts[counts > 0].min(), 25 * counts.max()


def _stats(mu_name: str, sigma_name: str, mu: float, sigma: float) -> str:
    # +0.0 turns -0.000 into 0.000.
    return rf"${mu_name} = {mu + 0.0:.3f}$" "\n" rf"${sigma_name} = {sigma:.3f}$"


def plot_data_distribution(h: dict, out_dir: str = "figures", name: str = "data_distribution"):
    """SIC (left) and residual (right) side by side, sharing the log y-axis."""
    mu, sigma, mu_r, sigma_r = h["sic_mean"], h["sic_std"], h["residual_mean"], h["residual_std"]
    fig, (ax_sic, ax_res) = plt.subplots(1, 2, sharey=True)

    ax_sic.stairs(h["sic"], SIC_BINS, fill=True, color=muted(COLORS[0]), linewidth=0)
    _stats_box(ax_sic, 0.97, "right", _stats(r"\mu", r"\sigma", mu, sigma))
    ax_sic.set_xlabel(r"SIC on day $k$")
    # Counts over the sampled windows only (every `stride`-th), so they scale with 1/stride.
    ax_sic.set_ylabel("pixel count")

    ax_res.stairs(h["residual"], RESIDUAL_BINS, fill=True, color=muted(COLORS[1]), linewidth=0)
    for s in (-sigma_r, sigma_r):
        ax_res.axvline(s, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)
    _stats_box(ax_res, 0.97, "right", _stats(r"\mu_r", r"\sigma_r", mu_r, sigma_r))
    ax_res.set_xlabel(r"residual $x_{k+1} - x_k$")

    ax_sic.set_yscale("log")  # shared, so this sets both
    ax_sic.set_ylim(*_ylim(h))

    savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=2.4)
    plt.close(fig)
    print(f"Wrote {out_dir}/{name}.pdf")


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
