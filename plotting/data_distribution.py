"""Distribution of the training data: SIC on day k (left) and the residual k+1 - k (right).

Both over water pixels of the training slice, from every `stride`-th window, as in
CARRA2Forecast.residual_stats (neighbouring windows are near-duplicates). The y-axis is
logarithmic: most water pixels are open water (SIC = 0) and many do not change in a month
(residual = 0), so on a linear axis those spikes would hide everything else. The dashed lines
on the right mark +-sigma_r, the measured residual std the DDPM divides by, and a small box
gives its value. No separate standardized-residual figure: dividing by sigma_r only relabels the
x-axis, so it would show the same shape.

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
    (sic_counts, _), (res_counts, _), sic_zero, res_zero, res_std = dask.compute(
        dsa.histogram(k_water, bins=SIC_BINS), dsa.histogram(r_water, bins=RESIDUAL_BINS),
        (k_water == 0).mean(), (r_water == 0).mean(), r_water.std(),
    )
    return {"sic": sic_counts, "residual": res_counts, "sic_zero": float(sic_zero),
            "residual_zero": float(res_zero), "residual_std": float(res_std), "n_windows": len(w)}


def plot_data_distribution(h: dict, out_dir: str = "figures"):
    fig, (ax_sic, ax_res) = plt.subplots(1, 2, sharey=True)
    panels = [(ax_sic, h["sic"], SIC_BINS, muted(COLORS[0]), r"SIC on day $k$"),
              (ax_res, h["residual"], RESIDUAL_BINS, muted(COLORS[1]), r"residual $x_{k+1} - x_k$")]
    for ax, counts, bins, color, xlabel in panels:
        ax.stairs(counts, bins, fill=True, color=color, linewidth=0)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
    for s in (-h["residual_std"], h["residual_std"]):
        ax_res.axvline(s, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)
    # Top right is empty: the peak is central and the tails stay below ~1e6.
    ax_res.text(0.97, 0.95, rf"$\sigma_r = {h['residual_std']:.3f}$", transform=ax_res.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=MUTED, linewidth=0.6))
    # Counts over the sampled windows only (every `stride`-th), so they scale with 1/stride.
    ax_sic.set_ylabel("pixel count")

    savefig(fig, out_dir, "data_distribution", width=TEXTWIDTH_IN, height=2.8)
    plt.close(fig)
    print(f"Wrote {out_dir}/data_distribution.pdf")


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
