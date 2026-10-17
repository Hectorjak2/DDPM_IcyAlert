"""Thesis figures for conditional DDPM samples.

Reads ``results/<run>/samples.pkl`` written by
``utils.helper_functions.download_conditional_samples`` and writes four PDFs next to it:

1. ``sampling_loss``  -- MSE against the target per reverse step, averaged over samples.
2. ``calibration``    -- mean predicted SIC per 0.01 target-SIC bin (over/under-prediction).
3. ``forecast_row``   -- k-1, k, k+1 (truth) and prediction for one sample.
4. ``forecast_grid``  -- truth (top) vs prediction (bottom) for four samples.

Works for both CARRA2-TEST and CARRA2-WEST without a mask file: land is already NaN in
the pickled targets and samples, so water = ``isfinite(target)`` at whatever resolution
the run used.

Run from the repo root:  python -m plotting.ddpm_plots <run_name | path/to/samples.pkl>
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from plotting.plot_config import savefig, TEXTWIDTH_IN, COLORS, MUTED, REF_LINESTYLE
from utils.dataloader import to_data_range

# Sea-ice colormap: open water dark blue, full ice white, land grey.
SIC_CMAP = plt.cm.Blues_r.copy()
SIC_CMAP.set_bad("lightgray")

N_BINS = 100  # 0.01-wide target-SIC bins for the calibration plot
MIN_BIN_PIXELS = 100  # bins with fewer pixels are left out of the calibration curves


def _month(date) -> str:
    """np.datetime64 -> 'Sep 2026'."""
    return pd.Timestamp(date).strftime("%b %Y")


def _as_data_range(field: torch.Tensor) -> torch.Tensor:
    """Older pickles stored targets/contexts in model range [-1, 1]. SIC is never
    negative, so any negative value means the field still needs converting."""
    finite = field[torch.isfinite(field)]
    if finite.numel() and finite.min() < 0:
        return to_data_range(field)
    return field


def load_samples(path: str | Path):
    """Load a samples.pkl, given its path or the run's results directory/name.

    Returns ``(run_dir, data)`` where every field in ``data`` is a numpy [H, W] array
    in data range [0, 1] with land as NaN, and ``sampling_losses`` is an [N, steps]
    array (or None for pickles written before the running loss existed).
    """
    path = Path(path)
    if path.suffix != ".pkl":
        path = (path if path.exists() else Path("results") / path) / "samples.pkl"

    with open(path, "rb") as f:
        payload = pickle.load(f)
    samples, contexts, targets, times = payload[:4]
    losses = payload[4] if len(payload) > 4 else None

    data = {
        "predictions": [s.squeeze().numpy() for s in samples],
        "targets": [_as_data_range(t).squeeze().numpy() for t in targets],
        # contexts are [2, H, W] = (k, k-1)
        "k": [_as_data_range(c)[0].numpy() for c in contexts],
        "km1": [_as_data_range(c)[1].numpy() for c in contexts],
        "times": times,  # (k+1, k, k-1) dates
        "sampling_losses": None,
    }
    # An intermediate version stored only the last sample's curve (a flat list of floats).
    if losses:
        if not isinstance(losses[0], (list, tuple)):
            losses = [losses]
        data["sampling_losses"] = np.asarray(losses, dtype=np.float64)

    return path.parent, data


def _clean_image_axis(ax):
    """Hide ticks, spines and grid but keep the axis so titles/ylabels still show."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _show_field(ax, field):
    return ax.imshow(field, cmap=SIC_CMAP, vmin=0.0, vmax=1.0, origin="lower",
                     interpolation="none")


def plot_sampling_loss(data, out_dir):
    """MSE vs reverse step, averaged over samples."""
    losses = data["sampling_losses"]
    if losses is None:
        print("No sampling losses in this pickle -- skipping sampling_loss.")
        return
    n_steps = losses.shape[1]
    # The reverse loop runs t = T-1 -> 0, so entry 0 belongs to the noisiest step.
    t = np.arange(n_steps - 1, -1, -1)

    fig, ax = plt.subplots()
    ax.plot(t, losses.mean(axis=0), color=COLORS[0])
    ax.set_xlim(n_steps - 1, 0)  # sampling runs left to right, ending at t = 0
    ax.set_ylim(bottom=0)
    ax.set_xlabel("timestep $t$")
    ax.set_ylabel("MSE over water")

    savefig(fig, out_dir, "sampling_loss", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def plot_calibration(data, out_dir):
    """Prediction per target-SIC bin, pooled over all water pixels of all samples.

    Shows mean, median and interquartile range rather than mean +- std: within a bin the
    predictions are bimodal (a misplaced ice edge gives ~0 or ~1 where the target is
    intermediate), so the std sits near its 0/1-mixture maximum in every bin and says
    little, while the quartiles show the split.
    """
    target = np.concatenate([t.ravel() for t in data["targets"]]).astype(np.float64)
    pred = np.concatenate([p.ravel() for p in data["predictions"]]).astype(np.float64)
    valid = np.isfinite(target) & np.isfinite(pred)
    target, pred = target[valid], pred[valid]

    # SIC exactly 1.0 goes in the last bin rather than a 101st one.
    bins = np.clip((target * N_BINS).astype(np.int64), 0, N_BINS - 1)
    counts = np.bincount(bins, minlength=N_BINS)
    # Group predictions by bin with one sort instead of a boolean mask per bin.
    per_bin = np.split(pred[np.argsort(bins, kind="stable")], np.cumsum(counts)[:-1])
    stats = np.full((N_BINS, 4), np.nan)  # mean, q25, median, q75
    for k, p in enumerate(per_bin):
        if p.size:
            stats[k] = [p.mean(), *np.percentile(p, [25, 50, 75])]
    centers = (np.arange(N_BINS) + 0.5) / N_BINS

    # CARRA2 has (almost) no SIC between 0 and ~0.05, which leaves the 0-0.01 bin -- the
    # bulk of all water pixels -- isolated between empty bins, and a line does not draw
    # through NaNs. Curves therefore run through the populated bins only, bridging the gap;
    # the count panel below shows that the bridged bins are (near-)empty. Bins with only a
    # handful of pixels are bridged too: their quantiles are noise and draw spurious kinks.
    filled = counts >= MIN_BIN_PIXELS
    mean, q25, median, q75 = stats[filled].T

    fig, (ax, ax_n) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE,
            label="perfect")
    ax.fill_between(centers[filled], q25, q75, color=COLORS[0], alpha=0.25, linewidth=0,
                    label="interquartile range")
    ax.plot(centers[filled], mean, color=COLORS[0], label="mean")
    ax.plot(centers[filled], median, color=COLORS[2], linewidth=1.0, label="median")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("predicted SIC")
    # Above the axes: inside, the band and curves leave no free corner.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    ax_n.bar(centers, counts, width=1 / N_BINS, color=COLORS[3], linewidth=0)
    ax_n.set_yscale("log")
    ax_n.set_ylabel("pixels")
    ax_n.set_xlabel("target SIC")

    savefig(fig, out_dir, "calibration", width=TEXTWIDTH_IN / 2, height=3.4)
    plt.close(fig)


def plot_forecast_row(data, out_dir, index: int = 0):
    """k-1, k, k+1 (truth) and prediction for one sample, with a shared colorbar."""
    kp1_date, k_date, km1_date = data["times"][index]
    panels = (
        (data["km1"][index], rf"$k{{-}}1$: {_month(km1_date)}"),
        (data["k"][index], rf"$k$: {_month(k_date)}"),
        (data["targets"][index], rf"$k{{+}}1$ truth: {_month(kp1_date)}"),
        (data["predictions"][index], rf"$k{{+}}1$ prediction"),
    )

    fig, axes = plt.subplots(1, 5, gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.06]})
    for ax, (field, title) in zip(axes[:4], panels):
        img = _show_field(ax, field)
        ax.set_title(title)
        _clean_image_axis(ax)
    fig.colorbar(img, cax=axes[4]).set_label("SIC", labelpad=4)

    savefig(fig, out_dir, "forecast_row", width=TEXTWIDTH_IN, height=1.9)
    plt.close(fig)


def plot_forecast_grid(data, out_dir, indices=None):
    """Truth (top row) vs prediction (bottom row) for up to four samples."""
    if indices is None:
        indices = range(min(4, len(data["predictions"])))
    indices = list(indices)
    n = len(indices)

    fig = plt.figure()
    gs = fig.add_gridspec(2, n + 1, width_ratios=[1] * n + [0.06])
    for col, i in enumerate(indices):
        ax_t = fig.add_subplot(gs[0, col])
        ax_p = fig.add_subplot(gs[1, col])
        img = _show_field(ax_t, data["targets"][i])
        _show_field(ax_p, data["predictions"][i])
        ax_t.set_title(_month(data["times"][i][0]))
        for ax in (ax_t, ax_p):
            _clean_image_axis(ax)
        if col == 0:
            ax_t.set_ylabel("truth")
            ax_p.set_ylabel("prediction")
    fig.colorbar(img, cax=fig.add_subplot(gs[:, n])).set_label("SIC", labelpad=4)

    savefig(fig, out_dir, "forecast_grid", width=TEXTWIDTH_IN, height=3.3)
    plt.close(fig)


def make_all(path, row_index: int = 0, grid_indices=None):
    out_dir, data = load_samples(path)
    plot_sampling_loss(data, out_dir)
    plot_calibration(data, out_dir)
    plot_forecast_row(data, out_dir, index=row_index)
    plot_forecast_grid(data, out_dir, indices=grid_indices)
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under results/, its directory, or a samples.pkl path")
    parser.add_argument("--row-index", type=int, default=0, help="sample shown in forecast_row")
    parser.add_argument("--grid-indices", type=int, nargs="+", default=None,
                        help="samples shown in forecast_grid (default: first four)")
    args = parser.parse_args()
    make_all(args.run, row_index=args.row_index, grid_indices=args.grid_indices)
