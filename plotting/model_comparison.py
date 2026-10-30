"""Thesis figures comparing two sets of forecasts, e.g. the residual UNet and the DDPM.

Reads two ``samples.pkl`` (as ``plotting.evaluation``) and writes to ``--out``:

1. ``calibration`` -- mean and interquartile range of the prediction per 0.01 target-SIC
   bin for both models, one panel per lead time (2x3 for a six-month rollout).
2. ``mse_by_lead`` -- MSE over water against lead time for both models, with the
   climatology baseline.

Both sets should be sampled on the same dates (same ``TrainConfig.sample_seed``); the
baseline is computed from the first set, and a warning is printed if the dates differ.
Only the lead times both sets have are plotted.

Per-model figures stay in ``plotting.evaluation``.

Run from the repo root:
    python -m plotting.model_comparison <unet_run> <ddpm_run> [--labels UNet DDPM] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MUTED, REF_LINESTYLE
from plotting.evaluation import (MAX_CALIBRATION_COLS, _load_climatology, calibration_stats,
                                 load_leads, sample_errors)

# One colour and marker per model, in the order the sample sets are given.
MODEL_STYLES = [(COLORS[0], "o"), (COLORS[7], "D")]
BAND_ALPHA = 0.2  # two overlapping interquartile bands stay readable


def _check_dates(models):
    """Warn if the sets were not sampled on the same target dates."""
    (name_a, leads_a), (name_b, leads_b) = models
    dates_a = [d[0] for d in leads_a[0]["times"]]
    dates_b = [d[0] for d in leads_b[0]["times"]]
    if dates_a != dates_b:
        print(f"Warning: {name_a} and {name_b} were sampled on different dates "
              f"({len(dates_a)} vs {len(dates_b)} samples) -- the baseline follows {name_a}.")


def plot_calibration(models, out_dir):
    """Calibration per lead time with both models in each panel: mean as a line and
    interquartile range as a band in the model's colour. See
    ``plotting.evaluation.plot_calibration`` for why the quartiles rather than the std."""
    n = len(models[0][1])
    cols = min(n, MAX_CALIBRATION_COLS)
    rows = -(-n // cols)

    fig = plt.figure()
    # A legend row on top: inside the panels, the band and curves leave no free corner.
    gs = fig.add_gridspec(1 + rows, cols, height_ratios=[0.5 if cols == 1 else 0.3] + [3] * rows)
    first = None
    for i in range(n):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[1 + r, c], sharex=first, sharey=first)
        first = first or ax

        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)
        for (name, leads), (color, _) in zip(models, MODEL_STYLES):
            centers, filled, (mean, q25, q75) = calibration_stats(leads[i])
            ax.fill_between(centers[filled], q25, q75, color=color, alpha=BAND_ALPHA,
                            linewidth=0)
            ax.plot(centers[filled], mean, color=color)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if n > 1:
            ax.set_title(f"Lead: {i + 1}")

        if c == 0:
            ax.set_ylabel("predicted SIC")
        else:
            ax.tick_params(labelleft=False)
        # x axis labelled only on the lowest panel of each column
        if i + cols >= n:
            ax.set_xlabel("target SIC")
        else:
            ax.tick_params(labelbottom=False)

    # One entry per model, its mean line drawn over its band.
    handles = [Line2D([], [], color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)]
    labels = ["perfect"]
    for (name, _), (color, _) in zip(models, MODEL_STYLES):
        handles.append((Patch(facecolor=color, alpha=BAND_ALPHA, linewidth=0),
                        Line2D([], [], color=color)))
        labels.append(f"{name} (mean, IQR)")
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="center", ncol=2 if cols == 1 else 3,
                     handler_map={tuple: HandlerTuple(ndivide=1)})

    width = TEXTWIDTH_IN / 2 if cols == 1 else TEXTWIDTH_IN
    height = 2.8 if cols == 1 else 0.4 + 2.1 * rows
    savefig(fig, out_dir, "calibration", width=width, height=height)
    plt.close(fig)


def plot_mse_by_lead(models, out_dir):
    """MSE over water against lead time for both models, with the climatology baseline
    (from the first set). Each point is the mean of the per-sample MSE, as in
    ``plotting.evaluation.plot_mse_by_lead``."""
    reference = models[0][1]
    climatology = _load_climatology(reference[0]["targets"][0].shape)
    lead_times = np.arange(1, len(reference) + 1)

    series = []  # (label, per-lead MSE, color, marker)
    for (name, leads), (color, marker) in zip(models, MODEL_STYLES):
        # The baseline is not needed per model, so no climatology here.
        series.append((name, [sample_errors(d, None).mse.mean() for d in leads], color, marker))
    if climatology is not None:
        baseline = [sample_errors(d, climatology).climatology.mean() for d in reference]
        series.append(("climatology", baseline, COLORS[3], "^"))

    fig, ax = plt.subplots()
    for label, mse, color, marker in series:
        ax.plot(lead_times, mse, color=color, marker=marker, markersize=3.5, linewidth=1.0,
                label=label)
    ax.set_xticks(lead_times)
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("MSE over water")
    ax.set_ylim(bottom=0)
    # Above the axes, so it never covers a point.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series))

    savefig(fig, out_dir, "mse_by_lead", width=TEXTWIDTH_IN / 2, height=2.6)
    plt.close(fig)


def make_all(runs, labels, out_dir):
    models = [(label, load_leads(run)[1]) for run, label in zip(runs, labels)]
    n = min(len(leads) for _, leads in models)
    if any(len(leads) != n for _, leads in models):
        print(f"The sets have different lead times -- plotting the first {n}.")
    models = [(label, leads[:n]) for label, leads in models]
    _check_dates(models)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_calibration(models, out_dir)
    plot_mse_by_lead(models, out_dir)
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs=2,
                        help="two runs: name under results/, its directory, or a samples.pkl path")
    parser.add_argument("--labels", nargs=2, default=["UNet", "DDPM"],
                        help="legend names of the two runs, in the same order")
    parser.add_argument("--out", default="results/model_comparison",
                        help="directory the figures are written to")
    args = parser.parse_args()
    make_all(args.runs, args.labels, args.out)
