"""Thesis figures for the sampling analysis of a conditional DDPM.

Reads ``results/<run>/loss_stats.pkl`` written by
``utils.sampling.analysis_sampling`` and writes two PDFs next to it:

1. ``truncated_sampling_loss`` -- final MSE when sampling with only the first t reverse
   steps (t = 100, 200, ..., T), averaged over samples.
2. ``context_sampling_loss``   -- MSE difference to the true-context run per reverse step,
   for a wrong (randomly drawn) context and, when present, a null (all-zero) context.
   Positive means a higher MSE than with the true context.

Run from the repo root:  python -m plotting.sample_analysis <run_name | path/to/loss_stats.pkl>
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MARKERS

# Increment of the timestep sweep in analysis_sampling (torch.arange(100, T + 100, 100))
T_INCREMENT = 100


def load_loss_stats(path: str | Path):
    """Load a loss_stats.pkl, given its path or the run's results directory/name.

    Returns ``(run_dir, varying, sampling, no_context, zero_context)`` as numpy arrays of
    shape [N, n_increments], [N, T], [N, T] and [N, T]. ``zero_context`` is None for older
    pickles written before the null-context run was added.
    """
    path = Path(path)
    if path.suffix != ".pkl":
        path = (path if path.exists() else Path("results") / path) / "loss_stats.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Sample analytics not found: {path}")

    with open(path, "rb") as f:
        stats = pickle.load(f)
    varying, sampling, no_context = (np.asarray(x) for x in stats[:3])
    zero_context = np.asarray(stats[3]) if len(stats) > 3 else None
    return path.parent, varying, sampling, no_context, zero_context


def plot_truncated_sampling_loss(varying, out_dir):
    """Final MSE vs number of reverse steps used, averaged over samples."""
    t = T_INCREMENT * np.arange(1, varying.shape[1] + 1)

    fig, ax = plt.subplots()
    ax.plot(t, varying.mean(axis=0), color=COLORS[0], marker=MARKERS[0])
    ax.set_ylim(bottom=0)
    ax.set_xticks(t)
    ax.set_xlabel("reverse steps used $t$")
    ax.set_ylabel("MSE over water")

    savefig(fig, out_dir, "truncated_sampling_loss", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def plot_context_sampling_loss(sampling, no_context, out_dir, zero_context=None):
    """MSE difference to the true-context run vs reverse step, averaged over samples.

    The runs share target and base field per sample, so the difference is taken per sample
    before averaging; the band is +-1 standard error of that paired difference.
    """
    n_steps = sampling.shape[1]
    # The reverse loop runs t = T-1 -> 0, so entry 0 belongs to the noisiest step.
    t = np.arange(n_steps - 1, -1, -1)

    runs = [("Wrong context", no_context, COLORS[2])]
    if zero_context is not None:
        runs.append(("Null context", zero_context, COLORS[4]))

    fig, ax = plt.subplots()
    ax.axhline(0, color=COLORS[3], linewidth=0.8, label="True context")
    for label, losses, color in runs:
        diff = losses - sampling
        mean = diff.mean(axis=0)
        se = diff.std(axis=0, ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else np.zeros_like(mean)
        ax.plot(t, mean, color=color, label=label)
        ax.fill_between(t, mean - se, mean + se, color=color, alpha=0.25, linewidth=0)
    ax.set_xticks(np.arange(0, n_steps + 1, T_INCREMENT))
    # Rotated: at half text width the 100-step labels overlap when horizontal.
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlim(n_steps, 0)  # sampling runs left to right, ending at t = 0
    ax.set_xlabel("timestep $t$")
    ax.set_ylabel(r"$\Delta$ MSE vs true context")
    ax.legend()

    savefig(fig, out_dir, "context_sampling_loss", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def make_all(path):
    out_dir, varying, sampling, no_context, zero_context = load_loss_stats(path)
    plot_truncated_sampling_loss(varying, out_dir)
    plot_context_sampling_loss(sampling, no_context, out_dir, zero_context)
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under results/, its directory, or a loss_stats.pkl path")
    args = parser.parse_args()
    make_all(args.run)
