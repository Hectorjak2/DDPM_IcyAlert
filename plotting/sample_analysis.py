"""Thesis figures for the sampling analysis of a conditional DDPM.

Reads ``results/<run>/loss_stats.pkl`` written by
``utils.helper_functions.analysis_sampling`` and writes two PDFs next to it:

1. ``truncated_sampling_loss`` -- final MSE when sampling with only the first t reverse
   steps (t = 100, 200, ..., T), averaged over samples.
2. ``context_sampling_loss``   -- MSE against the target per reverse step, with the
   true context vs a wrong (randomly drawn) context, averaged over samples.

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

    Returns ``(run_dir, varying, sampling, no_context)`` as numpy arrays of shape
    [N, n_increments], [N, T] and [N, T].
    """
    path = Path(path)
    if path.suffix != ".pkl":
        path = (path if path.exists() else Path("results") / path) / "loss_stats.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Sample analytics not found: {path}")

    with open(path, "rb") as f:
        varying, sampling, no_context = pickle.load(f)
    return path.parent, np.asarray(varying), np.asarray(sampling), np.asarray(no_context)


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


def plot_context_sampling_loss(sampling, no_context, out_dir):
    """MSE vs reverse step for true vs wrong context, averaged over samples."""
    n_steps = sampling.shape[1]
    # The reverse loop runs t = T-1 -> 0, so entry 0 belongs to the noisiest step.
    t = np.arange(n_steps - 1, -1, -1)

    fig, ax = plt.subplots()
    ax.plot(t, sampling.mean(axis=0), color=COLORS[0], label="True context")
    ax.plot(t, no_context.mean(axis=0), color=COLORS[2], label="Wrong context")
    # Symlog (linear on [0, 1], log above) stretches the last ~100 steps, where the curves
    # separate, while still showing t = 0, which a plain log axis cannot.
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(n_steps - 1, 0)  # sampling runs left to right, ending at t = 0
    ax.set_ylim(bottom=0)
    ax.set_xlabel("timestep $t$ (symlog)")
    ax.set_ylabel("MSE over water")
    ax.legend()

    savefig(fig, out_dir, "context_sampling_loss", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def make_all(path):
    out_dir, varying, sampling, no_context = load_loss_stats(path)
    plot_truncated_sampling_loss(varying, out_dir)
    plot_context_sampling_loss(sampling, no_context, out_dir)
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under results/, its directory, or a loss_stats.pkl path")
    args = parser.parse_args()
    make_all(args.run)
