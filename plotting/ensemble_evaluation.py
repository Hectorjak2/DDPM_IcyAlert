"""Rank histogram for ensemble forecasts.

Reads ``results/<run>/ensemble_samples.pkl`` written by ``ensemble.py`` and writes
``rank_histogram.pdf`` next to it: per-pixel rank of the true k+1 SIC among the M members,
pooled over all dates. A calibrated ensemble gives a flat histogram at 1/(M+1); a U shape means
too little spread, a dome too much, a slope a bias.

Two panels:
1. all water pixels;
2. the marginal ice zone, pixels with 0 < SIC < 1 on day k. Selected on day k, not on the
   target: conditioning on the observation biases the histogram (the forecaster's dilemma).

Ties (open water at 0, pack ice at 1) are broken at random, as in Hamill (2001); counting only
members strictly below the truth would pile every tie into rank 0 and fake a U shape.

Also prints the spread-skill pair per subset and season: mean ensemble spread next to the RMSE
of the ensemble mean, which should be about equal for a calibrated ensemble.

Run from the repo root:  python -m plotting.ensemble_evaluation <run_name | path/to/ensemble_samples.pkl> [--lead L]
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MUTED, REF_LINESTYLE
from plotting.evaluation import SEASONS, _as_data_range

SUBSETS = {"water": "all water", "miz": r"marginal ice zone (day $k$)"}
TIE_SEED = 0  # fixes the random tie-breaking, so the figure is reproducible


def load_ensemble(path: str | Path, lead: int = 1):
    """Load an ensemble_samples.pkl, given its path or the run's results directory/name.

    Returns ``(run_dir, data)`` where ``members`` is a list of [M, H, W] arrays and
    ``targets`` and ``k`` lists of [H, W] arrays, all in [0, 1] with land as NaN. For a
    rollout, ``lead`` picks the forecast of k+lead; ``times`` then holds (k+lead, k, k-1).
    """
    path = Path(path)
    if path.suffix != ".pkl":
        path = (path if path.exists() else Path("results") / path) / "ensemble_samples.pkl"

    with open(path, "rb") as f:
        samples, contexts, targets, times, member_seeds = pickle.load(f)

    data = {
        "members": [s[:, lead - 1].numpy() for s in samples],  # samples are [M, lead_months, H, W]
        "targets": [_as_data_range(t)[lead - 1].numpy() for t in targets],
        "k": [_as_data_range(c)[0].numpy() for c in contexts],  # contexts are [2, H, W] = (k, k-1)
        "times": [(d[lead - 1], d[-2], d[-1]) for d in times],  # (k+lead, k, k-1) dates
        "member_seeds": member_seeds,
    }
    return path.parent, data


def subset_mask(target: np.ndarray, k: np.ndarray, subset: str) -> np.ndarray:
    """[H, W] pixels that enter the histogram: water, and for 'miz' 0 < SIC(k) < 1."""
    water = np.isfinite(target)
    if subset == "miz":
        return water & (k > 0) & (k < 1)
    return water


def ranks(members: np.ndarray, target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rank of each target value among its members, in 0..M.

    members: [M, P], target: [P]. A target tied with e members can take any of the e+1
    positions between them, so its rank is drawn uniformly from those.
    """
    below = (members < target).sum(axis=0)
    ties = (members == target).sum(axis=0)
    return below + (rng.random(target.shape) * (ties + 1)).astype(np.int64)


def season_of(date) -> str:
    month = pd.Timestamp(date).month
    return next(season for season, months in SEASONS.items() if month in months)


def rank_statistics(data):
    """Per subset and season ('all', 'cool', 'melt'): rank counts [M+1] and the sums for
    spread-skill. Accumulated one date at a time, so a WEST ensemble fits in memory."""
    n_members = data["members"][0].shape[0]
    rng = np.random.default_rng(TIE_SEED)
    stats = {
        (subset, season): {"counts": np.zeros(n_members + 1, dtype=np.int64), "var": 0.0, "se": 0.0, "n": 0}
        for subset in SUBSETS for season in ("all", *SEASONS)
    }

    for members, target, k, dates in zip(data["members"], data["targets"], data["k"], data["times"]):
        season = season_of(dates[0])
        for subset in SUBSETS:
            mask = subset_mask(target, k, subset)
            m, y = members[:, mask].astype(np.float64), target[mask].astype(np.float64)
            r = ranks(m, y, rng)
            for key in ((subset, "all"), (subset, season)):
                s = stats[key]
                s["counts"] += np.bincount(r, minlength=n_members + 1)
                s["var"] += m.var(axis=0, ddof=1).sum()
                s["se"] += ((m.mean(axis=0) - y) ** 2).sum()
                s["n"] += y.size
    return n_members, stats


def print_spread_skill(n_members, stats):
    """Spread (sqrt of the mean member variance) vs RMSE of the ensemble mean. For a calibrated
    ensemble RMSE ~ sqrt((M+1)/M) * spread, so the ratio should be about 1."""
    factor = np.sqrt((n_members + 1) / n_members)
    header = f"{'subset':<8} {'season':<6} {'pixels':>10} {'spread':>9} {'RMSE':>9} {'RMSE/spread*':>13}"
    print(f"\nSpread-skill, {n_members} members (* divided by sqrt((M+1)/M) = {factor:.3f}; 1 = calibrated)")
    print(header)
    print("-" * len(header))
    for (subset, season), s in stats.items():
        if not s["n"]:
            continue
        spread = np.sqrt(s["var"] / s["n"])
        rmse = np.sqrt(s["se"] / s["n"])
        print(f"{subset:<8} {season:<6} {s['n']:>10} {spread:>9.4f} {rmse:>9.4f} {rmse / spread / factor:>13.2f}")
    print()


def plot_rank_histogram(n_members, stats, out_dir, name):
    """Relative frequency per rank, one panel per subset; bars for all dates, outlines per season."""
    edges = np.arange(n_members + 2) - 0.5
    flat = 1 / (n_members + 1)

    fig, axes = plt.subplots(1, len(SUBSETS), sharey=True)
    for ax, (subset, title) in zip(axes, SUBSETS.items()):
        counts = stats[(subset, "all")]["counts"]
        ax.bar(np.arange(n_members + 1), counts / counts.sum(), width=1.0, color=COLORS[0],
               alpha=0.6, linewidth=0, label="all dates")
        for season, color in zip(SEASONS, COLORS[1:]):
            c = stats[(subset, season)]["counts"]
            if c.sum():
                ax.stairs(c / c.sum(), edges, color=color, linewidth=1.2, label=f"{season} season")
        ax.axhline(flat, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE, label="flat")
        ax.set_title(title)
        ax.set_xlabel("rank of truth among members")
        ax.set_xticks(range(0, n_members + 1, 2))
        ax.set_xlim(edges[0], edges[-1])
    axes[0].set_ylabel("relative frequency")
    axes[0].legend()

    savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=2.8)
    plt.close(fig)


def make_all(path, out_dir: str | None = None, lead: int = 1):
    run_dir, data = load_ensemble(path, lead)
    if out_dir is None:
        out_dir = run_dir if lead == 1 else run_dir / f"lead{lead}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_members, stats = rank_statistics(data)
    print_spread_skill(n_members, stats)
    plot_rank_histogram(n_members, stats, out_dir, "rank_histogram")
    print(f"Wrote {out_dir}/rank_histogram.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under results/, its directory, or an ensemble_samples.pkl path")
    parser.add_argument("--out-dir", default=None, help="where the PDF goes (default: next to the pickle)")
    parser.add_argument("--lead", type=int, default=1,
                        help="for a rollout, score the forecast of k+lead (default out dir: lead<lead>/)")
    args = parser.parse_args()
    make_all(args.run, out_dir=args.out_dir, lead=args.lead)
