"""Rank histogram for ensemble forecasts.

Reads ``results/<run>/ensemble_samples.pkl`` written by ``ensemble.py`` and writes
``rank_histogram.pdf`` next to it: per-pixel rank of the true k+1 SIC among the M members,
pooled over all dates. A calibrated ensemble gives a flat histogram at 1/(M+1); a U shape means
too little spread, a dome too much, a slope a bias.

Only pixels in the marginal ice zone enter, 0 < SIC < 1 on day k. Open water and pack ice
(SIC exactly 0 or 1) are left out: there the truth ties with (nearly) every member, the
random tie-breaking spreads them evenly over the ranks, and since they are most of the
ocean they flatten the histogram whatever the ensemble does at the ice. The zone is
selected on day k, not on the target: conditioning on the observation biases the histogram
(the forecaster's dilemma).

Ties that remain inside the zone are broken at random, as in Hamill (2001); counting only
members strictly below the truth would pile every tie into rank 0 and fake a U shape.

Also prints the spread-skill pair per season: mean ensemble spread next to the RMSE of the
ensemble mean, which should be about equal for a calibrated ensemble.

``spread_skill_map.pdf`` shows the same pair per pixel, one row per season: spread and RMSE
over the season's dates on a shared colour scale, and their ratio (log scale, 1 =
calibrated). Over all ocean pixels, as the map itself shows where the ice is; the ratio is
left out where the spread is below MIN_MAP_SPREAD (open water, pack ice). Spread and RMSE
lining up says the ensemble knows *where* it is uncertain, the ratio whether it knows *how
much*. The per-pixel correlation of spread and RMSE is printed alongside.

Run from the repo root:  python -m plotting.ensemble_evaluation <run_name | path/to/ensemble_samples.pkl> [--lead L]
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MUTED, REF_LINESTYLE
from plotting.evaluation import SEASONS, _as_data_range, _clean_image_axis

TIE_SEED = 0  # fixes the random tie-breaking, so the figure is reproducible

# Spread-skill maps. Below this spread (open water, pack ice) spread and RMSE are both ~0 and
# their ratio is noise, so the ratio map leaves those pixels out.
MIN_MAP_SPREAD = 0.01
RATIO_LIMIT = 4  # ratio colour scale runs from 1/RATIO_LIMIT to RATIO_LIMIT, log, centred at 1
LAND_COLOR = "lightgray"  # as SIC_CMAP in plotting.evaluation

SPREAD_CMAP = plt.cm.viridis.copy()
SPREAD_CMAP.set_bad(LAND_COLOR)
RATIO_CMAP = plt.cm.RdBu_r.copy()
RATIO_CMAP.set_bad(alpha=0)  # transparent, so the background layer shows land vs masked ocean
# Background of the ratio map: masked ocean white, land grey.
BACKGROUND_CMAP = ListedColormap(["white"])
BACKGROUND_CMAP.set_bad(LAND_COLOR)


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


def miz_mask(target: np.ndarray, k: np.ndarray) -> np.ndarray:
    """[H, W] pixels that enter the histogram: water with 0 < SIC(k) < 1."""
    return np.isfinite(target) & (k > 0) & (k < 1)


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
    """Per season ('all', 'cool', 'melt'): rank counts [M+1] and the sums for spread-skill,
    over the marginal ice zone. Accumulated one date at a time, so a WEST ensemble fits in
    memory."""
    n_members = data["members"][0].shape[0]
    rng = np.random.default_rng(TIE_SEED)
    stats = {
        season: {"counts": np.zeros(n_members + 1, dtype=np.int64), "var": 0.0, "se": 0.0, "n": 0}
        for season in ("all", *SEASONS)
    }

    for members, target, k, dates in zip(data["members"], data["targets"], data["k"], data["times"]):
        mask = miz_mask(target, k)
        m, y = members[:, mask].astype(np.float64), target[mask].astype(np.float64)
        r = ranks(m, y, rng)
        for season in ("all", season_of(dates[0])):
            s = stats[season]
            s["counts"] += np.bincount(r, minlength=n_members + 1)
            s["var"] += m.var(axis=0, ddof=1).sum()
            s["se"] += ((m.mean(axis=0) - y) ** 2).sum()
            s["n"] += y.size
    return n_members, stats


def print_spread_skill(n_members, stats):
    """Spread (sqrt of the mean member variance) vs RMSE of the ensemble mean. For a calibrated
    ensemble RMSE ~ sqrt((M+1)/M) * spread, so the ratio should be about 1."""
    factor = np.sqrt((n_members + 1) / n_members)
    header = f"{'season':<6} {'pixels':>10} {'spread':>9} {'RMSE':>9} {'RMSE/spread*':>13}"
    print(f"\nSpread-skill over the marginal ice zone, {n_members} members "
          f"(* divided by sqrt((M+1)/M) = {factor:.3f}; 1 = calibrated)")
    print(header)
    print("-" * len(header))
    for season, s in stats.items():
        if not s["n"]:
            continue
        spread = np.sqrt(s["var"] / s["n"])
        rmse = np.sqrt(s["se"] / s["n"])
        print(f"{season:<6} {s['n']:>10} {spread:>9.4f} {rmse:>9.4f} {rmse / spread / factor:>13.2f}")
    print()


def plot_rank_histogram(n_members, stats, out_dir, name):
    """Relative frequency per rank over the marginal ice zone; bars for all dates, outlines
    per season."""
    edges = np.arange(n_members + 2) - 0.5
    flat = 1 / (n_members + 1)

    fig, ax = plt.subplots()
    counts = stats["all"]["counts"]
    ax.bar(np.arange(n_members + 1), counts / counts.sum(), width=1.0, color=COLORS[0],
           alpha=0.6, linewidth=0, label="all dates")
    for season, color in zip(SEASONS, COLORS[1:]):
        c = stats[season]["counts"]
        if c.sum():
            ax.stairs(c / c.sum(), edges, color=color, linewidth=1.2, label=f"{season} season")
    ax.axhline(flat, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE, label="flat")
    ax.set_xlabel("rank of truth among members")
    ax.set_ylabel("relative frequency")
    ax.set_xticks(range(0, n_members + 1, 2))
    ax.set_xlim(edges[0], edges[-1])
    ax.legend()

    savefig(fig, out_dir, name, width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def spread_skill_maps(data):
    """Per season ('cool', 'melt'): per-pixel spread, RMSE of the ensemble mean and their
    ratio over the dates of that season, each [H, W] with land NaN.

    Over all ocean pixels rather than the day-k marginal ice zone: the map itself shows where
    the ice is, and a per-date mask would give each pixel a different number of dates. The
    ratio is divided by sqrt((M+1)/M) as in print_spread_skill, so 1 = calibrated, and is NaN
    where the spread is below MIN_MAP_SPREAD.
    """
    n_members = data["members"][0].shape[0]
    shape = data["targets"][0].shape
    sums = {season: {"var": np.zeros(shape), "se": np.zeros(shape), "n": np.zeros(shape)}
            for season in SEASONS}

    for members, target, dates in zip(data["members"], data["targets"], data["times"]):
        s = sums[season_of(dates[0])]
        m = members.astype(np.float64)
        water = np.isfinite(target)
        s["var"][water] += m.var(axis=0, ddof=1)[water]
        s["se"][water] += ((m.mean(axis=0) - target) ** 2)[water]
        s["n"] += water

    factor = np.sqrt((n_members + 1) / n_members)
    maps = {}
    for season, s in sums.items():
        if not s["n"].any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):  # land: 0 / 0
            spread = np.sqrt(s["var"] / s["n"])
            rmse = np.sqrt(s["se"] / s["n"])
            ratio = np.where(spread >= MIN_MAP_SPREAD, rmse / spread / factor, np.nan)
        maps[season] = {"spread": spread, "rmse": rmse, "ratio": ratio}
    return maps


def print_map_correlation(maps):
    """Pearson correlation of per-pixel spread and RMSE, over pixels with spread >=
    MIN_MAP_SPREAD: does the ensemble know where it is uncertain? The open ocean, where both
    are ~0, is left out, as it would inflate the correlation."""
    print(f"Per-pixel spread vs RMSE correlation (pixels with spread >= {MIN_MAP_SPREAD})")
    for season, m in maps.items():
        keep = m["spread"] >= MIN_MAP_SPREAD
        r = np.corrcoef(m["spread"][keep], m["rmse"][keep])[0, 1]
        print(f"{season:<6} {keep.sum():>9} pixels  r = {r:.2f}")
    print()


def plot_spread_skill_map(maps, out_dir, name):
    """One row per season: spread and RMSE on a shared colour scale, and their ratio
    (red: RMSE > spread, overconfident; blue: underconfident; white: spread < MIN_MAP_SPREAD)."""
    # Shared scale, so equal colours are equal SIC errors in both maps and both seasons.
    vmax = np.nanpercentile(np.concatenate([m[k].ravel() for m in maps.values()
                                            for k in ("spread", "rmse")]), 99)
    ratio_norm = LogNorm(1 / RATIO_LIMIT, RATIO_LIMIT)

    n = len(maps)
    fig = plt.figure()
    # Colorbars horizontal below the maps: vertical ones between the columns would widen every
    # gap, as tight_layout spaces all columns alike.
    gs = fig.add_gridspec(n + 1, 3, height_ratios=[1] * n + [0.05])
    titles = ("spread", "RMSE", r"RMSE / spread$^*$")
    for row, (season, m) in enumerate(maps.items()):
        axes = [fig.add_subplot(gs[row, col]) for col in range(3)]
        img = axes[0].imshow(m["spread"], cmap=SPREAD_CMAP, vmin=0, vmax=vmax, origin="lower",
                             interpolation="none")
        axes[1].imshow(m["rmse"], cmap=SPREAD_CMAP, vmin=0, vmax=vmax, origin="lower",
                       interpolation="none")
        axes[2].imshow(np.where(np.isfinite(m["spread"]), 0.0, np.nan), cmap=BACKGROUND_CMAP,
                       origin="lower", interpolation="none")
        img_ratio = axes[2].imshow(m["ratio"], cmap=RATIO_CMAP, norm=ratio_norm, origin="lower",
                                   interpolation="none")
        for ax, title in zip(axes, titles):
            _clean_image_axis(ax)
            if row == 0:
                ax.set_title(title)
        axes[0].set_ylabel(f"{season} season")

    fig.colorbar(img, cax=fig.add_subplot(gs[n, :2]), orientation="horizontal",
                 extend="max").set_label("SIC")
    cbar = fig.colorbar(img_ratio, cax=fig.add_subplot(gs[n, 2]), orientation="horizontal",
                        extend="both")
    cbar.set_ticks([1 / 4, 1 / 2, 1, 2, 4], labels=["1/4", "1/2", "1", "2", "4"])
    cbar.minorticks_off()
    cbar.set_label("ratio")

    savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=1.9 * n + 0.7)
    plt.close(fig)


def make_all(path, out_dir: str | None = None, lead: int = 1):
    run_dir, data = load_ensemble(path, lead)
    if out_dir is None:
        out_dir = run_dir if lead == 1 else run_dir / f"lead{lead}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_members, stats = rank_statistics(data)
    print_spread_skill(n_members, stats)
    plot_rank_histogram(n_members, stats, out_dir, "rank_histogram")
    maps = spread_skill_maps(data)
    print_map_correlation(maps)
    plot_spread_skill_map(maps, out_dir, "spread_skill_map")
    print(f"Wrote {out_dir}/rank_histogram.pdf and spread_skill_map.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under results/, its directory, or an ensemble_samples.pkl path")
    parser.add_argument("--out-dir", default=None, help="where the PDF goes (default: next to the pickle)")
    parser.add_argument("--lead", type=int, default=1,
                        help="for a rollout, score the forecast of k+lead (default out dir: lead<lead>/)")
    args = parser.parse_args()
    make_all(args.run, out_dir=args.out_dir, lead=args.lead)
