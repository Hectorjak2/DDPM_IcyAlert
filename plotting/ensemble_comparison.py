"""Side-by-side comparison of the U-Net deep ensemble and the DDPM ensemble.

Edit ``UNET_RUN`` / ``DDPM_RUN`` (folder names under ``results/``, as in
``plotting.final_results``) and optionally ``LABELS`` below, then run from the repo root:

    python -m plotting.ensemble_comparison

1. Prints accuracy tables, mean ± std over the samples, for all samples and per season
   (cool Oct--Mar, melt Apr--Sep): the RMSE of the ensemble mean and of a single member
   (mean over members) next to persistence and climatology, and the fair CRPS next to the
   baselines' MAE (their CRPS). Then the spread-skill pair over the marginal ice zone and
   the per-pixel spread/RMSE correlation per ensemble (as ``plotting.ensemble_evaluation``).
2. ``<OUT_DIR>/ensemble_rank_histograms.pdf``: the rank histograms of
   ``plotting.ensemble_evaluation``, U-Net (left) and DDPM (right), on a shared y axis.
3. ``<OUT_DIR>/ensemble_spread_skill_map.pdf``: 2x4 grid, rows spread and RMSE, columns
   U-Net melt, U-Net cool, DDPM melt, DDPM cool, with a bold model label above each pair.
   One colour map (inferno) and scale for every panel, so equal colours are equal SIC errors
   across models, seasons, spread and RMSE.

Both ensembles are scored on the target dates they share, as in ``plotting.final_results``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from plotting.config import savefig, TEXTWIDTH_IN, MUTED
from plotting.ensemble_evaluation import (LAND_COLOR, draw_rank_histogram, print_map_correlation,
                                          print_spread_skill, rank_statistics, spread_skill_maps)
from plotting.evaluation import _clean_image_axis, load_leads, sample_errors
from plotting.final_results import _cell, _climatology, _subset

# Folder names under results/. U-Net is the top row / left panel, DDPM the bottom row / right.
UNET_RUN = "deep-ensemble"
DDPM_RUN = "ddpm-LC-256-ensemble"
RUNS = [UNET_RUN, DDPM_RUN]
# Display names (figure titles and row labels, tables).
LABELS = {
    UNET_RUN: "U-Net deep ensemble",
    DDPM_RUN: "DDPM ensemble",
}
# Short bold labels above each ensemble's column pair in the spread-skill map.
GROUP_LABELS = {UNET_RUN: "U-Net", DDPM_RUN: "DDPM"}
LEAD = 1             # lead time to score
OUT_DIR = "figures"  # where the figures are written

GROUPS = ("all", "cool", "melt")  # table columns
# Spread-skill map: season columns in this order within each ensemble's pair. One colour map
# and scale for every panel; inferno starts black at 0, so the lightgray land stands out.
MAP_SEASONS = ("melt", "cool")
MAP_CMAP = plt.cm.magma.copy()
MAP_CMAP.set_bad(LAND_COLOR)
MAP_VMAX_PERCENTILE = 99  # colour limit over all panels, as in plot_spread_skill_map


def _name(run: str) -> str:
    return LABELS.get(run, run)


def score_run(run: str, dates=None):
    """Load one ensemble and keep only what the tables and figures need, so the (large)
    pickle can be freed before the next one is loaded.

    With ``dates`` (a set of target dates), only the samples on those dates are scored.
    Returns a dict with the ``sample_errors`` DataFrame, ``n_members``, the rank/spread-skill
    ``stats``, the per-season spread/RMSE ``maps`` and the scored target ``dates``.
    """
    _, leads = load_leads(run)
    if len(leads) < LEAD:
        raise ValueError(f"{run} has {len(leads)} lead time(s), LEAD = {LEAD}.")
    data = leads[LEAD - 1]
    del leads
    if dates is not None:
        data = _subset(data, [i for i, d in enumerate(data["times"])
                              if pd.Timestamp(d[0]) in dates])
    n_members, stats = rank_statistics(data)
    if n_members < 2:
        raise ValueError(f"{run} has a single member -- not an ensemble.")
    return {
        "errors": sample_errors(data, _climatology(data["targets"][0].shape)),
        "n_members": n_members,
        "stats": stats,
        "maps": spread_skill_maps(data),
        "dates": [pd.Timestamp(d[0]) for d in data["times"]],
    }


def load_all(runs):
    """Score every run, one at a time, on the target dates they all share. Returns
    ``{run: score_run(...)}`` in ``runs`` order."""
    scores = {}
    for run in runs:
        print(f"Loading {run} ...")
        scores[run] = score_run(run)

    common = set.intersection(*(set(s["dates"]) for s in scores.values()))
    if not common:
        raise ValueError("The ensembles share no target dates -- nothing to compare.")
    for run, s in scores.items():
        if set(s["dates"]) != common:
            print(f"{run} has {len(set(s['dates']))} dates, {len(common)} shared with all runs "
                  f"-- reloading it on the shared dates only.")
            scores[run] = score_run(run, common)
    return scores


def _print_table(title, rows, reference):
    """``rows``: (label, per-sample DataFrame, column, transform). One cell per group in
    GROUPS: mean ± std over the samples of transform(column)."""
    width = 19
    name_width = max(len(label) for label, *_ in rows) + 2
    headers = []
    for group in GROUPS:
        n = len(reference) if group == "all" else int((reference.season == group).sum())
        headers.append(f"{group} (n={n})")
    header = f"{'':<{name_width}}" + "".join(f"{h:>{width}}" for h in headers)

    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for label, errors, column, transform in rows:
        cells = []
        for group in GROUPS:
            subset = errors if group == "all" else errors[errors.season == group]
            cells.append(_cell(transform(subset[column])) if len(subset) else "-")
        print(f"{label:<{name_width}}" + "".join(f"{c:>{width}}" for c in cells))


def print_accuracy_tables(scores):
    """RMSE (ensemble mean and single member) and CRPS, next to the baselines. The baselines
    depend only on the dates, so they come from the first run."""
    reference = next(iter(scores.values()))["errors"]
    has_climatology = reference.climatology.notna().any()
    identity = lambda x: x  # noqa: E731

    rows = []
    if has_climatology:
        rows.append(("Climatology", reference, "climatology", np.sqrt))
    rows.append(("Persistence", reference, "persistence", np.sqrt))
    rows += [(f"{_name(run)}, mean", s["errors"], "mse", np.sqrt) for run, s in scores.items()]
    rows += [(f"{_name(run)}, member", s["errors"], "member_mse", np.sqrt)
             for run, s in scores.items()]
    _print_table(f"RMSE over water (lead {LEAD}), per sample: mean ± std "
                 f"(mean = ensemble mean, member = single member averaged over members)",
                 rows, reference)

    rows = []
    if has_climatology:
        rows.append(("Climatology (MAE)", reference, "climatology_mae", identity))
    rows.append(("Persistence (MAE)", reference, "persistence_mae", identity))
    rows += [(_name(run), s["errors"], "crps", identity) for run, s in scores.items()]
    _print_table(f"Fair CRPS over water (lead {LEAD}), per sample: mean ± std "
                 f"(baselines: MAE, their CRPS)", rows, reference)
    if not has_climatology:
        print("(no climatology for this grid -- rows left out)")
    print()

    for run, s in scores.items():
        print(f"== {_name(run)} ==", end="")
        print_spread_skill(s["n_members"], s["stats"])
        print_map_correlation(s["maps"])


def plot_rank_histograms(scores, out_dir):
    """Rank histograms side by side, one per ensemble, on a shared y axis."""
    fig, axes = plt.subplots(1, len(scores), sharey=True, squeeze=False)
    for col, (ax, (run, s)) in enumerate(zip(axes[0], scores.items())):
        draw_rank_histogram(ax, s["n_members"], s["stats"], ylabel=col == 0, legend=col == 0)
        ax.set_title(f"{_name(run)} ($M = {s['n_members']}$)")
    savefig(fig, out_dir, "ensemble_rank_histograms", width=TEXTWIDTH_IN, height=2.8)
    plt.close(fig)


def _group_label(ax, text):
    """Bold ``text`` over a rule across ``ax``, an axes spanning one ensemble's columns. A
    row of its own, rather than an annotation on the top maps, so tight_layout does not
    widen every row gap to make room for it."""
    ax.axis("off")
    ax.text(0.5, 0.35, rf"\textbf{{{text}}}", ha="center", va="bottom", transform=ax.transAxes)
    ax.plot([0.02, 0.98], [0.2, 0.2], color=MUTED, linewidth=0.8, transform=ax.transAxes,
            clip_on=False)


def plot_spread_skill_grid(scores, out_dir):
    """Rows: spread and RMSE. Columns: one pair per ensemble (bold label above), one column
    per season in MAP_SEASONS within each pair. One colour map and scale for every panel,
    with a single horizontal colorbar under the maps."""
    seasons = [season for season in MAP_SEASONS
               if all(season in s["maps"] for s in scores.values())]
    if not seasons:
        print("No season with samples in every ensemble -- skipping ensemble_spread_skill_map.")
        return

    # Values above vmax (the top 1 %) are clipped to the brightest colour.
    vmax = np.nanpercentile(np.concatenate(
        [s["maps"][season][key].ravel() for s in scores.values() for season in seasons
         for key in ("spread", "rmse")]), MAP_VMAX_PERCENTILE)

    rows = (("spread", "Spread"), ("rmse", "RMSE"))
    n_seasons = len(seasons)
    # One block of season columns per ensemble, with a narrow empty column between blocks.
    width_ratios = []
    for i in range(len(scores)):
        width_ratios += ([0.06] if i else []) + [1] * n_seasons
    fig = plt.figure()
    # Rows: group labels, the maps, the colorbar.
    gs = fig.add_gridspec(len(rows) + 2, len(width_ratios), width_ratios=width_ratios,
                          height_ratios=[0.08] + [1] * len(rows) + [0.05])

    img = None
    for i, (run, s) in enumerate(scores.items()):
        first = i * (n_seasons + 1)  # gridspec column of this ensemble's first season
        _group_label(fig.add_subplot(gs[0, first:first + n_seasons]),
                     GROUP_LABELS.get(run, _name(run)))
        for j, season in enumerate(seasons):
            col = first + j
            for row, (key, row_label) in enumerate(rows, start=1):
                ax = fig.add_subplot(gs[row, col])
                img = ax.imshow(s["maps"][season][key], cmap=MAP_CMAP, vmin=0, vmax=vmax,
                                origin="lower", interpolation="none")
                _clean_image_axis(ax)
                if row == 1:
                    ax.set_title(season.capitalize())
                if col == 0:
                    ax.set_ylabel(row_label)

    # One colorbar under all maps: no extension arrow, no outline.
    cbar = fig.colorbar(img, cax=fig.add_subplot(gs[-1, :]), orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    cbar.set_label("SIC")

    # Square panels: the width per map column (less tight_layout's padding) sets the height
    # per row, plus room for the group labels, titles and the colorbar.
    panel = 0.95 * TEXTWIDTH_IN / sum(width_ratios)
    savefig(fig, out_dir, "ensemble_spread_skill_map", width=TEXTWIDTH_IN,
            height=panel * len(rows) + 1.0)
    plt.close(fig)


def main():
    scores = load_all(RUNS)
    print_accuracy_tables(scores)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rank_histograms(scores, out_dir)
    plot_spread_skill_grid(scores, out_dir)
    print(f"Wrote {out_dir}/ensemble_rank_histograms.pdf and ensemble_spread_skill_map.pdf")


if __name__ == "__main__":
    main()
