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
3. ``<OUT_DIR>/ensemble_season_maps_<key>.pdf``, one figure per ensemble (``FIGURE_KEYS``):
   rows melt and cool, columns the seasonal mean true residual, the mean predicted residual
   (ensemble mean - k), the ensemble spread and the RMSE of the ensemble mean. Spread and RMSE
   are the same on SIC and on the residual, as k is shared by every member and the truth.
   The residual columns use the fixed RESIDUAL_VMAX scale of the other mean-residual figures;
   spread and RMSE share one scale across both figures, so equal colours are equal SIC errors
   across models, seasons and columns.
4. ``<OUT_DIR>/ensemble_case_<CASE_DATE>.pdf``: one forecast, the one whose target is
   ``CASE_DATE``. One row per ensemble in ``CASE_RUNS``: the true residual, the ensemble-mean
   residual, the ensemble spread and the absolute error of the ensemble mean, with the spread
   and RMSE over water written under the last two. The dark lines inside the error maps are
   where the signed error changes sign (|error| passes through 0), mostly near the ice edge.
5. ``<OUT_DIR>/ensemble_spread_skill.pdf``: binned spread-skill diagram, one line per ensemble.
   The marginal-ice-zone pixels of all dates (as the rank histograms) are sorted by their
   member variance and split into SPREAD_SKILL_BINS equal-count bins; each point is the RMSE
   of the ensemble mean (x) against the RMS spread (y) of one bin. The spread is scaled by
   sqrt((M+1)/M), so a calibrated M-member ensemble lies on the dashed 1:1 line for any M;
   below it the ensemble is under-dispersive, above it over-dispersive.

Both ensembles are scored on the target dates they share, as in ``plotting.final_results``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MARKERS, MUTED, REF_LINESTYLE, muted
from plotting.ensemble_evaluation import (LAND_COLOR, SPREAD_SKILL_BINS, draw_rank_histogram,
                                          print_map_correlation, print_spread_skill,
                                          rank_statistics, spread_skill_curve, spread_skill_maps)
from plotting.evaluation import (RESIDUAL_CMAP, RESIDUAL_VMAX, _clean_image_axis, load_leads,
                                 mean_residuals, sample_errors)
from plotting.final_results import _cell, _climatology, _subset

# Folder names under results/. U-Net is the top row / left panel, DDPM the bottom row / right.
UNET_RUN = "deep-ensemble2"
DDPM_RUN = "ddpm-LC-256-ensemble-v3"
RUNS = [UNET_RUN, DDPM_RUN]
# Display names (figure titles and row labels, tables).
LABELS = {
    UNET_RUN: "U-Net deep ensemble",
    DDPM_RUN: "DDPM ensemble",
}
# File-name suffix of each ensemble's season-map figure.
FIGURE_KEYS = {UNET_RUN: "unet", DDPM_RUN: "ddpm"}
LEAD = 1                   # lead time to score
OUT_DIR = "final_figures"  # where the figures are written

GROUPS = ("all", "cool", "melt")  # table columns
# Season maps: rows in this order. Spread and RMSE share one colour map and scale.
MAP_SEASONS = ("melt", "cool")
MAP_CMAP = plt.cm.viridis.copy()
MAP_CMAP.set_bad(LAND_COLOR)
MAP_VMAX_PERCENTILE = 99  # spread/RMSE colour limit over all panels of both figures
# Residual maps: RESIDUAL_CMAP (RdBu) blended RESIDUAL_SOFTEN of the way to white, so the reds
# and blues are less saturated. White (no change) stays white.
RESIDUAL_SOFTEN = 0.25
SOFT_RESIDUAL_CMAP = ListedColormap(
    [muted(c, RESIDUAL_SOFTEN) for c in RESIDUAL_CMAP(np.linspace(0, 1, 256))])
SOFT_RESIDUAL_CMAP.set_bad(LAND_COLOR)

# Single-date figure: the forecast whose target (k+1) is CASE_DATE, one row per run in CASE_RUNS.
CASE_DATE = "2024-06-27"  # melt season; k = 2024-05-27
CASE_RUNS = [DDPM_RUN, UNET_RUN]
CASE_ROW_LABELS = {UNET_RUN: "U-Net", DDPM_RUN: "DDPM"}
CASE_RESIDUAL_CMAP = SOFT_RESIDUAL_CMAP     # residuals: white at 0
CASE_ERROR_CMAP = plt.cm.magma.copy()       # spread and absolute error
CASE_ERROR_CMAP.set_bad(LAND_COLOR)
CASE_VMAX_PERCENTILE = 99  # colour limits, over water; the top 1 % saturate

def _name(run: str) -> str:
    return LABELS.get(run, run)


def score_run(run: str, dates=None):
    """Load one ensemble and keep only what the tables and figures need, so the (large)
    pickle can be freed before the next one is loaded.

    With ``dates`` (a set of target dates), only the samples on those dates are scored.
    Returns a dict with the ``sample_errors`` DataFrame, ``n_members``, the rank/spread-skill
    ``stats``, the per-season spread/RMSE ``maps``, the per-season mean predicted and true
    ``residuals`` (``plotting.evaluation.mean_residuals``), the scored target ``dates`` and
    the ``case`` fields of the CASE_DATE forecast (``case_fields``; None if it is not scored).
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
        "residuals": mean_residuals(data),
        "dates": [pd.Timestamp(d[0]) for d in data["times"]],
        "case": case_fields(data, pd.Timestamp(CASE_DATE)),
        "curve": spread_skill_curve(data, n_members),
    }


def case_fields(data, date):
    """The forecast whose target is ``date``, as [H, W] fields with land NaN: ``true`` and
    ``mean`` residuals (truth and ensemble mean, minus k), member ``spread`` and the absolute
    ``error`` of the ensemble mean. None if no sample has that target date."""
    index = next((i for i, d in enumerate(data["times"]) if pd.Timestamp(d[0]) == date), None)
    if index is None:
        return None
    members = data["members"][index].astype(np.float64)
    target, k = data["targets"][index], data["k"][index]
    mean = members.mean(axis=0)
    return {
        "true": target - k,
        "mean": mean - k,
        "spread": members.std(axis=0, ddof=1),
        "error": np.abs(mean - target),
        "k_date": pd.Timestamp(data["times"][index][1]),
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


def plot_season_maps(scores, out_dir):
    """One figure per ensemble. Rows: the seasons in MAP_SEASONS. Columns: mean true residual,
    mean predicted residual, spread, RMSE. A horizontal colorbar under each column pair."""
    seasons = [season for season in MAP_SEASONS
               if all(season in s["maps"] and season in s["residuals"] for s in scores.values())]
    if not seasons:
        print("No season with samples in every ensemble -- skipping ensemble_season_maps.")
        return

    # Shared over both figures; values above vmax (the top 1 %) take the brightest colour.
    err_vmax = np.nanpercentile(np.concatenate(
        [s["maps"][season][key].ravel() for s in scores.values() for season in seasons
         for key in ("spread", "rmse")]), MAP_VMAX_PERCENTILE)
    res_vmax = RESIDUAL_VMAX

    titles = ("True mean\nresidual", "Predicted mean\nresidual", "Spread", "RMSE")
    for run, s in scores.items():
        fig = plt.figure()
        gs = fig.add_gridspec(len(seasons) + 1, len(titles),
                              height_ratios=[1] * len(seasons) + [0.05])
        res_img = err_img = None
        for row, season in enumerate(seasons):
            pred, true, n = s["residuals"][season]
            fields = (true, pred, s["maps"][season]["spread"], s["maps"][season]["rmse"])
            for col, field in enumerate(fields):
                ax = fig.add_subplot(gs[row, col])
                if col < 2:
                    res_img = ax.imshow(field, cmap=SOFT_RESIDUAL_CMAP, vmin=-res_vmax, vmax=res_vmax,
                                        origin="lower", interpolation="none")
                else:
                    err_img = ax.imshow(field, cmap=MAP_CMAP, vmin=0, vmax=err_vmax,
                                        origin="lower", interpolation="none")
                _clean_image_axis(ax)
                if row == 0:
                    ax.set_title(titles[col])
                if col == 0:
                    ax.set_ylabel(f"{season.capitalize()}, $n={n}$")

        # Both colorbars alike: no extension arrows, no outline.
        for img, cols, label in ((res_img, slice(0, 2), "SIC change"), (err_img, slice(2, 4), "SIC")):
            cbar = fig.colorbar(img, cax=fig.add_subplot(gs[-1, cols]), orientation="horizontal")
            cbar.outline.set_visible(False)
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            cbar.set_label(label)

        # Square panels: the width per map column (less the gaps) sets the height per row,
        # plus room for the two-line titles and the colorbars.
        panel = 0.8 * TEXTWIDTH_IN / len(titles)
        name = f"ensemble_season_maps_{FIGURE_KEYS.get(run, run)}"
        savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=(panel + 0.05) * len(seasons) + 0.9)
        plt.close(fig)
        print(f"Wrote {out_dir}/{name}.pdf")


def plot_case_maps(scores, out_dir):
    """The CASE_DATE forecast. One row per run in CASE_RUNS: true residual, ensemble-mean
    residual, spread, absolute error of the ensemble mean. Residuals on soft RdBu (white at 0),
    spread and error on magma; scales shared by the rows. The spread and RMSE over water are
    the titles of each row's spread and error maps."""
    runs = [run for run in CASE_RUNS if scores[run]["case"] is not None]
    if not runs:
        print(f"No forecast with target {CASE_DATE} -- skipping ensemble_case.")
        return
    cases = {run: scores[run]["case"] for run in runs}
    first = cases[runs[0]]
    water = np.isfinite(first["true"])

    res_vmax = np.nanpercentile(np.abs(first["true"][water]), CASE_VMAX_PERCENTILE)
    err_vmax = np.nanpercentile(np.concatenate(
        [c[key][water] for c in cases.values() for key in ("spread", "error")]),
        CASE_VMAX_PERCENTILE)

    titles = ("True\nresidual", "Ensemble mean\nresidual")  # spread/error: titled by value
    keys = ("true", "mean", "spread", "error")
    fig = plt.figure()
    gs = fig.add_gridspec(len(runs) + 1, len(keys), height_ratios=[1] * len(runs) + [0.05])

    print(f"Forecast {first['k_date']:%Y-%m-%d} -> {CASE_DATE}, over {water.sum()} water pixels:")
    res_img = err_img = None
    for row, run in enumerate(runs):
        c = cases[run]
        spread = np.sqrt(np.mean(c["spread"][water] ** 2))
        rmse = np.sqrt(np.mean(c["error"][water] ** 2))
        r = np.corrcoef(c["spread"][water], c["error"][water])[0, 1]
        print(f"  {_name(run):<22} spread {spread:.4f}  RMSE {rmse:.4f}  "
              f"per-pixel spread/error r = {r:.2f}")
        for col, key in enumerate(keys):
            ax = fig.add_subplot(gs[row, col])
            if key in ("true", "mean"):
                res_img = ax.imshow(c[key], cmap=CASE_RESIDUAL_CMAP, vmin=-res_vmax, vmax=res_vmax,
                                    origin="lower", interpolation="none")
            else:
                err_img = ax.imshow(c[key], cmap=CASE_ERROR_CMAP, vmin=0, vmax=err_vmax,
                                    origin="lower", interpolation="none")
            _clean_image_axis(ax)
            if key == "spread":
                ax.set_title(rf"Spread $= {spread:.3f}$")
            elif key == "error":
                ax.set_title(rf"RMSE $= {rmse:.3f}$")
            elif row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.set_ylabel(CASE_ROW_LABELS.get(run, _name(run)))
    print()

    for img, cols, label in ((res_img, slice(0, 2), "SIC change"),
                             (err_img, slice(2, 4), "Ensemble std / RMSE")):
        cbar = fig.colorbar(img, cax=fig.add_subplot(gs[-1, cols]), orientation="horizontal")
        cbar.outline.set_visible(False)
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        cbar.set_label(label)

    # Square maps: the width per map column (less the gaps) sets the height per row, plus the
    # spread/RMSE title above it; then the two-line titles and the colorbars.
    panel = 0.82 * TEXTWIDTH_IN / len(keys)
    name = f"ensemble_case_{CASE_DATE}"
    savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=(panel + 0.25) * len(runs) + 1.0)
    plt.close(fig)
    print(f"Wrote {out_dir}/{name}.pdf")


def plot_spread_skill(scores, out_dir):
    """Binned spread-skill diagram: RMSE of the ensemble mean (x) vs finite-M corrected
    spread (y), one line per ensemble, with the dashed 1:1 line of perfect calibration."""
    fig, ax = plt.subplots()
    top = 0.0
    print(f"Spread-skill bins ({SPREAD_SKILL_BINS}, marginal ice zone): RMSE / spread per bin")
    for (run, s), color, marker in zip(scores.items(), COLORS, MARKERS):
        rmse, spread = s["curve"]
        ax.plot(rmse, spread, color=color, marker=marker, markersize=4, label=_name(run))
        top = max(top, rmse.max(), spread.max())
        print(f"  {_name(run):<22} " + " ".join(f"{r:.2f}" for r in rmse / spread))
    print()

    top *= 1.05
    ax.plot([0, top], [0, top], color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE,
            label="perfect calibration")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_aspect("equal")
    ax.set_xlabel("RMSE of ensemble mean")
    ax.set_ylabel("Ensemble spread")
    ax.legend(loc="upper left")
    savefig(fig, out_dir, "ensemble_spread_skill", width=TEXTWIDTH_IN / 2, height=3.0)
    plt.close(fig)


def main():
    scores = load_all(RUNS)
    print_accuracy_tables(scores)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_rank_histograms(scores, out_dir)
    print(f"Wrote {out_dir}/ensemble_rank_histograms.pdf")
    plot_season_maps(scores, out_dir)
    plot_case_maps(scores, out_dir)
    plot_spread_skill(scores, out_dir)
    print(f"Wrote {out_dir}/ensemble_spread_skill.pdf")


if __name__ == "__main__":
    main()
