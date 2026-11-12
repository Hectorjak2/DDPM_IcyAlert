"""Final-results comparison of any number of sampled runs.

Edit ``RUNS`` (folder names under ``results/``) and optionally ``LABELS`` below, then run
from the repo root:

    python -m plotting.final_results

1. Prints the per-sample RMSE over water, mean ± std over the samples, for all samples and
   per season (cool Oct--Mar, melt Apr--Sep). Climatology and persistence come first, then
   one row per run in ``RUNS`` order.
2. Writes ``<OUT_DIR>/mean_residual_comparison.pdf``: the mean residual k+1 - k per season
   (rows: cool, melt), one column per run and the true residual in the last column, all on
   one symmetric colour scale.

Everything (runs, baselines, truth) is scored on the target dates that all runs share, so
every row covers the same samples. Runs sampled with the same ``TrainConfig.sample_seed``
share their first dates, e.g. a 13-date run is a subset of a 100-date one. A run with extra
dates is loaded a second time to drop them, and the script prints that it did.
"""
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import (RESIDUAL_CMAP, RESIDUAL_VMAX, _clean_image_axis,
                                 _load_climatology, load_leads, mean_residuals, sample_errors)

# Folder names under results/, in the order they appear in the table and the grid.
RUNS = [
    "ddpm-corrected-land-64",
    "ddpm-corrected-land-64-ip",
    "ddpm-corrected-land-128",
]
# Optional display names; a run not listed here is shown by its folder name.
LABELS = {
    "ddpm-corrected-land-64": "Shift (64)",
    "ddpm-corrected-land-64-ip": "Shift (64) IP",
    "ddpm-corrected-land-128": "Shift (128)",
    "ddpm-corrected-land-128-ip": "Shift (128) IP"
}
LEAD = 1             # lead time to score; every run needs at least this many leads
OUT_DIR = "figures"  # where the figure is written

GROUPS = ("all", "cool", "melt")  # table columns
GRID_SEASONS = ("cool", "melt")   # grid rows


def _name(run: str) -> str:
    """Display name for a run, for the printed tables."""
    return LABELS.get(run, run)


def _tex_name(run: str) -> str:
    """Display name for a run, safe for usetex."""
    label = _name(run)
    for char in "_%&#":
        label = label.replace(char, "\\" + char)
    return label


@cache
def _climatology(shape):
    """The climatology for a grid shape, loaded once (None if none is fitted)."""
    return _load_climatology(shape)


def _subset(data, keep):
    """``data`` (one lead of ``load_leads``) restricted to the sample indices in ``keep``."""
    out = {key: [value[i] for i in keep] for key, value in data.items()
           if key != "sampling_losses"}
    losses = data["sampling_losses"]
    out["sampling_losses"] = None if losses is None else losses[keep]
    return out


def load_run(run: str, dates=None):
    """Load one run and keep only what the table and grid need, so the (large) pickle can
    be freed before the next one is loaded.

    With ``dates`` (a set of target dates), only the samples on those dates are scored.
    Returns ``(errors, residuals, run_dates)``: the ``sample_errors`` DataFrame (with the
    baselines), the ``mean_residuals`` dict and the target dates of the scored samples.
    """
    _, leads = load_leads(run)
    if len(leads) < LEAD:
        raise ValueError(f"{run} has {len(leads)} lead time(s), LEAD = {LEAD}.")
    data = leads[LEAD - 1]
    del leads
    if dates is not None:
        data = _subset(data, [i for i, d in enumerate(data["times"])
                              if pd.Timestamp(d[0]) in dates])
    errors = sample_errors(data, _climatology(data["targets"][0].shape))
    residuals = mean_residuals(data)
    run_dates = [pd.Timestamp(d[0]) for d in data["times"]]
    return errors, residuals, run_dates


def load_all(runs):
    """Load every run, one at a time, and score all of them on the dates they share.

    Returns a list of ``(run, errors, residuals)``. The baselines depend only on the dates, so they are the same in every
    ``errors``; the table and the truth column take them from the first run.
    """
    loaded = []  # (run, errors, residuals, dates)
    for run in runs:
        print(f"Loading {run} ...")
        loaded.append((run, *load_run(run)))

    common = set.intersection(*(set(dates) for *_, dates in loaded))
    if not common:
        raise ValueError("The runs share no target dates -- nothing to compare.")

    results = []
    for run, errors, residuals, dates in loaded:
        if set(dates) != common:
            print(f"{run} has {len(set(dates))} dates, {len(common)} shared with all runs "
                  f"-- reloading it on the shared dates only.")
            errors, residuals, _ = load_run(run, common)
        results.append((run, errors, residuals))
    return results


def _cell(rmse: pd.Series) -> str:
    return f"{rmse.mean():.4f} ± {rmse.std(ddof=0):.4f}"


def print_rmse_table(results):
    """Per-sample RMSE over water, mean ± std, for all samples and per season."""
    reference = results[0][1]
    # sample_errors leaves the column NaN when no climatology is fitted for the grid.
    has_climatology = reference.climatology.notna().any()
    rows = []  # (label, per-sample errors DataFrame, column holding the MSE)
    if has_climatology:
        rows.append(("Climatology", reference, "climatology"))
    rows.append(("Persistence", reference, "persistence"))
    rows += [(_name(run), errors, "mse") for run, errors, _ in results]

    width = 19
    name_width = max(len(label) for label, _, _ in rows) + 2
    headers = []
    for group in GROUPS:
        n = len(reference) if group == "all" else int((reference.season == group).sum())
        headers.append(f"{group} (n={n})")
    header = f"{'':<{name_width}}" + "".join(f"{h:>{width}}" for h in headers)

    print(f"\nRMSE over water (lead {LEAD}), per sample: mean ± std")
    print(header)
    print("-" * len(header))
    for label, errors, column in rows:
        cells = []
        for group in GROUPS:
            subset = errors if group == "all" else errors[errors.season == group]
            cells.append(_cell(np.sqrt(subset[column])) if len(subset) else "-")
        print(f"{label:<{name_width}}" + "".join(f"{c:>{width}}" for c in cells))
    if not has_climatology:
        print("(no climatology for this grid -- row left out)")
    print()


def plot_mean_residual_grid(results, out_dir):
    """Mean residual k+1 - k: rows cool and melt, one column per run and the truth (from the
    first run) last, on one symmetric colour scale. Also prints the mean and RMS of
    predicted - true mean residual per run and season."""
    truth = results[0][2]
    seasons = [s for s in GRID_SEASONS if s in truth]
    if not seasons:
        print("No cool- or melt-season samples -- skipping mean_residual_comparison.")
        return

    # Columns: (title, {season: field}); the truth is the true residual of the first run.
    columns = [(_tex_name(run), {s: residuals[s][0] for s in seasons if s in residuals})
               for run, _, residuals in results]
    columns.append(("Truth", {s: truth[s][1] for s in seasons}))

    # Fixed scale for all panels, so runs and seasons compare directly (and match mean_residual).
    vmax = RESIDUAL_VMAX

    n_cols = len(columns)
    fig, axes = plt.subplots(len(seasons), n_cols, squeeze=False)
    img = None
    for col, (title, by_season) in enumerate(columns):
        for row, season in enumerate(seasons):
            ax = axes[row, col]
            _clean_image_axis(ax)
            if season in by_season:
                img = ax.imshow(by_season[season], cmap=RESIDUAL_CMAP, vmin=-vmax, vmax=vmax,
                                origin="lower", interpolation="none")
            if col == 0:
                ax.set_ylabel(season.capitalize())
        axes[0, col].set_title(title)

    # One inset colorbar per row next to the right-hand map, so it matches the image height;
    # a colorbar spanning several axes breaks savefig's tight_layout.
    for row in range(len(seasons)):
        cbar = fig.colorbar(img, cax=axes[row, -1].inset_axes([1.04, 0, 0.05, 1]),
                            ticks=[-vmax, 0, vmax], extend="both")
        cbar.ax.set_yticklabels([f"${-vmax:.2f}$", "$0$", f"${vmax:.2f}$"])
        cbar.set_label("SIC change", labelpad=4)

    print("Mean residual, predicted - truth, over water:")
    for run, _, residuals in results:
        parts = []
        for season in seasons:
            if season in residuals:
                diff = residuals[season][0] - truth[season][1]
                parts.append(f"{season}: mean {np.nanmean(diff):+.4f}, "
                             f"RMS {np.sqrt(np.nanmean(diff ** 2)):.4f}")
        print(f"  {_name(run)}: " + " | ".join(parts))
    print()

    # Square panels: the width per column sets the height per row, plus room for titles.
    panel = TEXTWIDTH_IN / (n_cols + 0.3)
    savefig(fig, out_dir, "mean_residual_comparison", width=TEXTWIDTH_IN,
            height=panel * len(seasons) + 0.4)
    plt.close(fig)


def main():
    if not RUNS:
        raise ValueError("RUNS is empty -- list the folders under results/ to compare.")
    results = load_all(RUNS)
    print_rmse_table(results)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_mean_residual_grid(results, out_dir)
    print(f"Wrote {out_dir}/mean_residual_comparison.pdf")


if __name__ == "__main__":
    main()
