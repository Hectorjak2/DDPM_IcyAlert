"""Final-results comparison of any number of sampled runs.

Edit ``RUNS`` (folder names under ``results/``) and optionally ``LABELS`` below, then run
from the repo root:

    python -m plotting.final_results

1. Prints the per-sample RMSE over water, mean ± standard error over the samples, for all samples and
   per season (cool Oct--Mar, melt Apr--Sep). Climatology and persistence come first, then
   one row per run in ``RUNS`` order. Then a paired t-test on the per-date RMSE for each
   pair in ``PAIRED_TESTS``.
2. Writes ``<OUT_DIR>/mean_residual_comparison.pdf``: the mean residual k+1 - k per season
   (rows: cool, melt), one column per run and the true residual in the last column, all on
   one symmetric colour scale.
3. Writes ``<OUT_DIR>/sampling_loss_comparison.pdf`` (half page width): the RMSE over water
   per reverse step, averaged over samples, one curve per run. Runs can have different T, so
   the x-axis is the normalised timestep t / T.
4. Writes ``<OUT_DIR>/calibration_comparison.pdf`` (half page width, same height as 3):
   reliability, the mean target SIC per predicted-SIC bin, one curve per run, with its own
   legend.
5. For the ensemble ``SEASON_RUN`` only, cool vs melt season (by target month):
   ``<OUT_DIR>/calibration_seasons.pdf``, the reliability of 4 with one curve per season, and
   ``<OUT_DIR>/spread_skill_seasons.pdf``, the binned spread-skill diagram of
   ``plotting.ensemble_comparison`` with one line per season.
6. Writes ``<OUT_DIR>/lead_time_comparison.pdf`` (square, half width, log y-axis): the RMSE over water against
   lead time for the six-month rollouts in ``LEAD_RUNS``, next to persistence and
   climatology, and prints the numbers. ``python -m plotting.final_results --lead-only``
   writes only this figure.

Everything (runs, baselines, truth) is scored on the target dates that all runs share, so
every row covers the same samples. Runs sampled with the same ``TrainConfig.sample_seed``
share their first dates, e.g. a 13-date run is a subset of a 100-date one. A run with extra
dates is loaded a second time to drop them, and the script prints that it did.
"""
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MARKERS, MUTED, REF_LINESTYLE
from plotting.ensemble_evaluation import SPREAD_SKILL_BINS, season_of, spread_skill_curve
from models.climatology import ensemble_members
from plotting.evaluation import (ICE_EDGE, RESIDUAL_CMAP, RESIDUAL_VMAX, SEASON_TITLES,
                                 _clean_image_axis, _iiee, _load_climatology, _sps,
                                 calibration_stats, load_leads, mean_residuals, sample_errors)

# Folder names under results/, in the order they appear in the table and the grid.
RUNS = [
    #"ddpm-corrected-land-32",
    #"ddpm-corrected-land-64",
    #"ddpm-corrected-land-128",
    #"ddpm-corrected-land-256",
    #"ddpm-corrected-land-linear",
    #"ddpm-corrected-land-cosine",
    "ddpm-LC-256-ensemble-v3",
    #"ensembles/member_5",
    "deep-ensemble2",

]
# Optional display names; a run not listed here is shown by its folder name.
LABELS = {
    "ddpm-corrected-land-64": "Shift (64)",
    "ddpm-corrected-land-128": "Shift (128)",
    "ddpm-corrected-land-256": "DDPM, member",
    "ddpm-corrected-land-linear": "Linear",
    "ddpm-corrected-land-32": "Shift (32)",
    "ddpm-corrected-land-cosine": "Cosine",
    "ensembles/member_5": "U-Net, member",
    "deep-ensemble2": "U-Net Ensemble, mean",
    "ddpm-LC-256-ensemble": "DDPM Ensemble, mean"

}
# Legend labels for calibration_comparison only, where a long label would run into the
# curves in the upper-left legend; runs not listed use LABELS.
CALIBRATION_LABELS = {
    "deep-ensemble2": "U-Net Ensemble,\nmean",
}
# The ensemble whose cool and melt seasons are compared in calibration_seasons and
# spread_skill_seasons (must be in RUNS); None skips both figures.
SEASON_RUN = "ddpm-LC-256-ensemble-v3"
# Curve order (legend order) and colour per season in those two figures.
SEASON_ORDER = ("cool", "melt")
SEASON_COLORS = {"cool": COLORS[6], "melt": COLORS[2]}
LEAD = 1             # lead time to score; every run needs at least this many leads
# Climatological ensemble for the IIEE / SPS table: one member per year, the observed SIC on
# the target's calendar day in each of the CLIM_ENSEMBLE_YEARS years before the target year
# (models.climatology.ensemble_members). 10 matches the size of the model ensembles, which
# matters since a finite ensemble's SPS depends on M. None leaves it out.
CLIM_ENSEMBLE_YEARS = 10
# Paired t-tests on the per-date RMSE (A - B), printed after the RMSE table. Each entry is a
# run in RUNS or a baseline, "climatology" or "persistence".
PAIRED_TESTS = [
    ("ddpm-LC-256-ensemble-v3", "deep-ensemble2"),
    ("ddpm-LC-256-ensemble-v3", "climatology"),
    ("deep-ensemble2", "climatology"),
    ("ddpm-LC-256-ensemble-v3", "climatology_ensemble"),
    ("deep-ensemble2", "climatology_ensemble"),
    ("ensembles/member_5", "ddpm-corrected-land-256"),
    ("ddpm-corrected-land-256", "climatology"),
    ("ensembles/member_5", "climatology"),
]
OUT_DIR = "final_figures"  # where the figures are written
# Six-month rollouts compared in lead_time_comparison, {folder under results/: label}.
# Scored on the initial dates k they all share; the baselines come from the first run.
# Empty skips the figure.
LEAD_RUNS = {
    "ddpm-corrected-land-256-l6": "DDPM",
    "ensembles/member_0": "U-Net",
}

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
    """Load one run and keep only what the table and figures need, so the (large) pickle
    can be freed before the next one is loaded.

    With ``dates`` (a set of target dates), only the samples on those dates are scored.
    Returns a dict with ``run``; ``errors``, the ``sample_errors`` DataFrame (with the
    baselines); ``residuals``, the ``mean_residuals`` dict; ``losses``, the RMSE over water
    per reverse step averaged over the samples ([steps], None if the pickle has none);
    ``calibration``, the ``calibration_stats`` tuple; ``dates``, the target dates of the
    scored samples; and, for ``SEASON_RUN`` only (else None), ``seasons``: per season
    {"calibration": calibration_stats, "curve": spread_skill_curve (None for a single
    forecast)} over that season's samples.
    """
    _, leads = load_leads(run)
    if len(leads) < LEAD:
        raise ValueError(f"{run} has {len(leads)} lead time(s), LEAD = {LEAD}.")
    data = leads[LEAD - 1]
    del leads
    if dates is not None:
        data = _subset(data, [i for i, d in enumerate(data["times"])
                              if pd.Timestamp(d[0]) in dates])
    losses = data["sampling_losses"]
    errors = sample_errors(data, _climatology(data["targets"][0].shape))
    if CLIM_ENSEMBLE_YEARS:
        errors = errors.join(climatology_ensemble_errors(data))
    return {
        "run": run,
        "errors": errors,
        "residuals": mean_residuals(data),
        # Per-sample RMSE, then the mean over samples, as in print_rmse_table.
        "losses": None if losses is None else np.sqrt(losses).mean(axis=0),
        "calibration": calibration_stats(data),
        "dates": [pd.Timestamp(d[0]) for d in data["times"]],
        "n_members": data["members"][0].shape[0],
        "seasons": season_stats(data) if run == SEASON_RUN else None,
    }


_clim_ensemble_scores = {}  # target date -> scores; they depend on the date only


def climatology_ensemble_errors(data) -> pd.DataFrame:
    """One row per sample of ``data``, in its order, with the ice-edge scores of the
    climatological ensemble (see CLIM_ENSEMBLE_YEARS), in km^2:
    ``climatology_ensemble_iiee`` of its mean, ``climatology_ensemble_member_iiee`` of a single
    member averaged over the members, and ``climatology_ensemble_sps``. Cached per target
    date, so every run reuses the first run's fields."""
    rows = []
    for target, dates in zip(data["targets"], data["times"]):
        date = pd.Timestamp(dates[0])
        if date not in _clim_ensemble_scores:
            members = ensemble_members(date, target.shape, CLIM_ENSEMBLE_YEARS)
            _clim_ensemble_scores[date] = {
                # Land is NaN in every member, so the plain mean keeps it NaN.
                "climatology_ensemble_iiee": _iiee(members.mean(axis=0), target),
                "climatology_ensemble_member_iiee":
                    float(np.mean([_iiee(m, target) for m in members])),
                "climatology_ensemble_sps": _sps(members, target),
            }
        rows.append(_clim_ensemble_scores[date])
    return pd.DataFrame(rows)


def season_stats(data):
    """Per season with samples, in SEASON_ORDER: the ``calibration_stats`` and the
    ``spread_skill_curve`` (None for a single forecast) over that season's samples only."""
    n_members = data["members"][0].shape[0]
    out = {}
    for season in SEASON_ORDER:
        keep = [i for i, d in enumerate(data["times"]) if season_of(d[0]) == season]
        if not keep:
            continue
        sub = _subset(data, keep)
        out[season] = {
            "n": len(keep),
            "calibration": calibration_stats(sub),
            "curve": spread_skill_curve(sub, n_members) if n_members > 1 else None,
        }
    return out


def load_all(runs):
    """Load every run, one at a time, and score all of them on the dates they share.

    Returns a list of ``load_run`` dicts. The baselines depend only on the dates, so they
    are the same in every ``errors``; the table and the truth column take them from the
    first run.
    """
    loaded = []
    for run in runs:
        print(f"Loading {run} ...")
        loaded.append(load_run(run))

    common = set.intersection(*(set(r["dates"]) for r in loaded))
    if not common:
        raise ValueError("The runs share no target dates -- nothing to compare.")

    results = []
    for r in loaded:
        if set(r["dates"]) != common:
            print(f"{r['run']} has {len(set(r['dates']))} dates, {len(common)} shared with "
                  f"all runs -- reloading it on the shared dates only.")
            r = load_run(r["run"], common)
        results.append(r)
    return results


def _cell(rmse: pd.Series) -> str:
    """Mean ± standard error of the mean, std / sqrt(n) with the sample std (ddof=1).
    The SE treats the dates as independent; nearby daily dates are correlated, so it is
    somewhat too small. Overlapping SEs do not decide whether two models differ: they are
    scored on the same dates, so compare them with a paired test on the per-date differences."""
    n = len(rmse)
    se = rmse.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return f"{rmse.mean():.4f} ± {se:.4f}"


def print_rmse_table(results):
    """Per-sample RMSE over water, mean ± standard error of the mean, for all samples and
    per season."""
    reference = results[0]["errors"]
    # sample_errors leaves the column NaN when no climatology is fitted for the grid.
    has_climatology = reference.climatology.notna().any()
    rows = []  # (label, per-sample errors DataFrame, column holding the MSE)
    if has_climatology:
        rows.append(("Climatology", reference, "climatology"))
    rows.append(("Persistence", reference, "persistence"))
    rows += [(_name(r["run"]), r["errors"], "mse") for r in results]

    width = 19
    name_width = max(len(label) for label, _, _ in rows) + 2
    headers = []
    for group in GROUPS:
        n = len(reference) if group == "all" else int((reference.season == group).sum())
        headers.append(f"{group} (n={n})")
    header = f"{'':<{name_width}}" + "".join(f"{h:>{width}}" for h in headers)

    print(f"\nRMSE over water (lead {LEAD}), per sample: mean ± standard error (std / sqrt(n))")
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


IIEE_UNIT = 1e3  # the IIEE / SPS tables print in 10^3 km^2


def _iiee_cell(values: pd.Series) -> str:
    """Mean ± standard error, then the sample std (ddof=1) over the dates, in 10^3 km^2.
    The std is the date-to-date spread; the SE (std / sqrt(n)) is the uncertainty of the
    mean -- the one to compare between rows, with the same caveats as ``_cell``."""
    values = values / IIEE_UNIT
    n = len(values)
    std = values.std(ddof=1) if n > 1 else np.nan
    return f"{values.mean():.1f} ± {std / np.sqrt(n):.1f} (sd {std:.1f})"


def print_iiee_table(results):
    """Per-sample Integrated Ice Edge Error (SIC > ICE_EDGE), mean ± standard error (sd), in
    10^3 km^2, for all samples and per season. Baselines first (with the climatological
    ensemble, scored like the model ensembles), then the IIEE of each run's
    forecast (the ensemble mean); for an ensemble also the mean IIEE of a single member and
    the SPS, its probabilistic version (= IIEE for a single forecast and the baselines)."""
    reference = results[0]["errors"]
    has_climatology = reference.climatology_iiee.notna().any()
    rows = []  # (label, per-sample errors DataFrame, column)
    if has_climatology:
        rows.append(("Climatology", reference, "climatology_iiee"))
    rows.append(("Persistence", reference, "persistence_iiee"))
    if "climatology_ensemble_sps" in reference:
        label = f"Clim. ensemble ({CLIM_ENSEMBLE_YEARS} y)"
        rows += [(label, reference, "climatology_ensemble_iiee"),
                 (f"  {label}, member", reference, "climatology_ensemble_member_iiee"),
                 (f"  {label}, SPS", reference, "climatology_ensemble_sps")]
    for r in results:
        rows.append((_name(r["run"]), r["errors"], "iiee"))
        if r["n_members"] > 1:
            rows.append((f"  {_name(r['run'])}, member", r["errors"], "member_iiee"))
            rows.append((f"  {_name(r['run'])}, SPS", r["errors"], "sps"))

    width = 29
    name_width = max(len(label) for label, _, _ in rows) + 2
    headers = []
    for group in GROUPS:
        n = len(reference) if group == "all" else int((reference.season == group).sum())
        headers.append(f"{group} (n={n})")
    header = f"{'':<{name_width}}" + "".join(f"{h:>{width}}" for h in headers)

    print(f"IIEE (SIC > {ICE_EDGE}, lead {LEAD}) in 10^3 km^2, per sample: "
          f"mean ± standard error (sd)")
    print(header)
    print("-" * len(header))
    for label, errors, column in rows:
        cells = []
        for group in GROUPS:
            subset = errors if group == "all" else errors[errors.season == group]
            cells.append(_iiee_cell(subset[column]) if len(subset) else "-")
        print(f"{label:<{name_width}}" + "".join(f"{c:>{width}}" for c in cells))
    if not has_climatology:
        print("(no climatology for this grid -- row left out)")
    print()


BASELINES = ("climatology", "persistence", "climatology_ensemble")
BASELINE_LABELS = {"climatology_ensemble": "Clim. ensemble"}  # else the capitalised name
# Per paired-test metric: (title, function of (errors DataFrame, run or baseline name) -> the
# per-sample score, format of the mean difference, format of its SE).
METRICS = {
    "rmse": ("RMSE over water",
             lambda e, n: np.sqrt(e[n if n in BASELINES else "mse"]), "{:+.4f}", "{:.4f}"),
    "iiee": ("IIEE (10^3 km^2)",
             lambda e, n: e[f"{n}_iiee" if n in BASELINES else "iiee"] / IIEE_UNIT,
             "{:+.1f}", "{:.1f}"),
    # A deterministic forecast's SPS is its IIEE, so the deterministic baselines use their
    # IIEE; the climatological ensemble has its own SPS.
    "sps": ("SPS (10^3 km^2)",
            lambda e, n: e["sps" if n not in BASELINES else
                           f"{n}_sps" if f"{n}_sps" in e else f"{n}_iiee"] / IIEE_UNIT,
            "{:+.1f}", "{:.1f}"),
}


def _per_date(results, name: str, metric: str) -> pd.Series:
    """Per-sample ``metric`` of a run in ``results`` or a baseline (taken from the first run),
    indexed by target date. A date drawn twice is averaged, so every date counts once and two
    series pair up by date."""
    run = results[0] if name in BASELINES else next(r for r in results if r["run"] == name)
    score = METRICS[metric][1](run["errors"], name)
    values = pd.Series(score.to_numpy(), index=run["dates"])
    season = pd.Series(run["errors"]["season"].to_numpy(), index=run["dates"])
    return values.groupby(level=0).mean(), season.groupby(level=0).first()


def print_paired_tests(results, metric: str = "rmse"):
    """Paired t-test per pair in PAIRED_TESTS, for all dates and per season: the per-date
    difference d = A - B of ``metric`` (a key of METRICS), its mean ± standard error
    s_d / sqrt(n), t = mean / SE (df = n - 1) and the two-sided p-value. d < 0 means A has
    the lower error. Pairs with a run that is not in RUNS are skipped.

    The dates are treated as independent; nearby daily dates are correlated, so the real n is
    smaller and the p-values are somewhat optimistic."""
    from scipy import stats

    title, score, diff_fmt, se_fmt = METRICS[metric]
    loaded = {r["run"] for r in results}

    def available(x):
        """x is a run in RUNS, or a baseline this metric has a score for."""
        if x not in BASELINES:
            return x in loaded
        try:
            score(results[0]["errors"], x)
            return True
        except KeyError:
            return False

    pairs = [(a, b) for a, b in PAIRED_TESTS if available(a) and available(b)]
    skipped = [(a, b) for a, b in PAIRED_TESTS if (a, b) not in pairs]
    if not pairs:
        print(f"No pair in PAIRED_TESTS is available -- skipping the {metric} test.\n")
        return

    def name(x):
        return BASELINE_LABELS.get(x, x.capitalize()) if x in BASELINES else _name(x)

    print(f"Paired t-test on per-date {title} (lead {LEAD}), d = A - B (d < 0: A better)")
    width = 44
    pad = max(len(name(a) + " vs " + name(b)) for a, b in pairs) + 2
    print(f"{'A vs B':<{pad}}" + "".join(f"{g:>{width}}" for g in GROUPS))
    print("-" * (pad + width * len(GROUPS)))
    for a, b in pairs:
        (score_a, season), (score_b, _) = (_per_date(results, a, metric),
                                           _per_date(results, b, metric))
        score_a, score_b = score_a.align(score_b, join="inner")
        season = season.reindex(score_a.index)
        cells = []
        for group in GROUPS:
            keep = season.index if group == "all" else season.index[season == group]
            d = (score_a - score_b)[keep]
            n = len(d)
            if n < 2:
                cells.append("-")
                continue
            se = d.std(ddof=1) / np.sqrt(n)
            t, p = stats.ttest_rel(score_a[keep], score_b[keep])
            cells.append(f"{diff_fmt.format(d.mean())} ± {se_fmt.format(se)}  "
                         f"t={t:+5.1f}  p={p:.0e}  n={n}")
        print(f"{name(a) + ' vs ' + name(b):<{pad}}" + "".join(f"{c:>{width}}" for c in cells))
    for a, b in skipped:
        print(f"(skipped {a} vs {b}: not in RUNS, or no {metric} for it)")
    print("n = distinct target dates (a date drawn twice counts once)\n")


def plot_mean_residual_grid(results, out_dir):
    """Mean residual k+1 - k: rows cool and melt, one column per run and the truth (from the
    first run) last, on one symmetric colour scale. Also prints the mean and RMS of
    predicted - true mean residual per run and season."""
    truth = results[0]["residuals"]
    seasons = [s for s in GRID_SEASONS if s in truth]
    if not seasons:
        print("No cool- or melt-season samples -- skipping mean_residual_comparison.")
        return

    # Columns: (title, {season: field}); the truth is the true residual of the first run.
    columns = [(_tex_name(r["run"]), {s: r["residuals"][s][0] for s in seasons
                                      if s in r["residuals"]})
               for r in results]
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
    for r in results:
        residuals = r["residuals"]
        parts = []
        for season in seasons:
            if season in residuals:
                diff = residuals[season][0] - truth[season][1]
                parts.append(f"{season}: mean {np.nanmean(diff):+.4f}, "
                             f"RMS {np.sqrt(np.nanmean(diff ** 2)):.4f}")
        print(f"  {_name(r['run'])}: " + " | ".join(parts))
    print()

    # Square panels: the width per column sets the height per row, plus room for titles.
    panel = TEXTWIDTH_IN / (n_cols + 0.3)
    savefig(fig, out_dir, "mean_residual_comparison", width=TEXTWIDTH_IN,
            height=panel * len(seasons) + 0.4)
    plt.close(fig)


def plot_sampling_loss_comparison(results, out_dir):
    """RMSE over water per reverse step, averaged over samples, one curve per run.

    Runs can use different numbers of diffusion steps T, so each curve is drawn against
    t / T; sampling runs left to right, from t / T = 1 (pure noise) to 0."""
    curves = [(r["run"], r["losses"]) for r in results if r["losses"] is not None]
    for r in results:
        if r["losses"] is None:
            print(f"{r['run']} has no sampling losses -- left out of sampling_loss_comparison.")
    if not curves:
        print("No run has sampling losses -- skipping sampling_loss_comparison.")
        return False

    fig, ax = plt.subplots()
    for run, losses in curves:
        n_steps = len(losses)
        # Entry 0 belongs to the noisiest step t = T-1 (see evaluation.plot_sampling_loss).
        t = np.arange(n_steps - 1, -1, -1) / n_steps
        ax.plot(t, losses, label=_tex_name(run), linewidth=1.25)
    ax.set_xlim(1, 0)
    # Fit the y-axis to the curves (they span ~0.20-0.32, so starting at 0 wasted most of
    # the panel). The bottom is left ~40% of the data range below the lowest curve, rounded
    # down to the 0.05 tick grid, so the legend fits under every curve.
    lo = min(losses.min() for _, losses in curves)
    hi = max(losses.max() for _, losses in curves)
    tick = 0.05
    ax.set_ylim(np.floor((lo - 0.4 * (hi - lo)) / tick) * tick, hi + 0.05 * (hi - lo))
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick))
    ax.set_xlabel("normalised timestep $t/T$")
    ax.set_ylabel("RMSE over water")
    ax.legend(loc="lower left", ncol=2, handlelength=1.2, handletextpad=0.5,
              columnspacing=0.8)

    savefig(fig, out_dir, "sampling_loss_comparison", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)
    return True


def _family_color(name: str):
    """Colour for a run's model family, from its lower-cased display label: DDPM runs share
    COLORS[0], UNet runs COLORS[1]; anything else falls back to the default cycle (None)."""
    if "ddpm" in name:
        return COLORS[0]
    if "unet" in name or "u-net" in name:
        return COLORS[1]
    return None


def plot_calibration_comparison(results, out_dir):
    """Reliability: mean target SIC per predicted-SIC bin (pooled over all water pixels of
    the shared samples), one curve per run, against the diagonal. Only the means: the interquartile bands
    of ``evaluation.plot_calibration`` would overlap into a blur with several runs.

    One colour per model family (from the display label: DDPM or UNet); single members
    (label contains "member") are dashed and slightly transparent, ensemble means solid. The
    curves hug the diagonal, so the legend goes in the empty upper-left corner. The black
    dashed diagonal (perfect calibration) is left to the caption."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    for r in results:
        centers, filled, (mean, _, _) = r["calibration"]
        name = _name(r["run"]).lower()
        member = "member" in name
        label = CALIBRATION_LABELS.get(r["run"], _tex_name(r["run"]))
        ax.plot(centers[filled], mean, label=label, linewidth=1.25,
                color=_family_color(name), linestyle="--" if member else "-",
                alpha=0.6 if member else 1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted SIC")
    ax.set_ylabel("mean target SIC")
    ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.3, labelspacing=0.3,
              borderaxespad=0.1)

    savefig(fig, out_dir, "calibration_comparison", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def plot_calibration_seasons(seasons, out_dir):
    """Reliability of SEASON_RUN, one curve per season, styled as calibration_comparison:
    the black dashed diagonal (perfect calibration) is left to the caption."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    for season, s in seasons.items():
        centers, filled, (mean, _, _) = s["calibration"]
        ax.plot(centers[filled], mean, label=SEASON_TITLES[season], linewidth=1.25,
                color=SEASON_COLORS[season])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted SIC")
    ax.set_ylabel("mean target SIC")
    ax.legend(loc="upper left", handlelength=1.4, handletextpad=0.3, labelspacing=0.3,
              borderaxespad=0.1)

    savefig(fig, out_dir, "calibration_seasons", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def plot_spread_skill_seasons(seasons, out_dir):
    """Binned spread-skill diagram of SEASON_RUN, one line per season, styled as
    ensemble_comparison.plot_spread_skill (dashed 1:1 line = perfect calibration)."""
    fig, ax = plt.subplots()
    top = 0.0
    print(f"Spread-skill bins of {_name(SEASON_RUN)} ({SPREAD_SKILL_BINS}, marginal ice zone): "
          f"RMSE / spread per bin")
    for (season, s), marker in zip(seasons.items(), MARKERS):
        rmse, spread = s["curve"]
        ax.plot(rmse, spread, color=SEASON_COLORS[season], marker=marker, markersize=4,
                label=SEASON_TITLES[season])
        top = max(top, rmse.max(), spread.max())
        print(f"  {season:<5} (n={s['n']:>3}) " + " ".join(f"{r:.2f}" for r in rmse / spread))
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
    savefig(fig, out_dir, "spread_skill_seasons", width=TEXTWIDTH_IN / 2, height=3.0)
    plt.close(fig)


def plot_season_figures(results, out_dir):
    """calibration_seasons and spread_skill_seasons for SEASON_RUN, if it is in RUNS."""
    if SEASON_RUN is None:
        return
    seasons = next((r["seasons"] for r in results if r["run"] == SEASON_RUN), None)
    if not seasons:
        print(f"SEASON_RUN {SEASON_RUN} is not in RUNS (or has no cool/melt samples) -- "
              f"skipping calibration_seasons and spread_skill_seasons.")
        return
    print(f"{_name(SEASON_RUN)} per season: "
          + ", ".join(f"{season} {s['n']} dates" for season, s in seasons.items()))
    plot_calibration_seasons(seasons, out_dir)
    print(f"Wrote {out_dir}/calibration_seasons.pdf")
    if any(s["curve"] is None for s in seasons.values()):
        print(f"{SEASON_RUN} is a single forecast -- skipping spread_skill_seasons.")
        return
    plot_spread_skill_seasons(seasons, out_dir)
    print(f"Wrote {out_dir}/spread_skill_seasons.pdf")


def lead_errors(run: str) -> pd.DataFrame:
    """Per-sample errors of a rollout at every lead: the ``sample_errors`` rows of each lead,
    with ``lead`` (1 first) and ``init``, the initial date k, so runs can be matched on k.
    Only the errors are kept, so the (large) pickle can be freed before the next run."""
    _, leads = load_leads(run)
    climatology = _climatology(leads[0]["targets"][0].shape)
    frames = []
    for lead, data in enumerate(leads, start=1):
        df = sample_errors(data, climatology)
        df["lead"] = lead
        df["init"] = [pd.Timestamp(d[1]) for d in data["times"]]
        frames.append(df)
    del leads
    return pd.concat(frames, ignore_index=True)


def plot_lead_time_comparison(out_dir):
    """RMSE over water against lead time (months) for every run in LEAD_RUNS, next to
    persistence (forecast k+L = k) and climatology (mean of the k+L calendar month). Each
    point is the mean of the per-sample RMSE over the initial dates all runs share, as in
    print_rmse_table; the baselines are taken from the first run. Also prints the table."""
    errors = {}
    for run in LEAD_RUNS:
        print(f"Loading {run} (all leads) ...")
        errors[run] = lead_errors(run)
    n_leads = min(df.lead.max() for df in errors.values())
    common = set.intersection(*(set(df.init) for df in errors.values()))
    if not common:
        raise ValueError("LEAD_RUNS share no initial dates -- nothing to compare.")
    for run, df in errors.items():
        if set(df.init) != common:
            print(f"{run} has {df.init.nunique()} initial dates, {len(common)} shared -- "
                  f"scoring the shared ones only.")
        errors[run] = df[df.init.isin(common) & (df.lead <= n_leads)]

    def rmse_by_lead(df, column):
        return np.sqrt(df[column]).groupby(df.lead).mean().to_numpy()

    reference = next(iter(errors.values()))
    # (label, RMSE per lead, colour, linestyle): models solid, baselines dashed grey.
    series = [(label, rmse_by_lead(errors[run], "mse"), _family_color(label.lower()), "-")
              for run, label in LEAD_RUNS.items()]
    series.append(("Persistence", rmse_by_lead(reference, "persistence"), COLORS[5], "--"))
    if reference.climatology.notna().any():
        series.append(("Climatology", rmse_by_lead(reference, "climatology"), COLORS[3], "--"))

    lead_times = np.arange(1, n_leads + 1)
    print(f"\nRMSE over water by lead time ({len(common)} initial dates), mean over samples")
    print(f"{'':<13}" + "".join(f"{f'lead {L}':>9}" for L in lead_times))
    for label, rmse, _, _ in series:
        print(f"{label:<13}" + "".join(f"{v:>9.4f}" for v in rmse))
    print()

    fig, ax = plt.subplots()
    for (label, rmse, color, linestyle), marker in zip(series, MARKERS):
        ax.plot(lead_times, rmse, label=label, color=color, linestyle=linestyle,
                marker=marker, markersize=3.5, linewidth=1.25)
    ax.set_xticks(lead_times)
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("RMSE over water")
    # Log scale: the models and climatology sit close together (~0.15-0.3) while persistence
    # climbs to ~0.6; a log axis keeps both readable. Plain decimal labels on a fixed grid.
    ax.set_yscale("log")
    lo = min(rmse.min() for _, rmse, _, _ in series)
    hi = max(rmse.max() for _, rmse, _, _ in series)
    ax.set_ylim(lo / 1.08, hi * 1.08)
    ticks = [t for t in (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
             if lo / 1.08 <= t <= hi * 1.08]
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"${t:g}$" for t in ticks]))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    # Above the axes, so it never covers a point; two columns fit the half width.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.6, handletextpad=0.4, columnspacing=0.9)

    # Square, half the text width.
    savefig(fig, out_dir, "lead_time_comparison", width=TEXTWIDTH_IN / 2,
            height=TEXTWIDTH_IN / 2)
    plt.close(fig)


def main(lead_only: bool = False):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if LEAD_RUNS:
        plot_lead_time_comparison(out_dir)
        print(f"Wrote {out_dir}/lead_time_comparison.pdf")
    if lead_only:
        return

    if not RUNS:
        raise ValueError("RUNS is empty -- list the folders under results/ to compare.")
    results = load_all(RUNS)
    print_rmse_table(results)
    #print_paired_tests(results)
    print_iiee_table(results)
    print_paired_tests(results, "iiee")
    print_paired_tests(results, "sps")

    plot_mean_residual_grid(results, out_dir)
    print(f"Wrote {out_dir}/mean_residual_comparison.pdf")
    if plot_sampling_loss_comparison(results, out_dir):
        print(f"Wrote {out_dir}/sampling_loss_comparison.pdf")
    plot_calibration_comparison(results, out_dir)
    print(f"Wrote {out_dir}/calibration_comparison.pdf")
    plot_season_figures(results, out_dir)


if __name__ == "__main__":
    import sys
    main(lead_only="--lead-only" in sys.argv)
