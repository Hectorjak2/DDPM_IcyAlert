"""Thesis figures for conditional DDPM samples.

Reads ``results/<run>/samples.pkl`` written by
``utils.sampling.forecast`` (``--lead L`` for a rollout) and writes four PDFs next to it:

1. ``sampling_loss``  -- MSE against the target per reverse step, averaged over samples.
2. ``calibration``    -- reliability: mean target SIC per 0.01 predicted-SIC bin,
   one panel per lead time (2x3 for a six-month rollout), always next to the pickle.
3. ``forecast_row``   -- k-1, k, k+1 (truth) and prediction for one sample.
4. ``forecast_grid``  -- truth (top) vs prediction (bottom) for four samples.
5. ``forecast_rows_melt`` / ``forecast_rows_cool`` -- four forecast rows each, one per
   target month, spread over the melt (Apr-Sep) and cool (Oct-Mar) season.
6. ``ice_edge_error`` -- prediction for one cool- and one melt-season sample, with the
   true (black) and predicted (magenta) 0.15 ice edge and large-error pixels in red.
7. ``mean_residual``  -- mean residual k+1 - k per season, predicted (top) vs truth (bottom).
   Random-sign sample noise averages out, so this shows the model's systematic signal
   (has it learned the seasonal melt/growth?) and any one-sided bias.
8. ``loss_history``   -- training vs validation loss per epoch (mean over all timesteps), from
   the ``loss_history.pkl`` that ``DDPM.train`` writes next to the weights.
   Skipped if the run has none. ``--loss-only`` draws just this one (no samples.pkl needed).

Also prints a table of the per-sample MSE over water, overall and per season, next to
the persistence and climatology baselines (see ``models.climatology``).

Ensembles: an ``ensemble_samples.pkl`` (``ensemble.py``, same layout) is read too, and
used when the run has no ``samples.pkl``. Every figure and the MSE then show the ensemble
mean. ``mse_by_lead`` adds the single-member MSE, and a second table lists the single-member
MSE and the ensemble CRPS next to the baselines' MAE (their CRPS). ``sampling_loss`` is
the first member's. ``--member M`` scores one member as a single forecast instead
(figures go to ``member<M>/``). Rank histogram and spread-skill: ``plotting.ensemble_evaluation``.

Works for both CARRA2-TEST and CARRA2-WEST without a mask file: land is already NaN in
the pickled targets and samples, so water = ``isfinite(target)`` at whatever resolution
the run used.

Run from the repo root:  python -m plotting.evaluation <run_name | path/to/samples.pkl>
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from models.climatology import Climatology
from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MARKERS, MUTED, REF_LINESTYLE
from utils.dataloader import to_data_range

# Sea-ice colormap: open water dark blue, full ice white, land grey.
SIC_CMAP = plt.cm.Blues_r.copy()
SIC_CMAP.set_bad("lightgray")
# Residual (k+1 - k) colormap: red = ice loss, blue = ice gain, land grey as above.
RESIDUAL_CMAP = plt.cm.RdBu.copy()
RESIDUAL_CMAP.set_bad("lightgray")
# Fixed symmetric colour limit for the mean-residual maps, so the figures compare across runs.
# A data-driven limit let a drifting model's predictions stretch the scale and wash out the
# (identical) truth panels. The true mean residual has |.| <= 0.25 at the 99th percentile on
# the 2024 forecast dates; anything beyond the limit is clipped (arrow ends on the colorbar).
RESIDUAL_VMAX = 0.3

N_BINS = 100  # 0.01-wide predicted-SIC bins for the reliability (calibration) plot
MIN_BIN_PIXELS = 100  # bins with fewer pixels are left out of the calibration curves
MAX_CALIBRATION_COLS = 3  # lead-time panels per row in the calibration grid

# Target (k+1) months per season, in seasonal order. Same split as MELT_SEASON /
# COOL_SEASON in config.py.
SEASONS = {"melt": (4, 5, 6, 7, 8, 9), "cool": (10, 11, 12, 1, 2, 3)}

ICE_EDGE = 0.15  # SIC contour that defines the ice edge
# CARRA2 grid spacing. The Lambert conformal grid is close to equal-area over the domain, so
# the IIEE / SPS treat every cell as GRID_KM^2 (the data carries no cell areas).
GRID_KM = 2.5
CELL_AREA_KM2 = GRID_KM ** 2
# Squared-error threshold for the red overlay (|error| > ~0.32). Absolute rather than a
# per-sample percentile, so the amount of red is comparable between the two seasons.
ERROR_THRESHOLD = 0.1
# Preferred target month per season in ice_edge_error: maximum (Mar) and minimum (Sep)
# ice extent.
EDGE_MONTHS = {"cool": 3, "melt": 9}
SEASON_TITLES = {"cool": "Cool (Oct--Mar)", "melt": "Melt (Apr--Sep)"}


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


def load_leads(path: str | Path, member: int | None = None):
    """Load a samples.pkl or ensemble_samples.pkl, given its path or the run's results
    directory/name (samples.pkl if the run has both).

    Returns ``(run_dir, leads)`` with one ``data`` dict per lead time, lead 1 first (a
    single entry for a one-month forecast). Every field in ``data`` is a numpy [H, W]
    array in data range [0, 1] with land as NaN, and ``sampling_losses`` is an [N, steps]
    array (or None for pickles written before the running loss existed, or by
    utils.sampling.forecast). In the dict for lead L, ``predictions`` and ``targets`` are
    k+L and ``times`` holds (k+L, k, k-1), so everything downstream reads as for lead 1.

    ``members`` holds the [M, H, W] ensemble per sample (M = 1 for a single forecast) and
    ``predictions`` its ensemble mean, so every figure and the MSE score the ensemble mean.
    ``member`` keeps only that one member, which then is scored as a single forecast.
    """
    path = Path(path)
    if path.suffix != ".pkl":
        run_dir = path if path.exists() else Path("results") / path
        path = run_dir / "samples.pkl"
        if not path.exists() and (run_dir / "ensemble_samples.pkl").exists():
            path = run_dir / "ensemble_samples.pkl"

    with open(path, "rb") as f:
        payload = pickle.load(f)
    samples, contexts, targets, times = payload[:4]
    # forecast() writes (..., member_seeds, sampling_losses); older pickles had the sampling
    # losses 5th, and an in-between version member_seeds (ints) 5th with no losses at all.
    if len(payload) > 5:
        losses = payload[5]
    else:
        losses = payload[4] if len(payload) > 4 and payload[4] and not isinstance(payload[4][0], int) else None
    # An intermediate version stored only the last sample's curve (a flat list of floats).
    if losses:
        if not isinstance(losses[0], (list, tuple)):
            losses = [losses]
        losses = np.asarray(losses, dtype=np.float64)

    # samples are [n_members, lead_months, H, W]; pickles older than forecast() held a single
    # member, some without the member/lead axes. targets are [lead_months, H, W].
    samples = [s if s.ndim == 4 else s.reshape(1, -1, *s.shape[-2:]) for s in samples]
    if member is not None:
        n_members = samples[0].shape[0]
        if not 0 <= member < n_members:
            raise ValueError(f"member {member} out of range: {path} has {n_members} member(s)")
        samples = [s[member:member + 1] for s in samples]
    targets = [_as_data_range(t).reshape(-1, *t.shape[-2:]) for t in targets]
    # contexts are [2, H, W] = (k, k-1)
    contexts = [_as_data_range(c) for c in contexts]

    leads = [{
        "members": [s[:, lead].numpy() for s in samples],
        # Land is NaN in every member, so the plain mean keeps it NaN. One member: no copy.
        "predictions": [(s[0, lead] if s.shape[0] == 1 else s[:, lead].mean(dim=0)).numpy()
                        for s in samples],
        "targets": [t[lead].numpy() for t in targets],
        "k": [c[0].numpy() for c in contexts],
        "km1": [c[1].numpy() for c in contexts],
        "times": [(d[lead], d[-2], d[-1]) for d in times],  # (k+lead, k, k-1) dates
        "sampling_losses": losses,
    } for lead in range(targets[0].shape[0])]

    return path.parent, leads


def load_samples(path: str | Path, lead: int = 1, member: int | None = None):
    """``(run_dir, data)`` for one lead time; see load_leads."""
    run_dir, leads = load_leads(path, member)
    return run_dir, leads[lead - 1]


def n_members(data) -> int:
    """Ensemble size of ``data`` (1 for a single forecast)."""
    return data["members"][0].shape[0]


def prediction_label(data) -> str:
    """What ``predictions`` is, for figure labels."""
    return "ensemble mean" if n_members(data) > 1 else "prediction"


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


def calibration_stats(data):
    """Reliability: the target per predicted-SIC bin, pooled over all water pixels of all
    samples. With an ensemble the prediction is the ensemble mean.

    Returns (centers, filled, (mean, q25, q75)), the statistics only over the bins in
    ``filled``.
    """
    target = np.concatenate([t.ravel() for t in data["targets"]]).astype(np.float64)
    pred = np.concatenate([p.ravel() for p in data["predictions"]]).astype(np.float64)
    valid = np.isfinite(target) & np.isfinite(pred)
    target, pred = target[valid], pred[valid]

    # Bin on the prediction (reliability, E[y | ŷ]), not on the target: conditioning on the
    # target gives E[ŷ | y], which flattens towards the middle even for a perfectly reliable
    # forecast (regression to the mean, strongest for the smooth ensemble mean).
    # SIC exactly 1.0 (or a prediction just outside [0, 1]) goes in an end bin.
    bins = np.clip((pred * N_BINS).astype(np.int64), 0, N_BINS - 1)
    counts = np.bincount(bins, minlength=N_BINS)
    # Group targets by bin with one sort instead of a boolean mask per bin.
    per_bin = np.split(target[np.argsort(bins, kind="stable")], np.cumsum(counts)[:-1])
    stats = np.full((N_BINS, 3), np.nan)  # mean, q25, q75
    for k, y in enumerate(per_bin):
        if y.size:
            stats[k] = [y.mean(), *np.percentile(y, [25, 75])]
    centers = (np.arange(N_BINS) + 0.5) / N_BINS

    # Most water pixels are predicted ~0, and a forecast that mimics CARRA2 has (almost) no
    # SIC between 0 and ~0.05, which can leave the 0-0.01 bin isolated between empty bins;
    # a line does not draw through NaNs. Curves therefore run through the populated bins
    # only, bridging any gap. Bins with only a handful of pixels are bridged too: their
    # quantiles are noise and draw spurious kinks.
    filled = counts >= MIN_BIN_PIXELS
    return centers, filled, stats[filled].T


def plot_calibration(leads, out_dir):
    """Reliability per lead time: for each lead, the target per predicted-SIC bin. One
    panel per lead, at most MAX_CALIBRATION_COLS per row (1x1 for a one-month forecast, 2x3
    for a six-month rollout), titled with the lead.

    Shows mean and interquartile range rather than mean +- std: within a bin the targets
    are bimodal (an intermediate prediction near the ice edge mostly falls on open water
    or on pack ice), so the std sits near its 0/1-mixture maximum in every bin and says
    little, while the quartiles show the split.
    """
    n = len(leads)
    cols = min(n, MAX_CALIBRATION_COLS)
    rows = -(-n // cols)

    fig = plt.figure()
    # A legend row on top: inside the panels, the band and curves leave no free corner.
    gs = fig.add_gridspec(1 + rows, cols, height_ratios=[0.5 if cols == 1 else 0.3] + [3] * rows)
    first = None
    for i, data in enumerate(leads):
        r, c = divmod(i, cols)
        centers, filled, (mean, q25, q75) = calibration_stats(data)

        ax = fig.add_subplot(gs[1 + r, c], sharex=first, sharey=first)
        first = first or ax

        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE,
                label="perfect")
        ax.fill_between(centers[filled], q25, q75, color=COLORS[0], alpha=0.25, linewidth=0,
                        label="interquartile range")
        ax.plot(centers[filled], mean, color=COLORS[0], label="mean")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if n > 1:
            ax.set_title(f"Lead: {i + 1}")

        if c == 0:
            ax.set_ylabel("target SIC")
        else:
            ax.tick_params(labelleft=False)
        # x axis labelled only on the lowest panel of each column
        if i + cols >= n:
            ax.set_xlabel("predicted SIC")
        else:
            ax.tick_params(labelbottom=False)

    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis("off")
    ax_legend.legend(*first.get_legend_handles_labels(), loc="center", ncol=2 if cols == 1 else 3)

    width = TEXTWIDTH_IN / 2 if cols == 1 else TEXTWIDTH_IN
    height = 2.8 if cols == 1 else 0.4 + 2.1 * rows
    savefig(fig, out_dir, "calibration", width=width, height=height)
    plt.close(fig)


def plot_forecast_row(data, out_dir, index: int = 0):
    """k-1, k, k+1 (truth) and prediction for one sample, with a shared colorbar."""
    kp1_date, k_date, km1_date = data["times"][index]
    panels = (
        (data["km1"][index], rf"$k{{-}}1$: {_month(km1_date)}"),
        (data["k"][index], rf"$k$: {_month(k_date)}"),
        (data["targets"][index], rf"$k{{+}}1$ truth: {_month(kp1_date)}"),
        (data["predictions"][index], rf"$k{{+}}1$ {prediction_label(data)}"),
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
            ax_p.set_ylabel(prediction_label(data))
    fig.colorbar(img, cax=fig.add_subplot(gs[:, n])).set_label("SIC", labelpad=4)

    savefig(fig, out_dir, "forecast_grid", width=TEXTWIDTH_IN, height=3.3)
    plt.close(fig)


def season_indices(data, season: str, n: int = 4) -> list[int]:
    """Pick up to ``n`` samples whose target (k+1) month is in ``season``, one per month
    and spread evenly over the season, in seasonal order (e.g. Apr -> Sep for melt)."""
    months = SEASONS[season]
    by_month = {m: [] for m in months}
    for i, dates in enumerate(data["times"]):
        m = pd.Timestamp(dates[0]).month
        if m in by_month:
            by_month[m].append(i)
    available = [m for m in months if by_month[m]]
    if not available:
        return []
    if len(available) >= n:
        picks = np.unique(np.linspace(0, len(available) - 1, n).round().astype(int))
        return [by_month[available[p]][0] for p in picks]
    # Fewer months than rows: fill up with further samples, still in seasonal order.
    return [i for m in available for i in by_month[m]][:n]


def plot_forecast_rows(data, out_dir, indices, name: str):
    """One forecast_row per sample, stacked: columns k-1, k, k+1 truth, prediction."""
    n = len(indices)
    fig = plt.figure()
    gs = fig.add_gridspec(n, 5, width_ratios=[1, 1, 1, 1, 0.06])
    headers = (r"$k{-}1$", r"$k$", r"$k{+}1$ truth", rf"$k{{+}}1$ {prediction_label(data)}")
    for row, i in enumerate(indices):
        fields = (data["km1"][i], data["k"][i], data["targets"][i], data["predictions"][i])
        for col, field in enumerate(fields):
            ax = fig.add_subplot(gs[row, col])
            img = _show_field(ax, field)
            _clean_image_axis(ax)
            if row == 0:
                ax.set_title(headers[col])
            if col == 0:
                # The k+1 month; k and k-1 are the one and two months before it.
                ax.set_ylabel(_month(data["times"][i][0]))
    fig.colorbar(img, cax=fig.add_subplot(gs[:, 4])).set_label("SIC", labelpad=4)

    savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=1.45 * n + 0.4)
    plt.close(fig)


def edge_index(data, season: str) -> int | None:
    """First sample whose target month is EDGE_MONTHS[season], else the first sample
    season_indices picks for that season (None if the season has no samples)."""
    for i, dates in enumerate(data["times"]):
        if pd.Timestamp(dates[0]).month == EDGE_MONTHS[season]:
            return i
    picks = season_indices(data, season, n=1)
    return picks[0] if picks else None


def plot_ice_edge_error(data, out_dir, indices: dict):
    """Prediction per season with the true (black) and predicted (magenta) ice edge and
    the pixels whose squared error exceeds ERROR_THRESHOLD in red.

    ``indices`` maps season name -> sample index; seasons mapped to None are left out.
    """
    panels = [(season, i) for season, i in indices.items() if i is not None]
    if not panels:
        print("No cool- or melt-season samples -- skipping ice_edge_error.")
        return
    n = len(panels)

    fig = plt.figure()
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.06])
    for col, (season, i) in enumerate(panels):
        target, pred = data["targets"][i], data["predictions"][i]
        ax = fig.add_subplot(gs[0, col])
        img = _show_field(ax, pred)

        sq_err = (pred - target) ** 2
        overlay = np.zeros((*target.shape, 4))
        overlay[np.isfinite(sq_err) & (sq_err > ERROR_THRESHOLD)] = (1.0, 0.0, 0.0, 0.6)
        ax.imshow(overlay, origin="lower", interpolation="none")

        ax.contour(target, levels=[ICE_EDGE], colors="black", linewidths=0.8)
        ax.contour(pred, levels=[ICE_EDGE], colors="magenta", linewidths=0.8)

        ax.set_title(f"{season.capitalize()}: {_month(data['times'][i][0])}")
        _clean_image_axis(ax)
    fig.colorbar(img, cax=fig.add_subplot(gs[0, n])).set_label("SIC", labelpad=4)

    handles = [
        Line2D([], [], color="black", linewidth=0.8, label="true ice edge"),
        Line2D([], [], color="magenta", linewidth=0.8,
               label="ensemble-mean ice edge" if n_members(data) > 1 else "predicted ice edge"),
        Patch(facecolor="red", alpha=0.6, label=rf"squared error $> {ERROR_THRESHOLD}$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)

    savefig(fig, out_dir, "ice_edge_error", width=TEXTWIDTH_IN, height=3.2)
    plt.close(fig)


def _water_mse(pred, target) -> float:
    """MSE over water. ``pred`` may be [M, H, W]: then the mean over all members."""
    sq_err = (pred - target) ** 2
    return float(sq_err[np.isfinite(sq_err)].mean())


def _water_mae(pred, target) -> float:
    abs_err = np.abs(pred - target)
    return float(abs_err[np.isfinite(abs_err)].mean())


def _water_crps(members, target) -> float:
    """Fair ensemble CRPS (Ferro 2014), averaged over water. members [M, H, W].

    CRPS = mean_i |x_i - y| - sum_{i<j} |x_i - x_j| / (M (M - 1)). It rewards an ensemble
    that is both close to the truth and spread as wide as its error, and is the MAE for a
    single member, so it compares directly with the MAE of persistence and climatology.
    """
    water = np.isfinite(target) & np.isfinite(members).all(axis=0)
    x = np.sort(members[:, water].astype(np.float64), axis=0)  # [M, P]
    y = target[water].astype(np.float64)
    m = x.shape[0]
    skill = np.abs(x - y).mean(axis=0)
    if m == 1:
        return float(skill.mean())
    # With the members sorted, sum_{i<j} |x_i - x_j| = sum_j (2j - M - 1) x_(j), j = 1..M.
    spread = (2 * np.arange(1, m + 1) - m - 1) @ x / (m * (m - 1))
    return float((skill - spread).mean())


def _iiee(pred, target) -> float:
    """Integrated Ice Edge Error (Goessling et al. 2016) in km^2: the area of water where the
    forecast and the target disagree on ice (SIC > ICE_EDGE) = overestimation + underestimation."""
    water = np.isfinite(pred) & np.isfinite(target)
    wrong = (pred[water] > ICE_EDGE) != (target[water] > ICE_EDGE)
    return float(wrong.sum() * CELL_AREA_KM2)


def _sps(members, target) -> float:
    """Spatial Probability Score (Goessling & Jung 2018) in km^2: the area integral of
    (P_f - O)^2, with P_f the share of members with SIC > ICE_EDGE and O the observed ice
    (0/1). members [M, H, W]; for a single member it is the IIEE."""
    water = np.isfinite(target) & np.isfinite(members).all(axis=0)
    prob = (members[:, water] > ICE_EDGE).mean(axis=0)
    obs = target[water] > ICE_EDGE
    return float(((prob - obs) ** 2).sum() * CELL_AREA_KM2)


def sample_errors(data, climatology) -> pd.DataFrame:
    """One row per sample: its season (of the target month), the MSE over water of the
    forecast (the ensemble mean), of persistence (forecast = k) and of climatology (mean of
    the target's calendar month over the training years; NaN without a climatology), and the
    share of water pixels whose squared error exceeds ERROR_THRESHOLD.

    Per ensemble: ``member_mse`` is the MSE of a single member, averaged over the members
    (the same as ``mse`` for one member), and ``crps`` the ensemble CRPS, next to the MAE of
    the two baselines (``persistence_mae``, ``climatology_mae``), which is their CRPS.

    Ice edge, in km^2: ``iiee`` of the forecast (the ensemble mean), ``member_iiee`` of a
    single member averaged over the members, ``sps`` of the ensemble (= ``member_iiee`` for
    one member), and ``persistence_iiee`` / ``climatology_iiee``, which are also their SPS."""
    rows = []
    for pred, members, target, k, dates in zip(data["predictions"], data["members"],
                                               data["targets"], data["k"], data["times"]):
        sq_err = (pred - target) ** 2
        water = np.isfinite(sq_err)
        month = pd.Timestamp(dates[0]).month
        clim = climatology.predict(month) if climatology is not None else None
        rows.append({
            "season": next(s for s, months in SEASONS.items() if month in months),
            "mse": float(sq_err[water].mean()),
            "member_mse": _water_mse(members, target),
            "crps": _water_crps(members, target),
            "persistence": _water_mse(k, target),
            "persistence_mae": _water_mae(k, target),
            "climatology": _water_mse(clim, target) if clim is not None else np.nan,
            "climatology_mae": _water_mae(clim, target) if clim is not None else np.nan,
            "large": float((sq_err[water] > ERROR_THRESHOLD).mean()),
            "iiee": _iiee(pred, target),
            "member_iiee": float(np.mean([_iiee(m, target) for m in members])),
            "sps": _sps(members, target),
            "persistence_iiee": _iiee(k, target),
            "climatology_iiee": _iiee(clim, target) if clim is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _load_climatology(shape):
    climatology = Climatology.for_shape(shape)
    if climatology is None:
        print("No climatology fitted for this grid -- run `python -m models.climatology` "
              "and copy its .npy next to the data to add the climatology baseline.")
    return climatology


def print_stats(data):
    """Per-sample MSE over water, averaged over all samples and per season, next to the
    persistence (forecast k+1 = k) and climatology (mean of the k+1 calendar month over
    the training years) baselines, and the share of water pixels whose squared error
    exceeds ERROR_THRESHOLD (the red pixels in ice_edge_error)."""
    climatology = _load_climatology(data["targets"][0].shape)
    df = sample_errors(data, climatology)
    m = n_members(data)
    groups = [("all", df), *((s, df[df.season == s]) for s in ("cool", "melt"))]
    groups = [(name, group) for name, group in groups if not group.empty]

    header = (f"{'':<6} {'n':>4} {'MSE (mean ± std)':>20} {'persistence MSE':>16} "
              f"{'climatology MSE':>16} {f'SE > {ERROR_THRESHOLD}':>10}")
    print("\nMSE over water, per sample, averaged over samples"
          + (f" (forecast = mean of {m} members)" if m > 1 else ""))
    print(header)
    print("-" * len(header))
    for name, group in groups:
        mse = f"{group.mse.mean():.5f} ± {group.mse.std(ddof=0):.5f}"
        clim = f"{group.climatology.mean():.5f}" if climatology is not None else "-"
        print(f"{name:<6} {len(group):>4} {mse:>20} {group.persistence.mean():>16.5f} "
              f"{clim:>16} {group.large.mean():>10.2%}")
    print()

    if m == 1:
        return
    header = (f"{'':<6} {'n':>4} {'member MSE':>11} {'CRPS':>9} {'persistence MAE':>16} "
              f"{'climatology MAE':>16}")
    print(f"Ensemble of {m} members over water: MSE of a single member (mean over members), "
          f"and CRPS next to the baselines' MAE (their CRPS)")
    print(header)
    print("-" * len(header))
    for name, group in groups:
        clim = f"{group.climatology_mae.mean():.5f}" if climatology is not None else "-"
        print(f"{name:<6} {len(group):>4} {group.member_mse.mean():>11.5f} "
              f"{group.crps.mean():>9.5f} {group.persistence_mae.mean():>16.5f} {clim:>16}")
    print()


def plot_mse_by_lead(leads, out_dir):
    """MSE over water against lead time, averaged over all samples, with the climatology
    baseline. Each point is the mean of the per-sample MSE, as in print_stats's "all" row.
    For an ensemble, both the ensemble mean and a single member (mean over members)."""
    climatology = _load_climatology(leads[0]["targets"][0].shape)
    lead_times = np.arange(1, len(leads) + 1)
    errors = [sample_errors(data, climatology) for data in leads]
    if n_members(leads[0]) > 1:
        series = [("mse", "ensemble mean", COLORS[0], "o"),
                  ("member_mse", "single member", COLORS[1], "s")]
    else:
        series = [("mse", "model", COLORS[0], "o")]
    if climatology is not None:
        series.append(("climatology", "climatology", COLORS[3], "^"))

    fig, ax = plt.subplots()
    for column, label, color, marker in series:
        ax.plot(lead_times, [df[column].mean() for df in errors], color=color, marker=marker,
                markersize=3.5, linewidth=1.0, label=label)
    ax.set_xticks(lead_times)
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("MSE over water")
    ax.set_ylim(bottom=0)
    # Above the axes, so it never covers a point.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series))

    savefig(fig, out_dir, "mse_by_lead", width=TEXTWIDTH_IN / 2, height=2.6)
    plt.close(fig)


def mean_residuals(data) -> dict:
    """{season: (mean predicted residual, mean true residual, n dates)}, each [H, W] SIC, land NaN.
    Seasons without samples are left out."""
    by_season = {season: ([], []) for season in SEASONS}
    for pred, target, k, dates in zip(data["predictions"], data["targets"], data["k"], data["times"]):
        month = pd.Timestamp(dates[0]).month  # month of the target
        season = next(s for s, months in SEASONS.items() if month in months)
        by_season[season][0].append(pred - k)
        by_season[season][1].append(target - k)
    # Land is NaN in every field, so a plain mean keeps it NaN
    return {season: (np.mean(p, axis=0), np.mean(t, axis=0), len(p))
            for season, (p, t) in by_season.items() if p}


def plot_mean_residual(data, out_dir):
    """Mean residual k+1 - k per season (columns), predicted (top row) vs truth (bottom row),
    on one fixed symmetric colour scale (RESIDUAL_VMAX). Also prints the mean and RMS of
    their difference."""
    means = mean_residuals(data)
    seasons = [s for s in ("cool", "melt") if s in means]
    if not seasons:
        print("No samples -- skipping mean_residual.")
        return

    # Fixed scale for all panels, so seasons, rows and runs compare directly
    vmax = RESIDUAL_VMAX

    fig, axes = plt.subplots(2, len(seasons), squeeze=False)
    for col, season in enumerate(seasons):
        pred, true, n = means[season]
        for row, field in enumerate((pred, true)):
            img = axes[row, col].imshow(field, cmap=RESIDUAL_CMAP, vmin=-vmax, vmax=vmax,
                                        origin="lower", interpolation="none")
            _clean_image_axis(axes[row, col])
        axes[0, col].set_title(f"{SEASON_TITLES[season]}, $n={n}$")
    axes[0, 0].set_ylabel("Predicted")
    axes[1, 0].set_ylabel("Truth")

    # One inset colorbar per row next to the right-hand map, so it matches the image height;
    # a colorbar spanning several axes breaks savefig's tight_layout.
    for row in range(2):
        cbar = fig.colorbar(img, cax=axes[row, -1].inset_axes([1.04, 0, 0.05, 1]),
                            ticks=[-vmax, 0, vmax], extend="both")
        cbar.ax.set_yticklabels([f"${-vmax:.2f}$", "$0$", f"${vmax:.2f}$"])
        cbar.set_label("SIC change", labelpad=4)

    print("Mean residual, predicted - truth, over water:")
    for season in seasons:
        diff = means[season][0] - means[season][1]
        print(f"  {season}: mean {np.nanmean(diff):+.4f}, RMS {np.sqrt(np.nanmean(diff ** 2)):.4f}")
    print()

    savefig(fig, out_dir, "mean_residual", width=TEXTWIDTH_IN, height=4.8)
    plt.close(fig)


def load_loss_history(run: str | Path) -> tuple[Path, dict] | None:
    """Find and load a run's loss_history.pkl, or None if it has none.

    Accepts the pickle itself, a run directory, or a run name. DDPM.train writes it to
    {output_dir}/{experiment_name}/ (results/), while samples.pkl may sit in samples_dir
    (/work3 on HPC), so a samples directory also falls back to results/<its name>/.
    """
    run = Path(run)
    if run.suffix == ".pkl" and run.name != "samples.pkl":
        candidates = [run]
    else:
        run_dir = run.parent if run.suffix == ".pkl" else run
        candidates = [run_dir / "loss_history.pkl", Path("results") / run_dir.name / "loss_history.pkl"]
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                return path.parent, pickle.load(f)
    return None


def plot_loss_history(history: dict, out_dir):
    """Training vs validation loss per epoch (DDPM: averaged over all timesteps).

    For the UNet the validation persistence loss (k+1 = k) is drawn as a reference line: it is
    the same every epoch, and the model is only useful below it.
    """
    epochs = np.arange(1, len(history["train"]) + 1)
    # Pickles written before "model" was stored are all from the DDPM.
    unet = history.get("model", "ddpm") == "unet"

    fig, ax = plt.subplots()
    ax.plot(epochs, history["train"], color=COLORS[0], marker=MARKERS[0], label="Training")
    if history["val"]:
        ax.plot(epochs, history["val"], color=COLORS[1], marker=MARKERS[1], label="Validation")
    if unet and history.get("val_persistence"):
        ax.axhline(np.mean(history["val_persistence"]), color=MUTED, linewidth=0.8,
                   linestyle=REF_LINESTYLE, label="Persistence (val)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    # Too long for the y-axis at half width, so the formula is the title. x_0 is the
    # standardized residual (k+1 - k) / sigma_r in both models.
    if unet:
        ax.set_title(r"$\|\hat{x}_0 - x_0\|^2$")
    else:
        # The truncated-SNR weight of ConditionalDDPM.compute_loss (not used with unfolded_loss).
        ax.set_title(r"$\min(\max(\mathrm{SNR}_t, 1), 5)\,\|\hat{x}_0 - x_0\|^2$")
    ax.set_ylabel("loss")
    ax.legend()

    savefig(fig, out_dir, "loss_history", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)


def make_loss_history(run, out_dir=None):
    """Draw loss_history for `run` into out_dir (default: next to the pickle), if it has one."""
    found = load_loss_history(run)
    if found is None:
        print(f"No loss_history.pkl for {run} -- skipping loss_history.")
        return
    history_dir, history = found
    out_dir = history_dir if out_dir is None else out_dir
    plot_loss_history(history, out_dir)
    print(f"Wrote {out_dir}/loss_history.pdf")


def make_all(path, row_index: int = 0, grid_indices=None, edge_indices=None, lead: int = 1,
             member: int | None = None):
    run_dir, leads = load_leads(path, member)
    # Training history does not depend on --lead, so it goes next to the samples like calibration.
    make_loss_history(run_dir, run_dir)
    if member is not None:
        # One member of an ensemble: own directory, so the ensemble-mean figures are kept.
        run_dir = run_dir / f"member{member}"
        run_dir.mkdir(exist_ok=True)
    data = leads[lead - 1]
    out_dir = run_dir
    if lead > 1:
        out_dir = out_dir / f"lead{lead}"
        out_dir.mkdir(exist_ok=True)
    print_stats(data)
    plot_sampling_loss(data, out_dir)
    # Every lead in one figure, so it does not depend on --lead and goes next to the pickle.
    plot_calibration(leads, run_dir)
    plot_mse_by_lead(leads, run_dir)
    #plot_forecast_row(data, out_dir, index=row_index)
    #plot_forecast_grid(data, out_dir, indices=grid_indices)
    for season in SEASONS:
        indices = season_indices(data, season)
        if indices:
            plot_forecast_rows(data, out_dir, indices, f"forecast_rows_{season}")
        else:
            print(f"No {season}-season samples -- skipping forecast_rows_{season}.")
    if edge_indices is None:
        edge_indices = {season: edge_index(data, season) for season in ("cool", "melt")}
    else:
        edge_indices = dict(zip(("cool", "melt"), edge_indices))
    plot_ice_edge_error(data, out_dir, edge_indices)
    plot_mean_residual(data, out_dir)
    print(f"Wrote figures to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run name under results/, its directory, or a samples.pkl path")
    parser.add_argument("--row-index", type=int, default=0, help="sample shown in forecast_row")
    parser.add_argument("--grid-indices", type=int, nargs="+", default=None,
                        help="samples shown in forecast_grid (default: first four)")
    parser.add_argument("--edge-indices", type=int, nargs=2, default=None,
                        metavar=("COOL", "MELT"),
                        help="samples shown in ice_edge_error (default: first Mar and Sep)")
    parser.add_argument("--lead", type=int, default=1,
                        help="for a rollout, score the forecast of k+lead (figures go to lead<lead>/)")
    parser.add_argument("--member", type=int, default=None,
                        help="for an ensemble, score only this member instead of the ensemble mean "
                             "(figures go to member<member>/)")
    parser.add_argument("--loss-only", action="store_true",
                        help="only draw loss_history (run name, run dir or loss_history.pkl; no samples.pkl needed)")
    args = parser.parse_args()
    if args.loss_only:
        make_loss_history(args.run)
    else:
        make_all(args.run, row_index=args.row_index, grid_indices=args.grid_indices,
                 edge_indices=args.edge_indices, lead=args.lead, member=args.member)
