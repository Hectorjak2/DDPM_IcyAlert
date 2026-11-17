"""Thesis figure: radial power spectral density of the target SIC and the forecasts of any
number of sampled runs in the land-free Denmark Strait window (plotting.spectral).

Edit ``RUNS`` (folder names under ``results/``) below (display names come from ``LABELS``
in plotting.final_results), then run from the repo root:

    python -m plotting.spectral_comparison

Each curve is the PSD averaged over the target dates that all runs share (runs sampled with
the same ``TrainConfig.sample_seed`` share their first dates), against wavelength 1/k in km.
The target curve comes from the first run. A model curve below the target at a given
wavelength is missing structure at that scale (smoothing).

Writes ``<OUT_DIR>/spectral_comparison.pdf`` (half page width, the same size and styling as
``final_results``' calibration_comparison, so the two sit side by side).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import load_leads
from plotting.final_results import _family_color, _name, _tex_name
from plotting.spectral import find_landfree_region, mean_spacing_km, radial_psd
from utils.dataloader import CARRA2_DATA_ROOT, CROPPED_X, CROPPED_Y

# Folder names under results/, in the order they appear in the legend. Display names come
# from plotting.final_results.LABELS so the legend matches calibration_comparison.
RUNS = [
    "ddpm-corrected-land-256",
    "ddpm-LC-256-ensemble",
    "ensembles/member_5",
    "deep-ensemble",
]
LEAD = 1                   # lead time to score; every run needs at least this many leads
OUT_DIR = "final_figures"  # where the figure is written

# Styling as in final_results.plot_calibration_comparison: one colour per model family
# (DDPM COLORS[0], UNet COLORS[1]); single members dashed and slightly transparent,
# ensemble means solid. The target is solid black, drawn on top.
TARGET_COLOR = "black"
MEMBER_ALPHA = 0.6

# The WEST window inside the full-domain store (utils/prepare_west_zarr.py). Only the
# full-domain store still has latitude/longitude.
WEST_Y0, WEST_X0 = 300, 884

# Decades of empty space below the lowest curve, room for the legend.
LEGEND_ROOM_DECADES = 3

# Wavelength ticks [km].
WAVELENGTH_TICKS = [300, 100, 50, 25, 10, 5]


def cropped_lat_lon():
    """Latitude and longitude [degrees] of the CARRA_CROPPED grid."""
    ds = xr.open_zarr(Path(CARRA2_DATA_ROOT) / "dataset.zarr", decode_coords="all")
    y = slice(WEST_Y0 + CROPPED_Y.start, WEST_Y0 + CROPPED_Y.stop)
    x = slice(WEST_X0 + CROPPED_X.start, WEST_X0 + CROPPED_X.stop)
    ds = ds.isel(y=y, x=x)
    return ds.latitude.values, ds.longitude.values


def load_psds(run, lat, lon, window=None):
    """Per-date PSDs of one run's predictions (and targets), so the (large) pickle can be
    freed before the next one is loaded.

    ``window`` is ``(region, dx_km)``; when None it is found from this run's first target.
    Returns ``(k, predictions, targets, window)`` with ``predictions`` and ``targets`` dicts
    target date -> PSD.
    """
    _, leads = load_leads(run)
    if len(leads) < LEAD:
        raise ValueError(f"{run} has {len(leads)} lead time(s), LEAD = {LEAD}.")
    data = leads[LEAD - 1]
    del leads

    if window is None:
        region = find_landfree_region(data["targets"][0], lat)
        dx_km = mean_spacing_km(lat, lon, region)
        print(f"Window {region}, mean spacing {dx_km:.3f} km")
        window = (region, dx_km)
    region, dx_km = window

    dates = [pd.Timestamp(t[0]) for t in data["times"]]
    k = None
    predictions, targets = {}, {}
    # An ensemble (ensemble_samples.pkl, or any pickle with M > 1 members per date) is
    # scored by its ensemble mean, data["predictions"]; averaging members that disagree on
    # small scales removes power there, which is what the figure should show. A single
    # forecast is its only member.
    n_members = data["members"][0].shape[0]
    if n_members > 1:
        print(f"{run}: ensemble of {n_members} members, plotting the ensemble mean")
    for date, pred, target in zip(dates, data["predictions"], data["targets"]):
        k, predictions[date] = radial_psd(pred, dx_km, region)
        _, targets[date] = radial_psd(target, dx_km, region)
    return k, predictions, targets, window


def make_figure(runs, out_dir):
    if not runs:
        raise ValueError("RUNS is empty -- list the folders under results/ to compare.")

    lat, lon = cropped_lat_lon()
    window = None
    loaded = []  # (run, predictions, targets)
    for run in runs:
        print(f"Loading {run} ...")
        k, predictions, targets, window = load_psds(run, lat, lon, window)
        loaded.append((run, predictions, targets))

    common = sorted(set.intersection(*(set(p) for _, p, _ in loaded)))
    if not common:
        raise ValueError("The runs share no target dates (different sample_seed?).")
    for run, predictions, _ in loaded:
        if len(predictions) != len(common):
            print(f"{run} has {len(predictions)} dates, {len(common)} shared with all runs.")
    print(f"Using the {len(common)} test dates all runs share.")

    truth_by_date = loaded[0][2]
    # (label, psd, line style kwargs), styled as in final_results.plot_calibration_comparison.
    # The target comes first in the legend and is drawn on top of the models (zorder).
    series = [("Target", np.mean([truth_by_date[d] for d in common], axis=0),
               dict(color=TARGET_COLOR, linestyle="-", zorder=3))]
    for run, predictions, _ in loaded:
        psd = np.mean([predictions[d] for d in common], axis=0)
        name = _name(run).lower()
        member = "member" in name
        series.append((_tex_name(run), psd,
                       dict(color=_family_color(name), linestyle="--" if member else "-",
                            alpha=MEMBER_ALPHA if member else 1.0)))

    wavelength = 1 / k  # [km]
    fig, ax = plt.subplots()
    # No markers: they suggested the PSD is only computed at the marked points. Colour and
    # line style tell the curves apart; each curve is all radial wavenumber bins joined.
    for label, psd, style in series:
        ax.loglog(wavelength, psd, linewidth=1.25, label=label, **style)

    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel(r"PSD")
    ax.set_xlim(wavelength[0], wavelength[-1])  # large scales on the left
    # Room below the curves (in decades, the axis is log) so the legend in the lower left
    # does not sit on the curves.
    lo = min(psd.min() for _, psd, _ in series)
    hi = max(psd.max() for _, psd, _ in series)
    ax.set_ylim(lo / 10 ** LEGEND_ROOM_DECADES, hi * 2)
    ax.set_xticks(WAVELENGTH_TICKS, labels=[str(t) for t in WAVELENGTH_TICKS])
    ax.minorticks_off()

    # The PSD falls from the large scales (upper left) to the small ones (lower right), so
    # the legend goes inside the axes in the empty lower-left corner, styled as in
    # calibration_comparison.
    ax.legend(loc="lower left", handlelength=1.4, handletextpad=0.3, labelspacing=0.3,
              borderaxespad=0.1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir, "spectral_comparison", width=TEXTWIDTH_IN / 2, height=2.8)
    plt.close(fig)
    print(f"Wrote {out_dir / 'spectral_comparison.pdf'}")


if __name__ == "__main__":
    make_figure(RUNS, OUT_DIR)
