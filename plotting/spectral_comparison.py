"""Thesis figure: radial power spectral density of the target SIC and the forecasts of any
number of sampled runs in the land-free Denmark Strait window (plotting.spectral).

Edit ``RUNS`` (folder names under ``results/``) and optionally ``LABELS`` below, then run
from the repo root:

    python -m plotting.spectral_comparison

Each curve is the PSD averaged over the target dates that all runs share (runs sampled with
the same ``TrainConfig.sample_seed`` share their first dates), against wavelength 1/k in km.
The target curve comes from the first run. A model curve below the target at a given
wavelength is missing structure at that scale (smoothing).

Writes ``<OUT_DIR>/spectral_comparison.pdf``.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plotting.config import savefig, TEXTWIDTH_IN, COLORS
from plotting.evaluation import load_leads
from plotting.spectral import find_landfree_region, mean_spacing_km, radial_psd
from utils.dataloader import CARRA2_DATA_ROOT, CROPPED_X, CROPPED_Y

# Folder names under results/, in the order they appear in the legend.
RUNS = [
    "deep-ensemble",
    "ddpm-corrected-land-64",
    "ddpm-titanfull-shif64",
    "ddpm-corrected-land-128",
]

# Optional display names; a run not listed here is shown by its folder name.
LABELS = {
    "deep-ensemble": "UNet Ensemble",
    "ddpm-corrected-land-64": "LC 64", 
    "ddpm-corrected-land-64-ip": "LC 64 IP",
    "ddpm-corrected-land-128": "LC 128",
    "ddpm-corrected-land-128-ip": "LC 128 IP"
}
LEAD = 1             # lead time to score; every run needs at least this many leads
OUT_DIR = "figures"  # where the figure is written

# The target is black; runs take these colours in RUNS order (the first two match
# plotting.model_comparison: UNet teal, DDPM plum). Runs whose folder name contains
# "ensemble" are drawn solid, all other runs dashed.
TARGET_COLOR = "black"
RUN_COLORS = [COLORS[i] for i in (0, 7, 1, 2, 3, 6)]
MODEL_ALPHA = 0.8

# The WEST window inside the full-domain store (utils/prepare_west_zarr.py). Only the
# full-domain store still has latitude/longitude.
WEST_Y0, WEST_X0 = 300, 884

# Wavelength ticks [km].
WAVELENGTH_TICKS = [300, 100, 50, 25, 10, 5]


def _tex_name(run: str) -> str:
    """Display name for a run, safe for usetex."""
    label = LABELS.get(run, run)
    for char in "_%&#":
        label = label.replace(char, "\\" + char)
    return label


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
    for date, pred, target in zip(dates, data["predictions"], data["targets"]):
        k, predictions[date] = radial_psd(pred, dx_km, region)
        _, targets[date] = radial_psd(target, dx_km, region)
    return k, predictions, targets, window


def make_figure(runs, out_dir):
    if not runs:
        raise ValueError("RUNS is empty -- list the folders under results/ to compare.")
    if len(runs) > len(RUN_COLORS):
        raise ValueError(f"At most {len(RUN_COLORS)} runs fit the colour cycle, got {len(runs)}.")

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
    # (label, psd, line style kwargs). The target is solid black, opaque and drawn on top;
    # models are slightly transparent, ensembles solid and everything else dashed.
    series = [("Target", np.mean([truth_by_date[d] for d in common], axis=0),
               dict(color=TARGET_COLOR, linestyle="-", zorder=3))]
    for i, (run, predictions, _) in enumerate(loaded):
        psd = np.mean([predictions[d] for d in common], axis=0)
        linestyle = "-" if "ensemble" in run else "--"
        series.append((_tex_name(run), psd,
                       dict(color=RUN_COLORS[i], linestyle=linestyle, alpha=MODEL_ALPHA)))

    wavelength = 1 / k  # [km]
    fig, ax = plt.subplots()
    # No markers: they suggested the PSD is only computed at the marked points. Colour and
    # line style tell the curves apart; each curve is all radial wavenumber bins joined.
    for label, psd, style in series:
        ax.loglog(wavelength, psd, linewidth=1.2, label=label, **style)

    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel(r"PSD")
    ax.set_xlim(wavelength[0], wavelength[-1])  # large scales on the left
    ax.set_xticks(WAVELENGTH_TICKS, labels=[str(t) for t in WAVELENGTH_TICKS])
    ax.minorticks_off()

    # Legend in a box below the axes; tight_layout shrinks the axes to keep the figure square.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=True,
              handlelength=1.5, columnspacing=1.0, borderaxespad=0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir, "spectral_comparison", width=TEXTWIDTH_IN / 2, height=TEXTWIDTH_IN / 2)  # square
    plt.close(fig)
    print(f"Wrote {out_dir / 'spectral_comparison.pdf'}")


if __name__ == "__main__":
    make_figure(RUNS, OUT_DIR)
