"""Thesis figure: radial power spectral density of the target SIC, the UNet forecasts and
the DDPM forecasts in the land-free Denmark Strait window (plotting.spectral).

Each curve is the PSD averaged over the test dates both runs share (the UNet and DDPM
pickles are drawn with the same ``sample_seed``, so a shorter run's dates are a prefix of
the longer's), against wavelength 1/k in km. A model curve below the target at a given
wavelength is missing structure at that scale (smoothing).

Writes ``<out>/spectral_comparison.pdf``. Run from the repo root:
    python -m plotting.spectral_comparison [unet_run] [ddpm_run] [--out figures]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MARKERS
from plotting.evaluation import load_leads
from plotting.spectral import find_landfree_region, mean_spacing_km, radial_psd
from utils.dataloader import CARRA2_DATA_ROOT, CROPPED_X, CROPPED_Y

# The WEST window inside the full-domain store (utils/prepare_west_zarr.py). Only the
# full-domain store still has latitude/longitude.
WEST_Y0, WEST_X0 = 300, 884

# Wavelength ticks [km].
WAVELENGTH_TICKS = [300, 100, 50, 25, 10, 5]


def cropped_lat_lon():
    """Latitude and longitude [degrees] of the CARRA_CROPPED grid."""
    ds = xr.open_zarr(Path(CARRA2_DATA_ROOT) / "dataset.zarr", decode_coords="all")
    y = slice(WEST_Y0 + CROPPED_Y.start, WEST_Y0 + CROPPED_Y.stop)
    x = slice(WEST_X0 + CROPPED_X.start, WEST_X0 + CROPPED_X.stop)
    ds = ds.isel(y=y, x=x)
    return ds.latitude.values, ds.longitude.values


def mean_psd(fields, dx_km, region):
    """Wavenumbers and PSD averaged over a list of [H, W] fields."""
    psds = [radial_psd(f, dx_km, region) for f in fields]
    return psds[0][0], np.mean([p for _, p in psds], axis=0)


def make_figure(unet_run, ddpm_run, out_dir):
    _, unet = load_leads(unet_run)
    _, ddpm = load_leads(ddpm_run)
    unet, ddpm = unet[0], ddpm[0]  # lead 1

    n = min(len(unet["times"]), len(ddpm["times"]))
    if [t[0] for t in unet["times"][:n]] != [t[0] for t in ddpm["times"][:n]]:
        raise ValueError("The two runs were not sampled on the same dates (different sample_seed?).")
    print(f"Using the {n} test dates both runs share.")

    lat, lon = cropped_lat_lon()
    region = find_landfree_region(ddpm["targets"][0], lat)
    dx_km = mean_spacing_km(lat, lon, region)
    print(f"Window {region}, mean spacing {dx_km:.3f} km")

    k, truth = mean_psd(ddpm["targets"][:n], dx_km, region)
    _, psd_unet = mean_psd(unet["predictions"][:n], dx_km, region)
    _, psd_ddpm = mean_psd(ddpm["predictions"][:n], dx_km, region)

    wavelength = 1 / k  # [km]
    fig, ax = plt.subplots()
    # Colours match plotting.model_comparison (UNet teal, DDPM plum); the target is charcoal.
    series = [("Target", truth, COLORS[5], MARKERS[0]),
              ("UNet", psd_unet, COLORS[0], MARKERS[1]),
              ("DDPM", psd_ddpm, COLORS[7], MARKERS[3])]
    for label, psd, color, marker in series:
        ax.loglog(wavelength, psd, color=color, marker=marker, markevery=(2, 6), markersize=3,
                  linewidth=1.2, label=label)

    ax.set_xlabel("wavelength (km)")
    ax.set_ylabel(r"PSD")
    ax.set_xlim(wavelength[0], wavelength[-1])  # large scales on the left
    ax.set_xticks(WAVELENGTH_TICKS, labels=[str(t) for t in WAVELENGTH_TICKS])
    ax.minorticks_off()

    ax.legend(loc="lower left")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    savefig(fig, out_dir, "spectral_comparison", width=TEXTWIDTH_IN / 2, height=TEXTWIDTH_IN / 2)  # square
    plt.close(fig)
    print(f"Wrote {out_dir / 'spectral_comparison.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("unet_run", nargs="?", default="member3-unet",
                        help="UNet run: name under results/, its directory, or a samples.pkl path")
    parser.add_argument("ddpm_run", nargs="?", default="ddpm-titanfull-shif128-ip",
                        help="DDPM run, as unet_run")
    parser.add_argument("--out", default="figures", help="directory the figure is written to")
    args = parser.parse_args()
    make_figure(args.unet_run, args.ddpm_run, args.out)
