"""Seasonal cycle of sea-ice extent in the cropped CARRA2 region: the training range against the
validation and test years.

Extent follows the NSIDC definition: the area of water cells with SIC > 0.15. Each CARRA2 cell is
taken as 2.5 km x 2.5 km = 6.25 km^2 (nominal grid spacing; the true cell area varies slightly with
latitude on the polar stereographic grid). Land is NaN and never counts.

Over day of year, the training years (training slice in config.py) give a shaded min-max band and
a mean. The validation and test splits are drawn as means on top, plus each test year as a thin
line, so the figure shows where the test years leave the range the model was trained on. Each mean
is the mean over that split's years for each day, smoothed with a centred SMOOTH_DAYS-day rolling
mean (validation has only 3 years, so its raw mean is noisy). The band is not smoothed. Day 366 is
dropped (leap years only). Days in no split (the gaps Nov-Dec 2016 and Nov-Dec 2019) are left out.

The daily extent series reads every day of the crop (~30 GB), so it is cached to a CSV; delete the
cache, or pass --recompute, to read the data again.

Run from the repo root:  python -m plotting.sea_ice_extent [--recompute]
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from config import TEST_SLICE, TRAINING_SLICE, VALIDATION_SLICE
from plotting.config import savefig, TEXTWIDTH_IN, COLORS, muted
from utils.dataloader import CARRA2_DATA_ROOT, CROPPED_X, CROPPED_Y

SIC_THRESHOLD = 0.15
CELL_AREA_KM2 = 2.5 * 2.5
SMOOTH_DAYS = 7
CACHE = "results/sea_ice_extent_cropped.csv"

TRAINING_COLOR, VALIDATION_COLOR, TEST_COLOR = COLORS[0], COLORS[7], COLORS[1]


def daily_extent(cache: str = CACHE, recompute: bool = False) -> pd.Series:
    """Extent in km^2 per day over the cropped region, indexed by date."""
    if os.path.exists(cache) and not recompute:
        return pd.read_csv(cache, index_col=0, parse_dates=True)["extent_km2"]
    da = xr.open_zarr(f"{CARRA2_DATA_ROOT}/west.zarr")["siconc"].isel(y=CROPPED_Y, x=CROPPED_X)
    n_cells = (da > SIC_THRESHOLD).sum(("y", "x")).compute()  # NaN > 0.15 is False, so land drops out
    extent = pd.Series(n_cells.values * CELL_AREA_KM2, index=pd.DatetimeIndex(da.time.values),
                       name="extent_km2")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    extent.to_csv(cache)
    return extent


def by_doy(extent: pd.Series, s: slice) -> pd.DataFrame:
    """Extent in 10^6 km^2 within slice s: rows day of year (1-365), columns year."""
    e = extent.loc[s.start:s.stop] / 1e6
    df = pd.DataFrame({"doy": e.index.dayofyear, "year": e.index.year, "extent": e.to_numpy()})
    return df[df.doy <= 365].pivot(index="doy", columns="year", values="extent")


def smoothed_mean(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=1).rolling(SMOOTH_DAYS, center=True, min_periods=1).mean()


def _years(s: slice) -> str:
    return f"{pd.Timestamp(s.start).year}--{pd.Timestamp(s.stop).year}"


def plot_sea_ice_extent(extent: pd.Series, out_dir: str = "figures"):
    train, val, test = (by_doy(extent, s) for s in (TRAINING_SLICE, VALIDATION_SLICE, TEST_SLICE))
    fig, ax = plt.subplots()

    ax.fill_between(train.index, train.min(axis=1), train.max(axis=1), color=muted(TRAINING_COLOR, 0.6),
                    linewidth=0, label=f"Training range ({_years(TRAINING_SLICE)})")
    for i, year in enumerate(test.columns):
        ax.plot(test.index, test[year], color=TEST_COLOR, linewidth=0.6, alpha=0.4,
                label="Test years" if i == 0 else None)
    ax.plot(train.index, smoothed_mean(train), color=TRAINING_COLOR, label="Training mean")
    ax.plot(val.index, smoothed_mean(val), color=VALIDATION_COLOR,
            label=f"Validation mean ({_years(VALIDATION_SLICE)})")
    ax.plot(test.index, smoothed_mean(test), color=TEST_COLOR, label=f"Test mean ({_years(TEST_SLICE)})")
    ax.legend(loc="lower left")

    month_starts = pd.date_range("2001-01-01", periods=12, freq="MS")  # non-leap year for day-of-year ticks
    ax.set_xticks(month_starts.dayofyear, [f"1 {m}" for m in month_starts.strftime("%b")])
    ax.set_xlim(1, 365)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(r"Extent ($10^6$ km$^2$)")

    savefig(fig, out_dir, "sea_ice_extent", width=TEXTWIDTH_IN, height=3.6)
    plt.close(fig)
    print(f"Wrote {out_dir}/sea_ice_extent.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true", help="ignore the cached daily extent")
    args = parser.parse_args()
    plot_sea_ice_extent(daily_extent(recompute=args.recompute))
