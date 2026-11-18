"""Climatology baseline: forecast k+1 as the mean SIC of its calendar month over the
training period.

Fitting reads every day of TRAINING_SLICE, so it is done once (on HPC, where the full
data lives) and cached as ``climatology_<AREA>.npy`` next to the data. Evaluation then
only loads that file:

    python -m models.climatology --area CARRA_WEST
"""
import argparse
from functools import cache
from pathlib import Path

import numpy as np

from config import TrainConfig
from models.models import DatasetChoice
from utils.dataloader import CARRA2, CARRA2_DATA_ROOT, CROPPED_X, CROPPED_Y


def cache_path(area: DatasetChoice) -> Path:
    if area == DatasetChoice.CARRA_CROPPED:
        return Path(CARRA2_DATA_ROOT) / "climatology_cropped.npy"
    return Path(CARRA2_DATA_ROOT) / f"climatology_{area.value}.npy"


class Climatology:
    def __init__(self, means: np.ndarray):
        self.means = means  # [12, H, W] in data range [0, 1], Jan first, land NaN

    @classmethod
    def fit(cls, area: DatasetChoice = DatasetChoice.CARRA_WEST, time_slice: slice = TrainConfig().training_slice):
        da = CARRA2("siconc", "cpu", time_slice=time_slice, area=area).da
        # One month at a time, so only one month's mean is held in memory at once.
        means = []
        for month in range(1, 13):
            print(f"Climatology: month {month}/12 ...")
            means.append(da.sel(time=da.time.dt.month == month).mean("time").values)
        return cls(np.stack(means).astype(np.float32))

    def save(self, path: str | Path):
        np.save(path, self.means)

    @classmethod
    def load(cls, path: str | Path):
        return cls(np.load(path))

    @classmethod
    def for_shape(cls, shape: tuple[int, int]):
        """Load the cached climatology whose grid matches ``shape``, or None if none has
        been fitted for it."""
        for area in (DatasetChoice.CARRA_CROPPED, DatasetChoice.CARRA_WEST, DatasetChoice.CARRA_TEST):
            path = cache_path(area)
            if path.exists():
                clim = cls.load(path)
                # The climatology is a per-pixel mean, so a WEST-sized fit cropped to the
                # CARRA_CROPPED window equals one fitted on the crop directly
                if area == DatasetChoice.CARRA_CROPPED and clim.means.shape[1:] != tuple(shape):
                    clim = cls(clim.means[:, CROPPED_Y, CROPPED_X])
                if clim.means.shape[1:] == tuple(shape):
                    return clim
        return None

    def predict(self, month: int) -> np.ndarray:
        """Forecast for a target date in calendar ``month`` (1-12, Jan=1)."""
        return self.means[month - 1]


@cache
def _siconc_for_shape(shape: tuple[int, int]):
    """The daily siconc DataArray (data range [0, 1], land NaN), cropped to the area whose grid
    is ``shape`` (CARRA_CROPPED, CARRA_WEST or CARRA_TEST)."""
    import xarray as xr
    da = xr.open_zarr(f"{CARRA2_DATA_ROOT}/west.zarr")["siconc"]
    crops = {DatasetChoice.CARRA_CROPPED: (CROPPED_Y, CROPPED_X),
             DatasetChoice.CARRA_WEST: (slice(None), slice(None)),
             DatasetChoice.CARRA_TEST: (slice(750, 1006), slice(100, 356))}  # as CARRA2
    for y, x in crops.values():
        cropped = da.isel(y=y, x=x)
        if (cropped.sizes["y"], cropped.sizes["x"]) == tuple(shape):
            return cropped
    raise ValueError(f"No CARRA2 area has grid {shape}.")


def ensemble_members(date, shape: tuple[int, int], n_years: int = 10) -> np.ndarray:
    """Climatological ensemble for target ``date``: the observed SIC on the same calendar day
    in each of the ``n_years`` years before the target year (sliding window; all of it is
    observed by forecast time). 29 Feb falls back to 28 Feb in non-leap years.

    Returns [n_years, H, W] in data range [0, 1], land NaN, oldest year first.
    """
    import pandas as pd
    date = pd.Timestamp(date)
    days = []
    for year in range(date.year - n_years, date.year):
        day = 28 if (date.month, date.day) == (2, 29) else date.day
        days.append(np.datetime64(pd.Timestamp(year, date.month, day).normalize(), "ns"))
    return _siconc_for_shape(tuple(shape)).sel(time=days).values.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--area", default=DatasetChoice.CARRA_CROPPED.value,
                        choices=[DatasetChoice.CARRA_CROPPED.value, DatasetChoice.CARRA_WEST.value,
                                 DatasetChoice.CARRA_TEST.value])
    args = parser.parse_args()
    area = DatasetChoice(args.area)
    Climatology.fit(area).save(cache_path(area))
    print(f"Wrote {cache_path(area)}")
