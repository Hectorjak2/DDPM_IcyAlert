import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from torchvision import datasets, transforms
from models.models import DatasetChoice

allowed_variables = ["siconc"]
CARRA_WEST_MASK = None
CARRA_TEST_MASK = None

# Overridden on HPC (see run.sh) since the real data lives under /work3, not
# the repo checkout. Defaults to the repo-relative path used for local dev.
CARRA2_DATA_ROOT = os.environ.get("CARRA2_DATA_ROOT", "data/CARRA2_DAILY")


def to_model_range(x: torch.Tensor) -> torch.Tensor:
    """Map data in [0, 1] to the [-1, 1] range the diffusion process assumes."""
    return x * 2.0 - 1.0


def to_data_range(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_model_range`: map [-1, 1] back to [0, 1]."""
    return (x + 1.0) / 2.0

class CARRA2(Dataset):
    def __init__(self, selected_variable: str, device: str, time_slice: slice | list[slice] | None = None, area: DatasetChoice = DatasetChoice.CARRA_WEST, batch_dim: bool = True):
        """
        This class is a PyTorch Dataset for the CARRA2 dataset.
        It allows for loading a specific variable, selecting a time slice.
        By default it will load the CARRA2-WEST area, but this can be changed by setting WEST to False.

        Args:
            selected_variable (str): The variable to load. Must be one of the allowed_variables.
            time_slice (slice | list[slice] | None): A slice, or a list of slices (e.g. one
                per year, for a recurring season that isn't itself a contiguous range), to
                select a time range. If None, all times are loaded. A list is concatenated
                along time, in the order given.
            batch_dim (bool): If True (default), __getitem__ returns [1, 1, H, W] with a leading
                batch dimension. Set to False for [1, H, W] (e.g. when wrapping in a torch DataLoader).
        """
        self.batch_dim = batch_dim
        self.device = device
        self.name = f"CARRA2-{selected_variable}"  # Add this
        self.area = area

        if selected_variable not in allowed_variables:
            raise ValueError(f"{selected_variable} not in {allowed_variables}")
        # data/CARRA2_DAILY/west.zarr is a pre-cropped, pre-rechunked copy of the
        # full-domain dataset.zarr, produced by utils/prepare_west_zarr.py. It
        # already *is* the WEST bounding box (y=300:1516, x=884:2100 in the full
        # domain), so WEST needs no further slicing here; TEST is a small crop of
        # it, expressed in WEST-local coordinates. See docs/architecture.md,
        # "Region Split: WEST vs. TEST".
        ds = xr.open_zarr(f"{CARRA2_DATA_ROOT}/west.zarr", decode_coords="all")
        da = ds[selected_variable]

        if isinstance(time_slice, list):
            da = xr.concat([da.sel(time=s) for s in time_slice], dim="time")
        elif time_slice is not None:
            da = da.sel(time=time_slice)

        if self.area == DatasetChoice.CARRA_TEST: 
            self.da = da.isel(y=slice(750, 1006), x=slice(100, 356))
        elif self.area == DatasetChoice.CARRA_WEST: 
            self.da = da

        else:
            raise ValueError(
                "The full domain is no longer available through CARRA2 (only "
                "data/CARRA2_DAILY/west.zarr is loaded). Use WEST=True or TEST=True, "
                "or open data/CARRA2_DAILY/dataset.zarr directly for full-domain access."
            )

        # The land/sea geometry is static in CARRA2, so derive the mask once from
        # the first time step instead of recomputing it per sample. Stored as
        # [1, H, W] float with 1.0 = water, 0.0 = land.
        first = self.da.isel(time=0).values
        self.finite_mask = (
            torch.from_numpy(np.isfinite(first)).float().unsqueeze(0).to(device)
        )

    def __len__(self):
        return self.da.sizes["time"]

    def __getitem__(self, idx):
        arr = self.da.isel(time=idx).values
        t = torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]
        # siconc is a fraction in [0, 1]; the diffusion process assumes roughly
        # zero-mean unit-scale data, so shift to [-1, 1]. NaN (land) stays NaN and
        # is handled by the loss mask.
        t = to_model_range(t)
        if self.batch_dim:
            t = t.unsqueeze(0)  # [1, 1, H, W]
        return t.to(self.device)


class CARRA2Forecast(CARRA2):
    def __init__(self, selected_variable: str, device: str, time_slice: slice | list[slice] | None = None, area: DatasetChoice = DatasetChoice.CARRA_WEST, batch_dim: bool = True):
        """
        A forecasting variant of CARRA2: each sample is a triplet of days one
        calendar *month* apart (k-1, k, k+1) -- e.g. 5 Sep, 5 Oct, 5 Nov --
        returned as (target, context) where target = day k+1 and context =
        stacked (day k, day k-1).

       

        `time_slice` may be a list of slices (e.g. one melt/cool season per
        year), which are not contiguous with each other. This is handled the
        same way as the month-end case: an anchor whose k-1 or k+1 date falls
        outside the included months (e.g. Aug -> Sep is fine but Sep -> Oct
        isn't, for an Apr-Sep melt season) simply has no match and is
        dropped, so a window can never silently splice together dates from
        two different blocks/years.
        """
        super().__init__(selected_variable, device, time_slice=time_slice, area=area, batch_dim=batch_dim)

        times = self.da.time.values
        time_to_idx = {t: i for i, t in enumerate(times)}

        windows = []
        for i, t in enumerate(times):
            ts = pd.Timestamp(t)
            km1_date = np.datetime64(ts - pd.DateOffset(months=1))
            kp1_date = np.datetime64(ts + pd.DateOffset(months=1))
            km1_i = time_to_idx.get(km1_date)
            kp1_i = time_to_idx.get(kp1_date)
            if km1_i is not None and kp1_i is not None:
                windows.append((km1_i, i, kp1_i))
        self._windows = np.array(windows, dtype=np.int64).reshape(-1, 3)

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        km1_i, k_i, kp1_i = self._windows[idx]
        km1 = super().__getitem__(km1_i)
        k = super().__getitem__(k_i)
        kp1 = super().__getitem__(kp1_i)
        context = torch.cat([kp1, k, km1], dim=0 if not self.batch_dim else 1)
        return context #k+1, k, k-1

    def get_dates(self, idx) -> tuple[np.datetime64, np.datetime64, np.datetime64]:
        """Return the (target, k, k-1) calendar dates backing sample `idx`, matching __getitem__'s (kp1, context=[k, km1]) layout."""
        km1_i, k_i, kp1_i = self._windows[idx]
        times = self.da.time.values
        return times[kp1_i], times[k_i], times[km1_i]


class FashionMNIST(Dataset):
    def __init__(self, train: bool = True, device: str = "cpu", batch_dim: bool = True):
        """
        This class is a PyTorch Dataset for the Fashion MNIST dataset.
        It allows for loading training or test data with automatic downloading.

        Args:
            train (bool): If True, loads the training set. If False, loads the test set.
            device (str): The device to load the data onto (e.g., "cpu" or "cuda").
            batch_dim (bool): If True (default), __getitem__ returns [1, 1, H, W] with a leading
                batch dimension. Set to False for [1, H, W] (e.g. when wrapping in a torch DataLoader).
        """
        self.batch_dim = batch_dim
        self.device = device
        self.name = f"FashionMNIST-{'train' if train else 'test'}"  # Add this
        self.finite_mask = None  # No land/sea geometry for this toy dataset


        # Define transforms to convert images to tensors
        transform = transforms.Compose([transforms.ToTensor()])

        # Download and load the data
        self.data = datasets.FashionMNIST(
            root="./data", train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        # img is already [1, 28, 28] from ToTensor(), i.e. in [0, 1] -> shift to [-1, 1]
        img = to_model_range(img)
        if self.batch_dim:
            img = img.unsqueeze(0)  # [1, 1, 28, 28]
        return img.to(self.device)

if __name__ == "__main__":
    TRAINING_SLICE = slice("2000-01-01", "2019-10-31")
    TEST_SLICE = slice("2020-01-01", "2025-12-31")
    MELT_SEASON = [slice(f"{year}-04-01", f"{year}-09-30") for year in range(2020, 2025)]
    COOL_SEASON = [slice(f"{year}-10-01", f"{year + 1}-03-31") for year in range(2020, 2024)]
    train_ds = CARRA2("siconc", area=DatasetChoice.CARRA_TEST, device="mps", batch_dim=False)

    train_ds = CARRA2Forecast("siconc", "mps", time_slice=TRAINING_SLICE, batch_dim=False)
    test_ds = CARRA2Forecast("siconc", "mps", time_slice=TEST_SLICE, batch_dim=False)
    melt_ds = CARRA2Forecast("siconc", "mps", time_slice=MELT_SEASON, batch_dim=False)
    cool_ds = CARRA2Forecast("siconc", "mps", time_slice=COOL_SEASON, batch_dim=False)