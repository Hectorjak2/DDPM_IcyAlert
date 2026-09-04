import torch
from torch.utils.data import Dataset
import xarray as xr

allowed_variables = ["siconc"]

class CARRA2(Dataset):
    def __init__(self, selected_variable: str, device: str, time_slice: slice | None = None, WEST: bool = True, TEST: bool = False, batch_dim: bool = True):
        """
        This class is a PyTorch Dataset for the CARRA2 dataset.
        It allows for loading a specific variable, selecting a time slice.
        By default it will load the CARRA2-WEST area, but this can be changed by setting WEST to False.

        Args:
            selected_variable (str): The variable to load. Must be one of the allowed_variables.
            time_slice (slice | None): A slice object to select a time range. If None, all times are loaded.
            WEST (bool): If True, loads the CARRA2-WEST area. If False, loads the full dataset.
            TEST (bool): If True, loads the test area. (smaller area for testing purposes)
            batch_dim (bool): If True (default), __getitem__ returns [1, 1, H, W] with a leading
                batch dimension. Set to False for [1, H, W] (e.g. when wrapping in a torch DataLoader).
        """
        self.batch_dim = batch_dim
        self.device = device

        if selected_variable not in allowed_variables:
            raise ValueError(f"{selected_variable} not in {allowed_variables}")
        ds = xr.open_zarr("data/CARRA2_MONTHLY/dataset.zarr", decode_coords="all")
        da = ds[selected_variable]

        if time_slice is not None:
            da = da.sel(time=time_slice)

        # TEST takes precedence so it can be used together with the default WEST=True.
        if TEST:
            #CARRA2 TEST AREA (128 x 128):
            self.da = da.isel(y=slice(1272, 1400), x=slice(800, 928))
        elif WEST:
            #CARRA2-WEST AREA:
            self.da = da.isel(y=slice(300, 1520), x=slice(880, 2100))
        else:
            self.da = da

    def __len__(self):
        return self.da.sizes["time"]

    def __getitem__(self, idx):
        arr = self.da.isel(time=idx).values
        t = torch.from_numpy(arr).float().unsqueeze(0)  # [1, H, W]
        if self.batch_dim:
            t = t.unsqueeze(0)  # [1, 1, H, W]
        return t.to(self.device)

if __name__ == "__main__":
    train_ds = CARRA2("siconc", time_slice=slice("2012-01", "2013-12"))
    test_ds  = CARRA2("siconc", time_slice=slice("2014-01", "2014-12"))
