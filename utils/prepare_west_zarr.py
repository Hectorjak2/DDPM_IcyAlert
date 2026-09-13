"""One-off preprocessing script: crop the full-domain CARRA2 zarr store down to the
WEST region and write it as its own, smaller, better-chunked zarr store.

Why this exists (see docs/architecture.md, "Region Split: WEST vs. TEST"):
- The full-domain store (data/CARRA2_DAILY/dataset.zarr) is 27GB on disk, but WEST
  training only ever touches ~18% of its pixels (1216x1216 out of 2880x2880).
- Its chunking is one chunk per full 2880x2880 frame, so reading a WEST/TEST crop
  still decompresses the *entire* frame per timestep. Rechunking to one chunk per
  cropped frame makes every __getitem__ read touch exactly the pixels it needs.

Run this manually, once (or whenever the WEST bounding box changes):
    python utils/prepare_west_zarr.py

Not wired into main.py -- this is a data-prep step, not part of training.
"""

import subprocess

import xarray as xr

SRC_PATH = "data/CARRA2_DAILY/dataset.zarr"
DST_PATH = "data/CARRA2_DAILY/west.zarr"

# Must match the WEST bounds in utils/dataloader.py's CARRA2 class.
WEST_Y = slice(300, 1516)
WEST_X = slice(884, 2100)


def main():
    ds = xr.open_zarr(SRC_PATH, decode_coords="all")

    # Only siconc + time are used by training; drop the rest (latitude/longitude
    # are static 2880x2880 grids -- 66MB each -- and valid_time/step/surface are
    # unused metadata).
    da = ds["siconc"].isel(y=WEST_Y, x=WEST_X)
    da = da.reset_coords(drop=True) if da.coords.keys() - {"time"} else da
    ds_west = da.to_dataset(name="siconc")

    # One chunk per frame, matching the per-sample access pattern in
    # CARRA2.__getitem__ (a single da.isel(time=idx).values read per call).
    ds_west = ds_west.chunk({"time": 1, "y": -1, "x": -1})

    print(f"Writing WEST crop ({ds_west.sizes}) to {DST_PATH} ...")
    ds_west.to_zarr(DST_PATH, mode="w")
    print("Done.")

    result = subprocess.run(["du", "-sh", DST_PATH], capture_output=True, text=True)
    print(f"{DST_PATH}: {result.stdout.strip()}")
    result = subprocess.run(["du", "-sh", SRC_PATH], capture_output=True, text=True)
    print(f"{SRC_PATH} (original, for comparison): {result.stdout.strip()}")


if __name__ == "__main__":
    main()
