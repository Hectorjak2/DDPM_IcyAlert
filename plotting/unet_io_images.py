"""
Renders the input/output images drawn in the SimpleUnet architecture figure (plotting/simple_unet_arch.py).

Real CARRA2-WEST fields (1216x1216, subsampled 3x for the figure), baseline U-Net (forecast setting):
    inputs : day k, day k-1, land mask  -> 3 channels
    output : predicted day k+1          -> 1 channel
Writes PNGs to plotting/assets/. Run from the repo root:  python3 plotting/unet_io_images.py
"""
import os

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
from matplotlib import colormaps, image  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "plotting", "assets")
DATA = os.environ.get("CARRA2_DATA_ROOT", os.path.join(ROOT, "data", "CARRA2_DAILY"))
DATES = {"km1": "2000-02-05", "k": "2000-03-05", "kp1": "2000-04-05"}  # one month apart (only year 2000 is stored locally)
STEP = 3

ICE = colormaps["Blues_r"].copy()   # 0 = open water (dark) -> 1 = ice (white)
ICE.set_bad("#C9C5BC")              # land
GREY = colormaps["gray"]


def main():
    os.makedirs(OUT, exist_ok=True)
    da = xr.open_zarr(os.path.join(DATA, "west.zarr"), decode_coords="all")["siconc"]
    # flipud: the y axis of the grid points north-up from row 0, images are drawn top-down.
    # fliplr: the images are drawn on the z axis of PlotNeuralNet's 3D view, which runs right-to-left.
    f = {k: np.fliplr(np.flipud(da.sel(time=d).values.astype(np.float32)[::STEP, ::STEP])) for k, d in DATES.items()}
    land = np.isnan(f["k"])

    image.imsave(os.path.join(OUT, "k.png"), np.ma.masked_invalid(f["k"]), cmap=ICE, vmin=0, vmax=1)
    image.imsave(os.path.join(OUT, "km1.png"), np.ma.masked_invalid(f["km1"]), cmap=ICE, vmin=0, vmax=1)
    image.imsave(os.path.join(OUT, "kp1.png"), np.ma.masked_invalid(f["kp1"]), cmap=ICE, vmin=0, vmax=1)
    image.imsave(os.path.join(OUT, "mask.png"), (~land).astype(np.float32), cmap=GREY, vmin=-0.3, vmax=1)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
