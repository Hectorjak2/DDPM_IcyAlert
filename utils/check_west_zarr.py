"""One-off diagnostic: scan data/CARRA2_DAILY/west.zarr for bad `siconc` values.

Why this exists: training loss went NaN. `compute_loss()` (models/ddpm.py) assumes
two things about the data that this script checks directly, per timestep:

1. The land/sea mask (`isfinite`) is identical across all timesteps -- CARRA2.__init__
   in utils/dataloader.py derives `finite_mask` once from time=0 and reuses it for
   every sample. If some other timestep has NaNs where time=0 doesn't (or vice
   versa), that frame's real land/water geometry silently disagrees with the mask
   used everywhere else.
2. Every frame has at least one finite (water) pixel -- if a whole frame is NaN,
   `se[mask].mean()` averages over an empty selection and returns NaN, which is
   exactly the kind of thing that surfaces many batches later as "loss is NaN"
   with no obvious cause.

It also checks the actual finite values: siconc is a fraction, so anything outside
[0, 1], or non-finite in a way that isn't a clean NaN (e.g. +-inf), is reported.

Run on the HPC (the full WEST store is real, sizeable data -- this streams over
time in chunks rather than loading everything into RAM at once):
    python utils/check_west_zarr.py

Not wired into main.py -- this is a data-diagnostic step, not part of training.
"""

import numpy as np
import xarray as xr

PATH = "data/CARRA2_DAILY/west.zarr"
VARIABLE = "siconc"
VALID_RANGE = (0.0, 1.0)
# Small tolerance for floating point noise right at the boundary.
TOL = 1e-6
# How many timesteps to pull into memory per read.
TIME_BLOCK = 32


def main():
    ds = xr.open_zarr(PATH, decode_coords="all")
    da = ds[VARIABLE]
    n_time = da.sizes["time"]
    print(f"{PATH}: {VARIABLE} has shape {dict(da.sizes)}")

    reference_mask = None  # isfinite mask from time=0, [H, W] bool
    all_nan_times = []
    mask_mismatch_times = []
    out_of_range_count = 0
    out_of_range_examples = []  # (time_idx, y, x, value), capped
    inf_count = 0
    inf_examples = []

    total_pixels = 0
    total_finite = 0

    for start in range(0, n_time, TIME_BLOCK):
        end = min(start + TIME_BLOCK, n_time)
        block = da.isel(time=slice(start, end)).values  # [T, H, W]

        finite = np.isfinite(block)
        is_inf = np.isinf(block)

        total_pixels += block.size
        total_finite += finite.sum()

        if reference_mask is None:
            reference_mask = finite[0].copy()

        for t_local in range(block.shape[0]):
            t_global = start + t_local
            frame_finite = finite[t_local]

            if not frame_finite.any():
                all_nan_times.append(t_global)

            if not np.array_equal(frame_finite, reference_mask):
                mask_mismatch_times.append(t_global)

            frame_inf = is_inf[t_local]
            n_inf = int(frame_inf.sum())
            if n_inf:
                inf_count += n_inf
                if len(inf_examples) < 10:
                    ys, xs = np.where(frame_inf)
                    inf_examples.append((t_global, int(ys[0]), int(xs[0]), float(block[t_local, ys[0], xs[0]])))

            frame = block[t_local]
            valid_float = frame_finite & ~frame_inf
            oob = valid_float & ((frame < VALID_RANGE[0] - TOL) | (frame > VALID_RANGE[1] + TOL))
            n_oob = int(oob.sum())
            if n_oob:
                out_of_range_count += n_oob
                if len(out_of_range_examples) < 10:
                    ys, xs = np.where(oob)
                    out_of_range_examples.append((t_global, int(ys[0]), int(xs[0]), float(frame[ys[0], xs[0]])))

        print(f"  checked time {end}/{n_time}", end="\r")

    print()
    print("=== Summary ===")
    print(f"Total pixels scanned: {total_pixels}")
    print(f"Finite pixels (water, per np.isfinite -- excludes NaN and +-inf): {total_finite}")
    print(f"Water fraction (finite/total): {total_finite / total_pixels:.4f}")
    print()

    print(f"Timesteps with an all-NaN frame (no water pixels at all): {len(all_nan_times)}")
    if all_nan_times:
        print(f"  indices (first 20): {all_nan_times[:20]}")

    print(f"Timesteps whose finite-mask disagrees with time=0's mask: {len(mask_mismatch_times)}")
    if mask_mismatch_times:
        print(f"  indices (first 20): {mask_mismatch_times[:20]}")
        print("  -> CARRA2.finite_mask (derived once from time=0) does not match "
              "reality at these times; compute_loss() will average over the wrong "
              "set of pixels for them.")

    print(f"+-inf values found: {inf_count}")
    for t, y, x, v in inf_examples:
        print(f"  time={t} y={y} x={x} value={v}")

    print(f"Finite values outside [{VALID_RANGE[0]}, {VALID_RANGE[1]}] (tol={TOL}): {out_of_range_count}")
    for t, y, x, v in out_of_range_examples:
        print(f"  time={t} y={y} x={x} value={v}")

    if not (all_nan_times or mask_mismatch_times or inf_count or out_of_range_count):
        print()
        print("No issues found: NaNs are consistently land, all frames have water "
              "pixels, and finite values stay within [0, 1].")


if __name__ == "__main__":
    main()
