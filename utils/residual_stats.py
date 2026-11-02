"""Mean and std of the one-month residual k+1 - k, which ConditionalDDPM diffuses.

Computing them reads the whole training slice, so it is done once here and cached in
utils/residual_stats.json, keyed by area and time slice. CARRA2Forecast loads its own
entry at init as `residual_mean` / `residual_std` (SIC units). Each entry also holds the
same numbers in the DDPM's [-1, 1] units (`*_ddpm`) and the UNet's standardized units
(`*_unet`). Commit the json so HPC uses the same numbers.

    python -m utils.residual_stats                      # smoke (CARRA_TEST) and real (CARRA_CROPPED) training sets
    python -m utils.residual_stats --area CARRA_WEST    # one area over TRAINING_SLICE
"""
import argparse
import json
from datetime import date

from config import SMOKE_TRAINING_SLICE, TRAINING_SLICE
from models.models import DatasetChoice
from utils.dataloader import RESIDUAL_STATS_PATH, CARRA2Forecast, residual_stats_key

# (area, time_slice) combinations computed when no --area is given
DEFAULT_RUNS = [
    (DatasetChoice.CARRA_TEST, SMOKE_TRAINING_SLICE),
    (DatasetChoice.CARRA_CROPPED, TRAINING_SLICE),
]


def compute(area: DatasetChoice, time_slice: slice, stride: int) -> dict:
    ds = CARRA2Forecast("siconc", "cpu", time_slice=time_slice, area=area, batch_dim=False)
    stats = ds.residual_stats(stride=stride)
    # The same residual in each model's units. Offsets cancel in a difference, so only the scale
    # factor carries over: DDPM [-1, 1] doubles it, the UNet's (x - field_mean) / field_std divides
    # it by field_std (the slice's own field std, i.e. what the UNet trains with on this slice).
    field_mean, field_std = float(ds.mean), float(ds.std)
    return {**stats,
            "std_ddpm": 2 * stats["std"], "mean_ddpm": 2 * stats["mean"],
            "std_unet": stats["std"] / field_std, "mean_unet": stats["mean"] / field_std,
            "field_mean": field_mean, "field_std": field_std,
            "stride": stride, "computed": date.today().isoformat()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", choices=[a.value for a in DatasetChoice if a != DatasetChoice.FASHION])
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    runs = [(DatasetChoice(args.area), TRAINING_SLICE)] if args.area else DEFAULT_RUNS
    cache = json.loads(RESIDUAL_STATS_PATH.read_text()) if RESIDUAL_STATS_PATH.exists() else {}
    for area, time_slice in runs:
        key = residual_stats_key(area, time_slice)
        print(f"Computing {key} ...", flush=True)
        cache[key] = compute(area, time_slice, args.stride)
        print(json.dumps(cache[key], indent=2), flush=True)
        # Write after every run so a later failure does not lose the earlier ones
        RESIDUAL_STATS_PATH.write_text(json.dumps(cache, indent=2) + "\n")
    print(f"Saved to {RESIDUAL_STATS_PATH}")
