"""Resample the schedule-comparison runs without the per-step x_0 clamp (NC = no clamp).

Same weights as ddpm-corrected-land-<run>; only DDPMConfig.clamp_x0 changes (sampling only, none of
these runs used the unfolded loss, so no retraining). The final reconstruction is still clamped to
[0, 1] and land is still zeroed at every step (ConditionalDDPM.constrain_x0).

Everything else matches the original runs' config.txt (lead_months = 1, sample_seed = 0, 100 dates,
final weights), so the new samples.pkl pairs with the old one date for date. Each date is also seeded
with seed 0 as before, so on the same device the initial noise is the same too.
"""
from config import UnetConfig, DDPMConfig, TrainConfig
from main import run_ddpm

# (new name suffix, schedule, cosine_shift_res; the shift is ignored by cosine and linear)
RUNS = [
    ("cosine", "cosine", 64),
    ("linear", "linear", 64),
    ("256", "shifted_cosine", 256),
    ("32", "shifted_cosine", 32),
    ("64", "shifted_cosine", 64),
    ("128", "shifted_cosine", 128),
]

# Fail now, not hours in, if any run's weights are missing (paths relative to the submit dir).
from pathlib import Path
missing = [p for p in (Path(TrainConfig().output_dir) / f"ddpm-corrected-land-{s}" / "final_weights.pth" for s, _, _ in RUNS)
           if not p.exists()]
assert not missing, f"missing weights: {[str(p) for p in missing]}"

for suffix, schedule, shift_res in RUNS:
    # Fresh configs each run, so nothing set for one run leaks into the next.
    unet_conf = UnetConfig()
    train_conf = TrainConfig()
    ddpm_conf = DDPMConfig()

    ddpm_conf.schedule = schedule
    ddpm_conf.cosine_shift_res = shift_res
    ddpm_conf.cosine_shift_res_t0 = None
    ddpm_conf.clamp_x0 = False          # the point of this run: clamp only the final reconstruction
    ddpm_conf.sampler = "ddpm"

    train_conf.experiment_name = f"ddpm-NC-{suffix}"
    train_conf.training = False
    train_conf.sampling = True
    train_conf.analysis_sampling = False
    train_conf.unfolded_loss = False
    train_conf.dose_loss = None
    train_conf.input_perturbation = None
    train_conf.validation_slice = None  # not training, so skip loading the validation block
    train_conf.number_of_samples = 100
    train_conf.sample_seed = 0
    train_conf.lead_months = 1          # as the original runs; config.py currently says 6
    # Relative to train_conf.output_dir (run_ddpm prepends it). Final weights: the original
    # samples were drawn right after training, i.e. from the final weights.
    train_conf.weights = f"ddpm-corrected-land-{suffix}/final_weights.pth"

    print(f"\n===== {train_conf.experiment_name}: {schedule} (shift {shift_res}), "
          f"weights {train_conf.weights}, clamp_x0={ddpm_conf.clamp_x0} =====")

    run_ddpm(
        dataset=train_conf.dataset,
        forecast_period=train_conf.forecasting_slice,
        timesteps=ddpm_conf.timesteps,
        batch_size=train_conf.batch_size,
        epochs=train_conf.epochs,
        lr=train_conf.lr,
        experiment_name=train_conf.experiment_name,
        sampling=train_conf.sampling,
        train=train_conf.training,
        load_weights=train_conf.weights,
        ddpm_config=ddpm_conf,
        train_config=train_conf,
        unet_config=unet_conf,
    )
