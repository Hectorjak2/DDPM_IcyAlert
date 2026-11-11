"""
Centralized configuration for DDPM_IcyAlert.

This file consolidates hyperparameters that were previously scattered across main.py,
models/ddpm.py, and models/unet.py. It serves as the single source of truth for what
configs are actually used during training.

When you want to change a hyperparameter, edit it here, then update main.py's __main__
block to use it.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

from models.models import DatasetChoice

@dataclass
class UnetConfig:
    """
    UNet architecture configuration for CARRA2 training.
    """
    out_channels: int = 1
    dropout: float = 0.1
    groups: int = 8
    time_emb_dim: int = 64
    channels: list = field(default_factory=lambda: [64, 128, 128, 128])
    n_res_blocks: int = 1
    attention: bool = True
    in_channels: int = 4

@dataclass
class DDPMConfig:
    """
    DDPM diffusion process configuration.
    """
    timesteps: int = 1000
    schedule: str = "shifted_cosine"

    # Only used when schedule == "linear".
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Only used when schedule == "laplace". Hang et al.'s defaults.
    laplace_mu: float = 0
    laplace_b: float = 0.5

    #Only used when schedule is cosine:
    cosine_s: float = 0.008
    # shifted_cosine: reference resolution of the log-SNR shift 2 log(res / image_size).
    # interpolated_cosine: the reference resolution at t=T (the noisy end).
    cosine_shift_res: int = 64
    # Only used when schedule == "interpolated_cosine" (Hoogeboom et al. 2023): the reference
    # resolution at t=0; the shift moves linearly in t between the two. None means image_size,
    # i.e. no shift at t=0, so the chain ends as clean as plain cosine (sqrt(alpha_bar) ~ 1 at t=0)
    # instead of shifted_cosine 64's 0.996. With 64 at t=T, on 896: ~5% near-clean steps, ~61%
    # transition, ~34% pure noise (shifted_cosine 64: 0%, 39%, 61%).
    cosine_shift_res_t0: int | None = None

    # Clamp the predicted x_0 residual at every reverse step so that k + r stays in [-1, 1].
    # The bounds are correct, but where the truth sits on one (open water k=-1, full ice k=+1;
    # ~half the domain) the clamp cuts only one side of a noisy estimate, biasing it inward
    # (haze over open water, too little solid ice). Only affects sampling, plus the simulated
    # step of the unfolded loss (TrainConfig.unfolded_loss); with that off, no retraining needed.
    # The final reconstruction is clamped to [-1, 1] either way.
    clamp_x0: bool = False

    # Reverse process used by ConditionalDDPM.sample. Sampling only: the same weights work with
    # either, so no retraining. "ddpm": ancestral sampling over all `timesteps` steps (Ho et al.).
    # "ddim": Song et al. (2021) over `ddim_steps` evenly spaced steps of the same schedule; fewer
    # steps means ~timesteps/ddim_steps x faster and fewer steps for errors to compound over
    # (exposure bias, the drift toward melt seen in cool-season chains).
    sampler: str = "ddpm"
    ddim_steps: int = 50
    # Noise injected per DDIM step. 0: deterministic given the initial noise (ensemble spread
    # comes only from x_T). 1: the DDPM posterior variance at the coarser steps.
    ddim_eta: float = 0

    # Conditioning of ConditionalDDPM. False: context is [k, k-1], network input
    # [x_t, k, k-1, mask] (4 channels). True: the standardized residual (k - (k-1)) / residual_std
    # is prepended, input [x_t, k-(k-1), k, k-1, mask] (5 channels). The UNet's in_channels follows
    # from this (ConditionalDDPM.in_channels), so weights trained with one setting do not load with the other.
    residual_context: bool = False

    # The residual std is not set here: it comes per (area, training slice) from
    # utils/residual_stats.json via CARRA2Forecast.residual_scale, and is written to config.txt.


# Train / validation / test. Validation is a fixed block at the end of the old training slice
# (2000-2019), so the test years stay untouched. CARRA2Forecast only builds windows whose k-1 and k+1
# lie inside its own slice, so the two-month gap between slices means no day is shared.
# Changing a training slice changes its residual-std key: rerun `python -m utils.residual_stats`.
TRAINING_SLICE = slice("2000-01-01", "2016-10-31")
VALIDATION_SLICE = slice("2017-01-01", "2019-10-31")
SMOKE_TRAINING_SLICE = slice("2000-01-01", "2003-10-31")
SMOKE_VALIDATION_SLICE = slice("2004-01-01", "2004-10-31")
TEST_SLICE = slice("2020-01-01", "2025-12-31")
SMOKE_TEST_SLICE = slice("2005-01-01", "2006-12-31")
MELT_SEASON = [slice(f"{year}-04-01", f"{year}-09-30") for year in range(2020, 2025)]
COOL_SEASON = [slice(f"{year}-10-01", f"{year + 1}-03-31") for year in range(2020, 2024)]


@dataclass
class TrainConfig:
    """
    Training parameters and hyperparameters.
    """
    #DDPM or Unet
    training_type: str = "ddpm" #"unet", "ddpm"

    #DDPM specific
    analysis_sampling: bool = False
    unfolded_loss: bool = False
    dose_loss: int = None#0.1  #p: chance of replacing x_t with noise
    # Input perturbation (Ning et al. 2023, DDPM-IP): x_t is built from noise + gamma * xi, xi ~ N(0, I),
    # so the network also sees x_t slightly off the forward process, as it does in the sampling chain
    # (exposure bias: the drift toward melt in cool-season samples). The target is unchanged. Training
    # only, so the validation loss stays on clean inputs. None disables it; the paper uses 0.1.
    input_perturbation: float | None = 0.1

    #Forecasting and training settings: 
    training_slice: slice = TRAINING_SLICE #Set to None if forecasting should be disabled
    forecasting_slice: slice = TEST_SLICE #Set to None if forecasting should be disabled
    validation_slice: slice | None = VALIDATION_SLICE #Per-epoch validation loss; None disables it
    validation_seed: int = 0  # fixes t and noise of the validation pass, so epochs are scored on the same draws
    
    #regular settings
    dataset = DatasetChoice.CARRA_CROPPED
    experiment_name: str = f"ddpm-titanfull-shif64-ip_lead6"
    batch_size: int = 4
    epochs: int = 10
    lr: float = 3e-4
    number_of_samples: int = 100
    sample_seed: int = 0  # fixes which test dates get forecast, so runs are scored on the same days
    lead_months: int = 6  # autoregressive rollout k+1 ... k+lead_months; >1 drops anchors without ground truth that far ahead, so dates differ from lead 1
    sampling: bool = True
    training: bool = False 
    weights: str = "ddpm-titanfull-shif64-ip/best_weights.pth"#"ddpm-titanfull-shif64-ip/final_weights.pth"

    #Saving settings
    output_dir: str = "results"  # weights and config.txt: {output_dir}/{experiment_name}/
    # Samples (samples.pkl, ensemble_samples.pkl, loss_stats.pkl): {samples_dir}/{experiment_name}/.
    # Separate from output_dir because samples are large; run.sh points TEMP_RESULT_STORAGE at /work3 on HPC.
    samples_dir: str = os.environ.get("TEMP_RESULT_STORAGE", "results")

@dataclass
class EnsembleConfig:
    """
    Ensemble sampling (see ensemble.py). Dates, test slice and dataset come from TrainConfig.
    """
    method: str = "ddpm" #"dropout" (MC dropout through the UNet baseline), "ddpm" (independent initial noise), "deep" (UNets trained from different seeds)
    weights: str = "results/ddpm-titanfull-shif64-ip/final_weights.pth" #results/unet-residual/final_weights.pth" #Path to the .pth of the model to ensemble
    members_dir: str = "results/ensembles" #Only used when method == "deep": where train_ensemble wrote member_{seed}/
    n_members: int = 10
    experiment_name: str = f"{method}-titan-ensemble-ip64"