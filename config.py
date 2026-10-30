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

@dataclass
class DDPMConfig:
    """
    DDPM diffusion process configuration.
    """
    timesteps: int = 1000
    schedule: str = "cosine"

    # Only used when schedule == "linear".
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Only used when schedule == "laplace". Hang et al.'s defaults.
    laplace_mu: float = -2
    laplace_b: float = 0.5

    #Only used when schedule is cosine: 
    cosine_s: float = 0.008
    cosine_shift_res: int = 20

    analysis_sampling: bool = False
    residual_std: float = 1.0  # spread of the k+1 − k residual (model units) over TRAINING_SLICE; 1.0 = unscaled (pre-change checkpoints)


TRAINING_SLICE = slice("2000-01-01", "2019-10-31")
TEST_SLICE = slice("2020-01-01", "2025-12-31")
SMOKE_TEST_SLICE = slice("2000-01-01", "2000-12-31")
MELT_SEASON = [slice(f"{year}-04-01", f"{year}-09-30") for year in range(2020, 2025)]
COOL_SEASON = [slice(f"{year}-10-01", f"{year + 1}-03-31") for year in range(2020, 2024)]


@dataclass
class TrainConfig:
    """
    Training parameters and hyperparameters.
    """
    #DDPM or Unet
    training_type: str = "ddpm" #"unet", "ddpm"

    #Forecasting settings: 
    forecasting_slice: slice = TEST_SLICE #Set to None if forecasting should be disabled
    
    #regular settings
    dataset = DatasetChoice.CARRA_CROPPED
    experiment_name: str = f"{training_type}-new-objective"
    batch_size: int = 4
    epochs: int = 10
    lr: float = 3e-4
    number_of_samples: int = 100
    sample_seed: int = 0  # fixes which test dates get forecast, so runs are scored on the same days
    lead_months: int = 1  # autoregressive rollout k+1 ... k+lead_months; >1 drops anchors without ground truth that far ahead, so dates differ from lead 1
    sampling: bool = True

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
    weights: str = "results/ddpm-rescropped-shif70/final_weights.pth" #results/unet-residual/final_weights.pth" #Path to the .pth of the model to ensemble
    members_dir: str = "results/ensembles" #Only used when method == "deep": where train_ensemble wrote member_{seed}/
    n_members: int = 10
    experiment_name: str = f"{method}-ensemble-50sam"