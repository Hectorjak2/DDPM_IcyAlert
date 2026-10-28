"""
Centralized configuration for DDPM_IcyAlert.

This file consolidates hyperparameters that were previously scattered across main.py,
models/ddpm.py, and models/unet.py. It serves as the single source of truth for what
configs are actually used during training.

When you want to change a hyperparameter, edit it here, then update main.py's __main__
block to use it.
"""

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
    schedule: str = "shifted_cosine"

    # Only used when schedule == "linear".
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Only used when schedule == "laplace". Hang et al.'s defaults.
    laplace_mu: float = -2
    laplace_b: float = 0.5

    #Only used when schedule is cosine: 
    cosine_s: float = 0.008
    cosine_shift_res: int = 128

    analysis_sampling: bool = False

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
    training_type: str = "unet" #"unet", "ddpm"

    #Forecasting settings: 
    forecasting_slice: slice = TEST_SLICE #Set to None if forecasting should be disabled
    
    #regular settings
    dataset = DatasetChoice.CARRA_CROPPED
    experiment_name: str = f"{training_type}-residual-cropped"
    batch_size: int = 4
    epochs: int = 10
    lr: float = 3e-4
    number_of_samples: int = 100
    sample_seed: int = 0  # fixes which test dates get forecast, so runs are scored on the same days
    sampling: bool = True

    #Saving samples setting 
    output_dir: str = f"results"

@dataclass
class EnsembleConfig:
    """
    Ensemble sampling (see ensemble.py). Dates, test slice and dataset come from TrainConfig.
    """
    method: str = "deep" #"dropout" (MC dropout through the UNet baseline), "ddpm" (independent initial noise), "deep" (UNets trained from different seeds)
    weights: str = "results/unet-residual/final_weights.pth" #results/unet-residual/final_weights.pth" #Path to the .pth of the model to ensemble
    members_dir: str = "results/ensembles" #Only used when method == "deep": where train_ensemble wrote member_{seed}/
    n_members: int = 10
    experiment_name: str = f"ensemble-{method}_tester"