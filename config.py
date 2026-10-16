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
    """UNet architecture configuration for CARRA2 training.
    """
    out_channels: int = 1
    dropout: float = 0.1
    groups: int = 8
    time_emb_dim: int = 64
    channels: list = field(default_factory=lambda: [64, 128, 128, 128])
    n_res_blocks: int = 1


@dataclass
class DDPMConfig:
    """DDPM diffusion process configuration.

    ``schedule`` selects how alpha_bar decays:

    - ``"linear"``: the original Ho et al. betas, kept so the pre-2026-09 runs stay
      reproducible for the write-up.
    - ``"shifted_cosine"`` (default): cosine schedule with its log-SNR shifted down by
      ``2 * log(shift_ref_resolution / image_size)``.

    The shift is the important part. Noise is i.i.d. per pixel, so averaging x_t over a
    k x k block cuts the noise std by k while a spatially smooth x0 keeps its magnitude:
    the SNR at scale k is ``sqrt(alpha_bar / (1 - alpha_bar)) * k``. Under the linear
    schedule at 1216x1216 that is 7.73 at k = the full domain, meaning x_T still reports
    the domain-mean ice concentration and q(x_T) is nowhere near N(0, I). The network then
    reads the global layout off its input instead of learning to generate one, and samples
    from a true N(0, I) collapse to the prior mean. The leak grows with image size (0.41 at
    64x64, 7.73 at 1216x1216), which is why an absolute beta range does not transfer.

    ``shift_ref_resolution=64`` is the "simple diffusion" (Hoogeboom et al.) prescription;
    see docs/architecture.md.
    """
    timesteps: int = 1000
    schedule: str = "shifted_cosine"
    shift_ref_resolution: int = 64
    # Only used when schedule == "linear".
    beta_start: float = 0.0001
    beta_end: float = 0.02
    device: str = "mps"  # Will be overridden by get_device() at runtime

TRAINING_SLICE = slice("2000-01-01", "2019-10-31")
TEST_SLICE = slice("2020-01-01", "2025-12-31")
SMOKE_TEST_SLICE = slice("2000-01-01", "2000-12-31")
MELT_SEASON = [slice(f"{year}-04-01", f"{year}-09-30") for year in range(2020, 2025)]
COOL_SEASON = [slice(f"{year}-10-01", f"{year + 1}-03-31") for year in range(2020, 2024)]

@dataclass
class TrainConfig:
    """Training parameters and hyperparameters.

    Note: The current __main__ block in main.py uses batch_size=2, epochs=2,
    which looks like a smoke-test config. See docs/architecture.md for details.
    For a real HPC run, you probably want larger values (e.g., batch_size=16, epochs=100).
    """
    #DDPM or Unet
    training_type: str = "unet" #"unet", "ddpm"

    #Forecasting settings: 
    forecasting_slice: slice = TEST_SLICE #Set to None if forecasting should be disabled
    
    #regular settings
    dataset = DatasetChoice.CARRA_WEST
    experiment_name: str = f"{training_type}-no_att_baseline"
    batch_size: int = 4
    epochs: int = 10
    lr: float = 3e-4
    number_of_samples: int = 10
    sampling: bool = True

