"""
Centralized configuration for DDPM_IcyAlert.

This file consolidates hyperparameters that were previously scattered across main.py,
models/ddpm.py, and models/unet.py. It serves as the single source of truth for what
configs are actually used during training.

When you want to change a hyperparameter, edit it here, then update main.py's __main__
block to use it.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class UnetConfig:
    """UNet architecture configuration for CARRA2 training.

    The defaults here are the downsized version optimized for 64GB HPC memory:
    ~10M params instead of the Unet class's own defaults (~78.7M).
    See docs/architecture.md for rationale.
    """
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 64          # Reduced from 128
    channel_mult: Tuple = (1, 2, 2, 2)  # Reduced from (1, 2, 2, 2, 4)
    num_res_blocks: int = 1          # Reduced from 2
    attention_levels: Tuple = ()     # No attention (was (3,))
    dropout: float = 0.1
    groups: int = 32


@dataclass
class DDPMConfig:
    """DDPM diffusion process configuration."""
    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    device: str = "mps"  # Will be overridden by get_device() at runtime


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Note: The current __main__ block in main.py uses batch_size=2, epochs=2,
    which looks like a smoke-test config. See docs/architecture.md for details.
    For a real HPC run, you probably want larger values (e.g., batch_size=16, epochs=100).
    """
    batch_size: int = 2
    epochs: int = 2
    lr: float = 1e-3


# Presets for common scenarios
SMOKE_TEST = TrainConfig(batch_size=256, epochs=2, lr=1e-3)
HPC_RUN = TrainConfig(batch_size=2, epochs=2, lr=1e-3)
