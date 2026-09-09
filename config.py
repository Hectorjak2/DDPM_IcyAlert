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
    # 2 input channels: the noised SIC field plus the static land/sea mask, so the
    # network can tell land from open water. Output is just the predicted noise.
    in_channels: int = 2
    out_channels: int = 1
    base_channels: int = 64          # Reduced from 128
    channel_mult: Tuple = (1, 2, 2, 2)  # Reduced from (1, 2, 2, 2, 4)
    num_res_blocks: int = 1          # Reduced from 2
    # attention_levels governs the down/up paths only; mid_attention governs the
    # bottleneck. Both are off, so the network has zero attention blocks. The
    # bottleneck sits at 152x152 = 23k tokens at the WEST resolution, where
    # attention is quadratic and was the single largest memory item.
    attention_levels: Tuple = ()     # No attention in down/up (was (3,))
    mid_attention: bool = False      # No attention in the bottleneck (was implicitly True)
    dropout: float = 0.1
    groups: int = 32


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
    schedule: str = "linear"
    shift_ref_resolution: int = 64
    # Only used when schedule == "linear".
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
    experiment_name: str = "linear_lr1e3_ep10_batch4"
    batch_size: int = 4
    epochs: int = 10
    lr: float = 1e-3
    number_of_samples: int = 8

