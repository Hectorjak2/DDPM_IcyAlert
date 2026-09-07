"""
Simple 4-level UNet for sea ice super-resolution.

Design notes
------------
- Conditioning is handled purely by channel concatenation at the input.
  Build the channel stack externally (e.g. [x_t, lr_sit, sic, mask, deform])
  and pass the total count as `in_channels`.
- Time / noise-level embedding is injected as an additive channel bias inside
  each ResBlock (easy to upgrade to full FiLM scale+shift later).
- No spatial attention (matches Brajard et al. 2026).
- GroupNorm throughout — stable for the small batch sizes typical in patch-based
  diffusion training.
- All building blocks are independent classes; swap or extend without touching
  the top-level SuperResUNet.

Spatial flow (default channels [32, 64, 96, 128], 3 downsampling steps)
------------------------------------------------------------------------
  init_conv          in_ch → C0            H × W
  DownBlock 0        C0    → C1,  skip@C1  H × W   → H/2 × W/2
  DownBlock 1        C1    → C2,  skip@C2  H/2      → H/4
  DownBlock 2        C2    → C3,  skip@C3  H/4      → H/8
  MidBlock           C3    → C3            H/8
  UpBlock 0          C3+C3 → C2            H/4
  UpBlock 1          C2+C2 → C1            H/2
  UpBlock 2          C1+C1 → C0            H
  final_conv         C0    → out_ch        H × W
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_groups(channels: int, desired: int = 8) -> int:
    """Largest divisor of `channels` that is <= `desired` (for GroupNorm)."""
    g = min(channels, desired)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


# ---------------------------------------------------------------------------
# Time / noise-level embedding
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for a scalar time / sigma value.

    Accepts t of shape (B,) — either discrete timestep indices or continuous
    noise-level values (e.g. log-sigma).  Output shape: (B, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / max(half - 1, 1)
        )                                   # (half,)
        args = t[:, None].float() * freqs[None, :]   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)   # (B, dim)


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Two-layer residual block with GroupNorm and time-embedding injection.

    Time is projected to `out_channels` scalars and added as a channel bias
    after the first convolution — a lightweight alternative to full FiLM that
    can be upgraded later by replacing the additive term with scale + shift.

    Parameters
    ----------
    in_channels, out_channels : spatial feature widths.
    time_dim : dimension of the time embedding vector fed in at forward time.
    groups   : target number of groups for GroupNorm (reduced automatically
               if `out_channels` is not divisible).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        groups: int = 8,
    ):
        super().__init__()
        self.norm1   = nn.GroupNorm(_valid_groups(in_channels,  groups), in_channels)
        self.conv1   = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2   = nn.GroupNorm(_valid_groups(out_channels, groups), out_channels)
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act     = nn.SiLU()
        self.skip    = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Additive time bias: (B, out_ch) -> (B, out_ch, 1, 1)
        h = h + self.time_proj(self.act(t_emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Encoder / decoder blocks
# ---------------------------------------------------------------------------

class DownBlock(nn.Module):
    """
    Two ResBlocks followed by stride-2 conv downsampling.

    Returns (downsampled_features, skip_connection).
    The skip is taken *before* downsampling so the decoder can recover the
    pre-downsampled spatial detail.
    """

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.res1      = ResBlock(in_channels,  out_channels, time_dim)
        self.res2      = ResBlock(out_channels, out_channels, time_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = self.res1(x, t_emb)
        x    = self.res2(x, t_emb)
        skip = x                          # (B, out_ch, H, W)
        x    = self.downsample(x)         # (B, out_ch, H/2, W/2)
        return x, skip


class MidBlock(nn.Module):
    """Bottleneck: two ResBlocks at the lowest spatial resolution."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_dim)
        self.res2 = ResBlock(channels, channels, time_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class UpBlock(nn.Module):
    """
    Bilinear 2× upsample, skip-connection concatenation, then two ResBlocks.

    Parameters
    ----------
    in_channels   : channels coming from the block below (before upsample).
    skip_channels : channels of the matching encoder skip connection.
    out_channels  : desired output channels.
    """

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int, time_dim: int
    ):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.res1 = ResBlock(in_channels + skip_channels, out_channels, time_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_dim)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SuperResUNet(nn.Module):
    """
    4-level UNet for conditional sea ice super-resolution.

    Conditioning is handled externally by stacking channels before calling
    forward.  A typical input stack might be::

        x_in = torch.cat([x_t, lr_sic, land_mask, ice_mask], dim=1)
        pred = model(x_in, t)

    Parameters
    ----------
    in_channels   : total channels in the pre-stacked conditioning tensor.
    out_channels  : output channels (usually 1 for a scalar residual field).
    base_dim      : channel width at the coarsest encoder level.
    channel_mults : multipliers applied to `base_dim` at each level.
                    Default (1, 2, 3, 4) → [32, 64, 96, 128] with base_dim=32.
    time_dim      : internal dimension of the time-embedding MLP.
    groups        : target GroupNorm group count (reduced per layer if needed).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_dim: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 3, 4),
        time_dim: int = 256,
        groups: int = 8,
    ):
        super().__init__()
        dims = [base_dim * m for m in channel_mults]   # e.g. [32, 64, 96, 128]
        n_levels = len(dims) - 1                        # number of down/up pairs

        # Time embedding: sinusoidal positional encoding → 2-layer MLP
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial channel expansion (no spatial change)
        self.init_conv = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList([
            DownBlock(dims[i], dims[i + 1], time_dim)
            for i in range(n_levels)
        ])

        # Bottleneck
        self.mid = MidBlock(dims[-1], time_dim)

        # Decoder (reversed: deepest first)
        self.ups = nn.ModuleList([
            UpBlock(
                in_channels=dims[n_levels - i],
                skip_channels=dims[n_levels - i],
                out_channels=dims[n_levels - i - 1],
                time_dim=time_dim,
            )
            for i in range(n_levels)
        ])

        # Final projection to output residual
        self.final_norm = nn.GroupNorm(_valid_groups(dims[0], groups), dims[0])
        self.final_conv = nn.Conv2d(dims[0], out_channels, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)  — pre-stacked noisy input + conditioning.
        t : (B,)                    — time steps or noise levels (float).

        Returns
        -------
        (B, out_channels, H, W)  — predicted residual.
        """
        t_emb = self.time_embed(t)      # (B, time_dim)

        x = self.init_conv(x)           # (B, C0, H, W)

        skips: list[torch.Tensor] = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid(x, t_emb)

        for up in self.ups:
            x = up(x, skips.pop(), t_emb)

        x = self.final_conv(F.silu(self.final_norm(x)))
        return x
