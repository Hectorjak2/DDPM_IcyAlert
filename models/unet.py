"""DDPM U-Net (Ho et al. 2020).

A U-Net with Wide-ResNet residual blocks, GroupNorm, self-attention at a chosen
resolution, and sinusoidal timestep conditioning injected into every residual
block. The network predicts the noise epsilon and returns a tensor with the same
shape as the input.

Only ``torch`` is used.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Transformer sinusoidal position embedding of the diffusion timestep.

    Args:
        t: 1-D tensor of shape ``[B]`` holding (integer) timesteps.
        dim: embedding dimension.
        max_period: controls the lowest frequency.

    Returns:
        Tensor of shape ``[B, dim]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    args = t[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _norm(ch: int, groups: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, ch), ch)


class Downsample(nn.Module):
    """Strided 3x3 convolution that halves the spatial resolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsample followed by a 3x3 convolution."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ResBlock(nn.Module):
    """Wide-ResNet style residual block with timestep conditioning.

    GroupNorm -> SiLU -> Conv3x3, add projected time embedding,
    GroupNorm -> SiLU -> Dropout -> Conv3x3, plus a 1x1 skip when channels change.
    """

    def __init__(self, in_ch: int, out_ch: int, t_dim: int, dropout: float = 0.0, groups: int = 32):
        super().__init__()
        self.norm1 = _norm(in_ch, groups)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Linear(t_dim, out_ch)

        self.norm2 = _norm(out_ch, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Multi-head spatial self-attention with a residual connection."""

    def __init__(self, ch: int, groups: int = 32, num_heads: int = 4):
        super().__init__()
        if ch % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.norm = _norm(ch, groups)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w)
        q, k, v = qkv.unbind(1)                       # each [B, heads, d, HW]
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))  # [B, heads, HW, d]
        out = F.scaled_dot_product_attention(q, k, v)       # [B, heads, HW, d]
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class TimestepBlock(nn.Module):
    """Sequential container that forwards the time embedding only to ResBlocks."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, t_emb) if isinstance(layer, ResBlock) else layer(x)
        return x


class Unet(nn.Module):
    """DDPM U-Net.

    Args:
        in_channels: channels of the (noised) input image.
        out_channels: channels of the predicted noise (usually == in_channels).
        base_channels: channel count after the input convolution.
        channel_mult: per-resolution channel multiplier; its length is the number
            of resolution levels.
        num_res_blocks: residual blocks per resolution level.
        attention_resolutions: spatial resolutions (in pixels) at which to insert
            self-attention blocks.
        dropout: dropout probability inside residual blocks.
        groups: number of groups for GroupNorm.
        image_size: spatial size of the input; used to know which level matches
            ``attention_resolutions``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16,),
        dropout: float = 0.1,
        groups: int = 32,
        image_size: int = 128,
    ):
        super().__init__()
        self.base_channels = base_channels
        t_dim = base_channels * 4

        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ---- downsampling path ----
        self.down_blocks = nn.ModuleList()
        ch = base_channels
        now_res = image_size
        skip_chs = [ch]
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, t_dim, dropout, groups)]
                ch = out_ch
                if now_res in attention_resolutions:
                    layers.append(AttentionBlock(ch, groups))
                self.down_blocks.append(TimestepBlock(layers))
                skip_chs.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepBlock([Downsample(ch)]))
                skip_chs.append(ch)
                now_res //= 2

        # ---- bottleneck ----
        self.mid = TimestepBlock([
            ResBlock(ch, ch, t_dim, dropout, groups),
            AttentionBlock(ch, groups),
            ResBlock(ch, ch, t_dim, dropout, groups),
        ])

        # ---- upsampling path ----
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_chs.pop(), out_ch, t_dim, dropout, groups)]
                ch = out_ch
                if now_res in attention_resolutions:
                    layers.append(AttentionBlock(ch, groups))
                if level != 0 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    now_res *= 2
                self.up_blocks.append(TimestepBlock(layers))

        assert not skip_chs, f"skip connection bookkeeping mismatch: {skip_chs}"

        self.out = nn.Sequential(
            _norm(ch, groups),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``[B, in_channels, H, W]`` noised image.
            t: ``[B]`` timesteps.

        Returns:
            ``[B, out_channels, H, W]`` predicted noise.
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.base_channels))

        h = self.in_conv(x)
        skips = [h]
        for block in self.down_blocks:
            h = block(h, t_emb)
            skips.append(h)

        h = self.mid(h, t_emb)

        for block in self.up_blocks:
            h = block(torch.cat([h, skips.pop()], dim=1), t_emb)

        return self.out(h)


if __name__ == "__main__":
    model = Unet(image_size=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.1f}M")

    x = torch.randn(2, 1, 128, 128)
    t = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        out = model(x, t)
    print("input :", tuple(x.shape))
    print("output:", tuple(out.shape))
    assert out.shape == x.shape
