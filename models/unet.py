import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def _valid_groups(channels: int, groups: int) -> int:
    """Find largest divisor of channels that is <= groups"""
    while groups > 1:
        if channels % groups == 0:
            return groups
        groups -= 1
    return 1

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TimestepEmbedding(nn.Module): 
    """
    Timestep embedding following Vaswani et al., 2017 (Attention is all you need)
    """
    def __init__(self, d: int): 
        super().__init__()
        self.d = d

    def forward(self, t: torch.Tensor): 
        device = t.device
        half = self.d // 2
        
        tmp = torch.arange(half, device=device)
        freqs = 1.0 / (10000 ** (tmp / half))
        args = t[:, None] * freqs[None, :]  # shape [B, half]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # shape [B, d]
        
        return embedding 

class MonthlyEmbedding(nn.Module): 
    """
    Monthly embedding following The Fourier basis
    """
    def __init__(self, Nf: int): 
        super().__init__()
        self.Nf = Nf

    def forward(self, month: torch.Tensor):
        device = month.device

        tmp = torch.arange(1, (self.Nf+1), device=device)
        const = (2 * torch.pi) / (12)
        args = month[:, None] * const * tmp[None, :]  # shape [B, Nf]

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # shape [B, 2*Nf]

        return embedding

class Downsampling(nn.Module): 
    """
    Downsampling method for the U-Net architecture. Uses a 3x3 convolution with stride 2
    """
    def __init__(self, in_channels: int, out_channels: int): 
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor): 
        return self.conv(x)


class Upsampling(nn.Module):
    """
    Upsampling method for the U-Net architecture. Uses nearest neighbor 
    interpolation followed by a 3x3 convolution, following Ho et al. (2020).
    """
    def __init__(self, in_channels: int, out_channels: int): 
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        #self.up2   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) #another option for upsampling
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor): 
        return self.conv(self.up1(x))

class AttentionBlock(nn.Module): 
    """
    Factorized attention block. Instead of running attention on sequence length on all pixels in the field
    (Which at the bottle neck is 152*152=23104 => 23104^2 ~ 500 million) we factorize and compute an approximate 
    attention row and column wise. So for each row, each pixel in the row is compared to every other in the row: 
    (152*152*152) *2 ~7 million.

    The q, k, v projections act on the full channel vector and are split into heads afterwards, so each
    head learns its own projection. Rows and columns get separate projections. The output projection is
    zero-initialised so the block starts as the identity and doesn't disturb the residual stream early on.
    """
    def __init__(self, ch: int, n_heads: int, groups: int = 8):
        super().__init__()
        assert ch%n_heads == 0, f"{ch} should be divisible by n_heads"

        self.n_heads = n_heads
        self._norm = nn.GroupNorm(_valid_groups(ch, groups), ch)
        self.qkv_row = nn.Linear(ch, ch*3)
        self.qkv_col = nn.Linear(ch, ch*3)
        self.proj_out = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _attend(self, seq: torch.Tensor, qkv: nn.Linear):
        """seq: [N, L, C] -> multi-head self-attention along L -> [N, L, C]"""
        n, l, c = seq.shape
        q, k, v = qkv(seq).reshape(n, l, 3, self.n_heads, c // self.n_heads).permute(2, 0, 3, 1, 4)  # each  (N, n_heads, L, c//n_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.permute(0, 2, 1, 3).reshape(n, l, c)

    def forward(self, x:torch.Tensor):
        b, c, h, w = x.shape
        x_old = x
        x = self._norm(x).permute(0, 2, 3, 1)  # [B, H, W, C], channels last so a reshape groups whole pixels

        #Row wise: each sequence is one row of w pixels
        out1 = self._attend(x.reshape(b*h, w, c), self.qkv_row).reshape(b, h, w, c)

        #Column wise: each sequence is one column of h pixels
        out2 = self._attend(x.transpose(1, 2).reshape(b*w, h, c), self.qkv_col).reshape(b, w, h, c).transpose(1, 2)

        out = self.proj_out(out1 + out2).permute(0, 3, 1, 2)  # back to [B, C, H, W]
        return x_old + out


class ResidualBlock(nn.Module): 
    """
    Residual block for the U-Net architecture. 
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, month_emb_dim: int, groups: int = 8, dropout: float = 0.0): 
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_groups(in_channels,  groups), in_channels)
        self.norm2 = nn.GroupNorm(_valid_groups(out_channels,  groups), out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.month_emb_proj = nn.Linear(month_emb_dim, out_channels)
        self.activation = nn.SiLU()
        self.dropout = dropout

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, m_emb: torch.Tensor): 
        """
        Following Ho et al. (2020): 
        h = SiLU(GN(x)) -> Conv2D -> Add time embedding -> SiLU(GN(h)) -> Dropout -> Conv2D -> Add internal skip connection
        """
        h = self.activation(self.norm1(x))
        h = self.conv1(h)

        # SiLU before projecting: the embedding MLPs end in a Linear, so without it the two Linears collapse into one
        t_emb_proj = self.time_emb_proj(self.activation(t_emb))[:, :, None, None]  # reshape to [B, C, 1, 1]
        m_emb_proj = self.month_emb_proj(self.activation(m_emb))[:, :, None, None]  # reshape to [B, C, 1, 1]

        h = h + t_emb_proj + m_emb_proj

        h = self.activation(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)

        return h + self.skip(x)  # skip connection


class BaseUnet(nn.Module):
    """Base class for U-Net models with weight saving/loading functionality."""

    def save_weights(self, path: str) -> None:
        """Save model weights to a file.

        Args:
            path: Path where the weights will be saved.
        """
        torch.save(self.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_weights(self, path: str, device: str = "cpu") -> None:
        """Load model weights from a file.

        Args:
            path: Path to the weights file.
            device: Device to load weights onto ('cpu' or 'cuda').
        """
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        print(f"Model weights loaded from {path}")

    def save_checkpoint(self, path: str, optimizer=None, epoch: int = None, loss: float = None) -> None:
        """Save model checkpoint with optional optimizer state and metadata.

        Args:
            path: Path where the checkpoint will be saved.
            optimizer: Optional optimizer to save state.
            epoch: Optional epoch number.
            loss: Optional loss value.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            checkpoint["epoch"] = epoch
        if loss is not None:
            checkpoint["loss"] = loss
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, optimizer=None, device: str = "cpu") -> dict:
        """Load model checkpoint and optionally restore optimizer state.

        Args:
            path: Path to the checkpoint file.
            optimizer: Optional optimizer to restore state.
            device: Device to load weights onto ('cpu' or 'cuda').

        Returns:
            Dictionary with checkpoint metadata (epoch, loss, etc.)
        """
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}")
        return {k: v for k, v in checkpoint.items() if k not in ["model_state_dict", "optimizer_state_dict"]}

class SimpleUnet(BaseUnet): 
    """
    A simple U-Net architecture
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, Nf: int = 4, channels: list = [32, 64, 96, 128], groups: int = 8, dropout: float = 0.0, land_mask: torch.Tensor = None, device: str = "mps", n_res_blocks: int = 1, attention: bool = True): 
        super().__init__()
        self.device = device
        self.channels = channels
        self.attention = attention
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )

        self.month_mlp = nn.Sequential(
            MonthlyEmbedding(Nf),
            nn.Linear(2*Nf, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )

        self.base_channel = channels[0]

        self.in_conv = nn.Conv2d(in_channels, self.base_channel, 3, padding=1)
        emb = time_emb_dim * 4  # after the MLP, the embedding dimension is expanded to 4x

        #downsampling path
        self.down_blocks = nn.ModuleList()
        prev_channels = self.base_channel 
        skip_chs = [prev_channels]  # after in_conv
        for level, ch in enumerate(self.channels):
            for _ in range(n_res_blocks):
                self.down_blocks.append(ResidualBlock(prev_channels, ch, emb, emb, groups, dropout))
                prev_channels = ch
                skip_chs.append(ch)

            if level != len(self.channels) - 1:  # no downsample after the last level
                self.down_blocks.append(Downsampling(ch, ch))
                skip_chs.append(ch)

        #bottleneck
        if self.attention: 
            self.bottleneck = nn.ModuleList([
                ResidualBlock(prev_channels, prev_channels, emb, emb, groups, dropout),
                AttentionBlock(prev_channels, n_heads=4, groups=groups),
                ResidualBlock(prev_channels, prev_channels, emb, emb, groups, dropout)])
        else: 
            self.bottleneck = nn.ModuleList([
                ResidualBlock(prev_channels, prev_channels, emb, emb, groups, dropout),
                ResidualBlock(prev_channels, prev_channels, emb, emb, groups, dropout)])

        #upsampling path
        self.up_blocks = nn.ModuleList()
        for level, ch in reversed(list(enumerate(self.channels))):
            for _ in range(n_res_blocks + 1):  # one ResBlock per skip saved at this level
                self.up_blocks.append(ResidualBlock(prev_channels + skip_chs.pop(), ch, emb, emb, groups, dropout))
                prev_channels = ch
            if level != 0:  # no upsample after the full-resolution level
                self.up_blocks.append(Upsampling(ch, ch))

        assert not skip_chs, f"skip connection bookkeeping mismatch: {skip_chs}"

        #final norm + conv
        self.final_norm = nn.GroupNorm(_valid_groups(prev_channels, groups), prev_channels)
        self.final_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1)
        # Zero-initialised so the output starts at 0: the residual UNet baseline then starts
        # exactly at persistence, and the DDPM at predicting zero noise (as in Dhariwal & Nichol).
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

        #### For UNET baseline 
        if land_mask is not None:
            land_mask = land_mask.to(self.device).float()
            while land_mask.dim() < 4:
                land_mask = land_mask.unsqueeze(0)  # -> [1, 1, H, W]
            # Match the [-1, 1] scale of the image channel: +1 water, -1 land.
            self.mask_channel = land_mask * 2.0 - 1.0
            self.water_mask = land_mask.bool()
            self.image_size = max(land_mask.shape[-2:])
        else:
            self.mask_channel = None
            self.water_mask = None

    def forward(self, x: torch.Tensor, t: torch.Tensor, month: torch.Tensor):
        t_emb = self.time_mlp(t.float())
        m_emb = self.month_mlp(month.float())
        
        x = self.in_conv(x)
        skip_connections = [x]

        for block in self.down_blocks:
            x = block(x, t_emb, m_emb) if isinstance(block, ResidualBlock) else block(x)
            skip_connections.append(x)

        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                x = block(x, t_emb, m_emb)
            else: 
                x = block(x)

        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                x = block(torch.cat([x, skip_connections.pop()], dim=1), t_emb, m_emb)
            else:
                x = block(x)

        assert not skip_connections, f"skip connection bookkeeping mismatch: {skip_connections}"
        
        return self.final_conv(F.silu(self.final_norm(x)))

def train_network(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu", epochs: int = 1, experiment_name: str = "default", output_dir: str = "results"):
    model.to(device)
    model.train()
    os.makedirs(f"{output_dir}/{experiment_name}", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        total_p = 0.0
        for step, batch in enumerate(dataloader):
            batch, month = batch
            batch = batch.to(device)
            month = month.to(device)
            batch = torch.nan_to_num(batch, nan=0.0)  # Replace NaNs with zeros
            
            target = batch[:, 0:1]  # target is the first channel (k+1); keep [B, 1, H, W] so it doesn't broadcast against pred

            assert model.mask_channel is not None, "Land mask channel is not set in the model."
            x = torch.concat([batch[:,1:], model.mask_channel.expand(batch.shape[0], -1, -1, -1)], dim=1)  #keeping only the context (k, k-1) and removing the target (k+1) from the input to the model. Concatenating the land mask channel

            optimizer.zero_grad()
            t = torch.Tensor([0]).to(device)

            # The network predicts the residual k+1 - k, as in ConditionalDDPM; reconstruct
            # k+1 by adding day k back. The loss on the reconstruction equals the loss on the
            # residual, so it stays directly comparable to persistence below.
            pred = batch[:, 1:2] + model(x, t, month)

            assert pred.shape == target.shape
            se = (pred - target) ** 2
            loss = se[model.water_mask.expand_as(se)].mean()

            with torch.no_grad():
                se_p = (batch[:, 1:2] - target) ** 2          # persistence: x(k+1) = x(k)
                loss_p = se_p[model.water_mask.expand_as(se_p)].mean()

            total_loss += loss.item()
            total_p += loss_p.item()                          # init total_p = 0.0 next to total_loss
            print(f"Epoch {epoch} step {step} -- loss {loss.item():.4f}")
            if step % 50 == 0:
                print(f"Epoch {epoch} step {step} -- loss {total_loss/(step+1):.4f}  "
                    f"persistence {total_p/(step+1):.4f}  ratio {total_loss/max(total_p,1e-8):.3f}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    model.save_weights(f"{output_dir}/{experiment_name}/final_weights.pth")
    return model
