import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import BaseUnet

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

class SimpleUnet(BaseUnet): 
    """
    A simple U-Net architecture
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, Nf: int = 4, channels: list = [32, 64, 96, 128], groups: int = 8, dropout: float = 0.0, land_mask: torch.Tensor = None, device: str = "mps", n_res_blocks: int = 1): 
        super().__init__()
        self.device = device
        self.channels = channels
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
            x = block(x, t_emb, m_emb)

        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                x = block(torch.cat([x, skip_connections.pop()], dim=1), t_emb, m_emb)
            else:
                x = block(x)

        assert not skip_connections, f"skip connection bookkeeping mismatch: {skip_connections}"
        
        return self.final_conv(F.silu(self.final_norm(x)))

def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu", epochs: int = 1, experiment_name: str = "default"):
    model.to(device)
    model.train()
    os.makedirs(f"results/{experiment_name}", exist_ok=True)

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

            pred = model(x, t, month)
            
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

    model.save_weights(f"results/{experiment_name}/final_weights.pth")
