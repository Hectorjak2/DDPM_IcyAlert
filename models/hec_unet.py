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

        t_emb_proj = self.time_emb_proj(t_emb)[:, :, None, None]  # reshape to [B, C, 1, 1]
        m_emb_proj = self.month_emb_proj(m_emb)[:, :, None, None]  # reshape to [B, C, 1, 1]

        h = h + t_emb_proj + m_emb_proj

        h = self.activation(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)

        return h + self.skip(x)  # skip connection

class SimpleUnet(BaseUnet): 
    """
    A simple U-Net architecture
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, Nf: int = 4, channels: list = [32, 64, 96, 128], groups: int = 8, dropout: float = 0.0, land_mask: torch.Tensor = None, device: str = "mps"): 
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

        #downsampling path
        self.down_blocks = nn.ModuleList()
        prev_channels = self.base_channel 
        skip_chs = []  # after in_conv
        for ch in self.channels:
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock(prev_channels, ch, time_emb_dim * 4, time_emb_dim * 4, groups, dropout),
                #ResidualBlock(ch, ch, time_emb_dim * 4, time_emb_dim * 4, groups, dropout),
                Downsampling(ch, ch)]))
            skip_chs.append(ch)
            prev_channels = ch

        #bottleneck
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResidualBlock(prev_channels, prev_channels, time_emb_dim * 4, time_emb_dim * 4, groups, dropout))
        self.bottleneck.append(ResidualBlock(prev_channels, prev_channels, time_emb_dim * 4, time_emb_dim * 4, groups, dropout))

        #upsampling path
        self.up_blocks = nn.ModuleList()
        for ch in reversed(self.channels):
            self.up_blocks.append(nn.ModuleList([
                Upsampling(prev_channels, ch), 
                ResidualBlock(ch + skip_chs.pop(), ch, time_emb_dim * 4, time_emb_dim * 4, groups, dropout),
                #ResidualBlock(ch, ch, time_emb_dim * 4, time_emb_dim * 4, groups, dropout)                                   
                ]))
            prev_channels = ch

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
        x = self.in_conv(x)

        t_emb = self.time_mlp(t.float())
        m_emb = self.month_mlp(month.float())

        #downsampling
        skip_connections = []
        for res1, downsample in self.down_blocks:
            x = res1(x, t_emb, m_emb)
            #x = res2(x, t_emb, m_emb)
            skip_connections.append(x)
            x = downsample(x)

        #bottleneck
        for block in self.bottleneck:
            x = block(x, t_emb, m_emb)

        #upsampling
        for upsample, res1 in self.up_blocks:
            x = upsample(x)
            skip_x = skip_connections.pop()
            x = torch.cat([x, skip_x], dim=1)  # concatenate along channel dimension
            x = res1(x, t_emb, m_emb)
            #x = res2(x, t_emb, m_emb)

        assert not skip_connections, f"skip connection bookkeeping mismatch: {skip_connections}"
        
        return self.final_conv(F.silu(self.final_norm(x)))

def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: str = "cpu", epochs: int = 1, experiment_name: str = "default"):
    model.to(device)
    model.train()
    os.makedirs(f"results/{experiment_name}", exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch, month = batch
            batch = batch.to(device)
            month = month.to(device)
            batch = torch.nan_to_num(batch, nan=0.0)  # Replace NaNs with zeros
            
            target = batch[:, 0]  # target is the first channel (k+1)

            assert model.mask_channel is not None, "Land mask channel is not set in the model."
            x = torch.concat([batch[:,1:], model.mask_channel.expand(batch.shape[0], -1, -1, -1)], dim=1)  #keeping only the context (k, k-1) and removing the target (k+1) from the input to the model. Concatenating the land mask channel

            optimizer.zero_grad()
            t = torch.Tensor([0]).to(device)

            pred = model(x, t, month)

            se = (pred - target) ** 2
            loss = se[model.water_mask.expand_as(se)].mean()
            
            total_loss += loss.item()

            print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.4f} ")#(t={t.tolist()})")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    model.save_weights(f"results/{experiment_name}/final_weights.pth")
