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
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, groups: int = 8, dropout: float = 0.0): 
        super().__init__()
        self.norm1 = nn.GroupNorm(_valid_groups(in_channels,  groups), in_channels)
        self.norm2 = nn.GroupNorm(_valid_groups(out_channels,  groups), out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.activation = nn.SiLU()
        self.dropout = dropout

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor): 
        """
        Following Ho et al. (2020): 
        h = SiLU(GN(x)) -> Conv2D -> Add time embedding -> SiLU(GN(h)) -> Dropout -> Conv2D -> Add internal skip connection
        """
        h = self.activation(self.norm1(x))
        h = self.conv1(h)

        t_emb_proj = self.time_emb_proj(t_emb)[:, :, None, None]  # reshape to [B, C, 1, 1]
        h = h + t_emb_proj

        h = self.activation(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)

        return h + self.skip(x)  # skip connection

class SimpleUnet(BaseUnet): 
    """
    A simple U-Net architecture
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, channels: list = [32, 64, 96, 128], groups: int = 8, dropout: float = 0.0): 
        super().__init__()
        self.channels = channels
        self.time_mlp = nn.Sequential(
            TimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4)
        )
        self.activation = nn.SiLU()

        self.base_channel = channels[0]

        self.in_conv = nn.Conv2d(in_channels, self.base_channel, 3, padding=1)

        #downsampling path
        self.down_blocks = nn.ModuleList()
        prev_channels = self.base_channel 
        skip_chs = []  # after in_conv
        for ch in self.channels:
            self.down_blocks.append(ResidualBlock(prev_channels, ch, time_emb_dim * 4, groups, dropout))
            skip_chs.append(ch)
            self.down_blocks.append(Downsampling(ch, ch))
            prev_channels = ch

        #bottleneck
        self.bottleneck = ResidualBlock(prev_channels, prev_channels, time_emb_dim * 4, groups, dropout)

        #upsampling path
        self.up_blocks = nn.ModuleList()
        for ch in reversed(self.channels):
            self.up_blocks.append(Upsampling(prev_channels, ch))
            self.up_blocks.append(ResidualBlock(ch + skip_chs.pop(), ch, time_emb_dim * 4, groups, dropout))
            prev_channels = ch

        assert not skip_chs, f"skip connection bookkeeping mismatch: {skip_chs}"

        #final norm + conv
        self.final_norm = nn.GroupNorm(_valid_groups(prev_channels, groups), prev_channels)
        self.final_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.in_conv(x)

        t_emb = self.time_mlp(t.float())

        #downsampling
        skip_connections = []
        for block in self.down_blocks:
            if isinstance(block, ResidualBlock):
                x = block(x, t_emb)
                skip_connections.append(x)
            else:  # Downsampling
                x = block(x)

        #bottleneck
        x = self.bottleneck(x, t_emb)

        #upsampling
        for block in self.up_blocks:
            if isinstance(block, Upsampling):
                x = block(x)
                skip_x = skip_connections.pop()
                x = torch.cat([x, skip_x], dim=1)  # concatenate along channel dimension
            else:  # ResidualBlock
                x = block(x, t_emb)

        return self.final_conv(F.silu(self.final_norm(x)))

def train(model: nn.Module, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str = "cpu", epochs: int = 1):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:

            inputs, targets, timesteps = batch
            inputs, targets, timesteps = inputs.to(device), targets.to(device), timesteps.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, timesteps)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")