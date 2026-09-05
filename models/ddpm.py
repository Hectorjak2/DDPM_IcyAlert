import os
from random import sample

import torch
from torch.optim import Adam
import torch.nn.functional as F

class DDPM:
    def __init__(self, timesteps: int = 1000, device: str = "mps", image_size: int = 128):
        self.timesteps = timesteps
        self.device = device
        self.image_size = image_size

    def beta_schedule(self, timesteps, start=0.0001, end=0.02):
        """
        Linear schedule for beta values.

        Returns a tensor of shape (timesteps,) with linearly spaced values from start to end.
        """
        return torch.linspace(start, end, timesteps).to(self.device)

    def get_alphas(self, betas: torch.Tensor): 
        """
        Compute alpha and alpha_bar from beta values.

        Returns two tensors: alphas and alphas_bar both of shape (timesteps,).
        """
        alphas = (1.0 - betas).to(self.device)
        alphas_bar = torch.cumprod(alphas, dim=0).to(self.device)

        return alphas, alphas_bar

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x0: Original image tensor of shape [B, C, H, W].
            t: Timesteps tensor of shape [B].
            noise: Optional pre-sampled noise of shape [B, C, H, W].
        """

        if noise is None:
            noise = torch.randn_like(x0)

        # Get the beta schedule and compute alphas and alphas_bar
        betas = self.beta_schedule(self.timesteps)
        alphas, alphas_bar = self.get_alphas(betas)

        # Get the corresponding alpha_bar for each timestep t, reshaped to [B, 1, 1, 1]
        alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)

        # Compute x_t using the forward diffusion formula
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt, noise

    def compute_loss(self, model, x0: torch.Tensor, t: torch.Tensor): 
        """
        Compute the loss for the DDPM model. 

        Args:
            model: The U-Net model.
            x0: Original image tensor of shape [B, C, H, W].
            t: Timesteps tensor of shape [B].
        """
        # Mask of valid (water) pixels; land is NaN in x0
        mask = torch.isfinite(x0)

        # Replace NaNs so they don't propagate through the conv layers and
        # contaminate the predictions at valid pixels
        x0 = torch.nan_to_num(x0, nan=0.0)

        # Sample x_t using the forward diffusion process, keeping the noise target
        noise = torch.randn_like(x0)
        xt, noise = self.q_sample(x0, t, noise)

        # Predict the noise using the model
        predicted_noise = model(xt, t)

        # Mean squared error over masked (water) pixels only
        se = (predicted_noise - noise) ** 2
        loss = se[mask].mean()

        return loss
    
    @torch.no_grad()
    def p_sample(self, model, x, t, ): 
        """
        Reverse diffusion process: p(x_{t-1} | x_t)

        Args:
            model: The U-Net model.
            x: Current image tensor of shape [B, C, H, W].
            t: Current timestep tensor of shape [B].
        """
        betas = self.beta_schedule(self.timesteps)
        alphas, alphas_bar = self.get_alphas(betas)
        z = torch.randn_like(x) if t[0] > 1 else torch.zeros_like(x)

        #Reshape 
        betas_t = betas[t].view(-1, 1, 1, 1)
        alphas_t = alphas[t].view(-1, 1, 1, 1)
        alphas_bar_t = alphas_bar[t].view(-1, 1, 1, 1)

        # Getting xt_{-1}
        xt = 1/torch.sqrt(alphas_t) * (x - (1- alphas_t)/(torch.sqrt(1-alphas_bar_t)) * model(x, t)) + torch.sqrt(betas_t) * z

        return xt
    
    @torch.no_grad()
    def sample(self, model): 
        xt = torch.randn((1, 1, self.image_size, self.image_size), device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], device=self.device).long()
            xt = self.p_sample(model, xt, t_tensor)

        return xt

    def train(self, model, dataloader, device: str, timesteps: int, epochs: int, lr: float = 1e-3): 
        n_params = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n_params / 1e6:.1f}M")
        os.makedirs("checkpoints", exist_ok=True)

        model.to(device)
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            if epoch % 10 == 0 and epoch > 0: 
                model.save_checkpoint(f"checkpoints/model_epoch_{epoch}.pth", 
                                    optimizer=optimizer, 
                                    epoch=epoch)
                            
            for step, batch in enumerate(dataloader):
                batch = batch.to(device)

                optimizer.zero_grad()

                # sample t from U(0,T)
                t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
                
                loss = self.compute_loss(model, batch, t)

                print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.3f}")

                loss.backward()
                optimizer.step() 

        model.save_weights(f"checkpoints/model_final_t{timesteps}_epochs{epochs}_.pth")


if __name__ == "__main__":
    """model = Unet()
    optimizer = torch.optim.Adam(model.parameters())

    # Save during training
    if epoch % 10 == 0:
        model.save_checkpoint("checkpoints/model_epoch_{epoch}.pth", 
                            optimizer=optimizer, 
                            epoch=epoch, 
                            loss=current_loss)

    # Resume training
    metadata = model.load_checkpoint("checkpoints/model_epoch_50.pth", 
                                    optimizer=optimizer, 
                                    device="cuda")
    start_epoch = metadata["epoch"] + 1"""
