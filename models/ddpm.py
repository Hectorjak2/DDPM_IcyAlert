import torch
import torch.nn.functional as F

class DDPM:
    def __init__(self, timesteps: int = 1000, device: str = "mps"):
        self.timesteps = timesteps
        self.device = device
    
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

