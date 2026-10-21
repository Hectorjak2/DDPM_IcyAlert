import math
import os
from random import sample

import torch
from torch.optim import Adam
import torch.nn.functional as F
from typing import Optional
from config import DDPMConfig
from utils.dataloader import to_data_range


class DDPM:
    # Noise schedules accepted by build_schedule(); plotting iterates over this.
    SCHEDULES = ("linear", "cosine", "laplace")

    def __init__(
        self,
        timesteps: int = 1000,
        device: str = "mps",
        image_size: int = 128,
        land_mask: Optional[torch.Tensor] = None,
        schedule: str = None,
    ):
        """
        DDPM wrapper.

        Args:
            schedule: one of DDPM.SCHEDULES
        """
        self.timesteps = timesteps
        self.device = device
        self.image_size = image_size
        self.schedule = schedule if schedule is not None else DDPMConfig.schedule
        self.output_name = "" # Initialize output_name to an empty string

        # The land mask is resolved first
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

        betas, alphas, alphas_bar = self.build_schedule(timesteps)

        # alpha_bar_{t-1}, with alpha_bar_{-1} = 1 so the t=0 posterior reduces to x0.
        alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_bar[:-1]])

        # Coefficients of the posterior q(x_{t-1} | x_t, x_0), see Ho et al. eq. (7).
        one_minus_alphas_bar = 1.0 - alphas_bar
        posterior_mean_coef_x0 = betas * torch.sqrt(alphas_bar_prev) / one_minus_alphas_bar
        posterior_mean_coef_xt = (
            (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / one_minus_alphas_bar
        )
        posterior_variance = betas * (1.0 - alphas_bar_prev) / one_minus_alphas_bar

        for name, tensor in (
            ("betas", betas),
            ("alphas", alphas),
            ("alphas_bar", alphas_bar),
            ("alphas_bar_prev", alphas_bar_prev),
            ("posterior_mean_coef_x0", posterior_mean_coef_x0),
            ("posterior_mean_coef_xt", posterior_mean_coef_xt),
            ("posterior_variance", posterior_variance),
        ):
            setattr(self, name, tensor.float().to(self.device))

    def build_schedule(self, timesteps: int):
        """
        Build the diffusion schedule, returning ``(betas, alphas, alphas_bar)``.
        """
        if self.schedule not in self.SCHEDULES:
            raise ValueError(
                f"unknown schedule {self.schedule!r}; expected one of {self.SCHEDULES}"
            )

        if self.schedule == "linear":
            betas = torch.linspace(
                DDPMConfig.beta_start, DDPMConfig.beta_end, timesteps, dtype=torch.float64)
            alphas = 1.0 - betas
            alphas_bar = torch.cumprod(alphas, dim=0)

        else: 
            if self.schedule == "cosine":
                s = DDPMConfig.cosine_s
                x = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
                f = torch.cos((x + s) / (1.0 + s) * math.pi / 2.0) ** 2
                alphas_bar = (f[1:] / f[0]).clamp(1e-12, 1.0 - 1e-12)

            elif self.schedule == "laplace":
                mu, b = DDPMConfig.laplace_mu, DDPMConfig.laplace_b
                t = (torch.arange(timesteps, dtype=torch.float64) + 0.5) / timesteps
                
                log_snr = mu - b * torch.sign(0.5 - t) * torch.log(1.0 - 2.0 * (t - 0.5).abs())
                alphas_bar = torch.sigmoid(log_snr).clamp(1e-12, 1.0 - 1e-12)

            # Shared derivation for all non-linear schedules
            alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_bar[:-1]])
            betas = (1.0 - alphas_bar / alphas_bar_prev).clamp(0.0, 0.999)
            alphas = 1.0 - betas
            
        return betas, alphas, alphas_bar


    def model_input(self, xt: torch.Tensor) -> torch.Tensor:
        """Assemble the network input, appending the static land mask channel if present."""
        if self.mask_channel is None:
            return xt
        return torch.cat([xt, self.mask_channel.expand(xt.shape[0], -1, -1, -1)], dim=1)

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

        # Get the corresponding alpha_bar for each timestep t, reshaped to [B, 1, 1, 1]
        alpha_bar_t = self.alphas_bar[t].view(-1, 1, 1, 1)

        # Compute x_t using the forward diffusion formula
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt

    def compute_loss(self, model, x0: torch.Tensor, t: torch.Tensor): 
        """
        Compute the loss for the DDPM model. 

        Args:
            model: The U-Net model.
            x0: Original image tensor of shape [B, C, H, W].
            t: Timesteps tensor of shape [B].
        """

        # Replace NaNs so they don't propagate through the conv layers and
        # contaminate the predictions at valid pixels
        x0 = torch.nan_to_num(x0, nan=0.0)

        # Sample x_t using the forward diffusion process, keeping the noise target
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        # Predict the noise using the model
        predicted_noise = model(self.model_input(xt), t)

        # Mean squared error over masked (water) pixels only. water_mask is [1, 1, H, W];
        # expand it to the batch size since boolean-mask indexing needs an exact shape
        # match, not just a broadcastable one.
        se = (predicted_noise - noise) ** 2
        loss = se[self.water_mask.expand_as(se)].mean()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, ):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)

        Parameterised through the predicted x_0 so it can be clamped to the valid
        [-1, 1] data range at every step; without that, small errors compound over
        T reverse steps and the sample drifts far outside the data range.

        Args:
            model: The U-Net model.
            x: Current image tensor of shape [B, C, H, W].
            t: Current timestep tensor of shape [B].
        """
        # No noise is added on the final step (t == 0), which is deterministic.
        z = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)

        #Reshape
        alphas_bar_t = self.alphas_bar[t].view(-1, 1, 1, 1)
        coef_x0 = self.posterior_mean_coef_x0[t].view(-1, 1, 1, 1)
        coef_xt = self.posterior_mean_coef_xt[t].view(-1, 1, 1, 1)
        variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)

        # Recover x_0 from the predicted noise and clamp it to the data range
        predicted_noise = model(self.model_input(x), t)
        x0_pred = (x - torch.sqrt(1 - alphas_bar_t) * predicted_noise) / torch.sqrt(alphas_bar_t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Posterior mean of q(x_{t-1} | x_t, x_0), plus noise
        mean = coef_x0 * x0_pred + coef_xt * x
        xt = mean + torch.sqrt(variance_t) * z

        return xt

    @torch.no_grad()
    def sample(self, model):
        """Generate one sample, returned in data range [0, 1] with land set to NaN."""
        if self.water_mask is not None:
            height, width = self.water_mask.shape[-2:]
        else:
            height = width = self.image_size

        xt = torch.randn((1, 1, height, width), device=self.device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], device=self.device).long()
            xt = self.p_sample(model, xt, t_tensor)

        # Back from the model's [-1, 1] range to concentrations in [0, 1]
        xt = to_data_range(xt.clamp(-1.0, 1.0))

        if self.water_mask is not None:
            xt = xt.masked_fill(~self.water_mask, float("nan"))

        return xt

    def train(self, model, dataloader, experiment_name: str, device: str, timesteps: int, epochs: int, lr: float = 1e-3): 
        self.output_name = f"{model._get_name()}_{dataloader.dataset.name}"

        n_params = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n_params / 1e6:.1f}M")
        os.makedirs(f"results/{experiment_name}", exist_ok=True)

        model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr=lr)

        # Bucket boundaries for per-timestep loss reporting. It would be nice to be able to 
        # see the loss in the t-intervals, so how well is it performing when t is high, low, etc.
        n_buckets = 5
        # Every 2 epochs: with epochs=10 (~10h total, ~1h/epoch) the old "% 10" cadence
        # never fired inside range(epochs), so a crash lost the whole run with no
        # fallback. This trades a bit of I/O for an actual safety net.
        checkpoint_every = 20
        for epoch in range(epochs):
            if epoch % checkpoint_every == 0 and epoch > 0:
                model.save_checkpoint(f"results/{experiment_name}/model_epoch_{epoch}.pth",
                                    optimizer=optimizer,
                                    epoch=epoch)

            bucket_sum = [0.0] * n_buckets
            bucket_count = [0] * n_buckets

            for step, batch in enumerate(dataloader):
                # Conditional datasets yield (context, month); others a bare tensor.
                month = None
                if isinstance(batch, (list, tuple)):
                    batch, month = batch
                    month = month.to(device)
                batch = batch.to(device)

                optimizer.zero_grad()

                # sample t from U(0,T)
                t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()

                loss = (self.compute_loss(model, batch, t) if month is None
                        else self.compute_loss(model, batch, t, month))

                print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.4f} ")#(t={t.tolist()})")

                bucket = min(int(t.float().mean().item() / timesteps * n_buckets), n_buckets - 1)
                bucket_sum[bucket] += loss.item()
                bucket_count[bucket] += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            width = timesteps // n_buckets
            summary = " | ".join(
                f"t[{i * width}-{(i + 1) * width - 1}]: "
                + (f"{bucket_sum[i] / bucket_count[i]:.4f}" if bucket_count[i] else "n/a")
                for i in range(n_buckets)
            )
            print(f"Epoch {epoch} loss by timestep bucket -- {summary}")

        model.save_weights(f"results/{experiment_name}/final_weights.pth")

class ConditionalDDPM(DDPM):
    def __init__(self, *args, forecast_period: slice, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecast_period = forecast_period

    def compute_loss(self, model, x0: torch.Tensor, t: torch.Tensor, month: torch.Tensor): 
        """
        Compute the loss for the DDPM model. 

        Args:
            model: The U-Net model.
            x0: Original image tensor of shape [B, C, H, W].
            t: Timesteps tensor of shape [B].
            month: Month (0-11) of the target k+1, shape [B].
        """

        # Replace NaNs so they don't propagate through the conv layers and
        # contaminate the predictions at valid pixels
        x0 = torch.nan_to_num(x0, nan=0.0) #k+1, k, k-1

        # Sample x_t using the forward diffusion process, keeping the noise target
        noise = torch.randn_like(x0[:,0].unsqueeze(1))
        xt = self.q_sample(x0[:,0].unsqueeze(1), t, noise) #Only taking k+1

        #reconstructing the input to be [xt, k, k-1]
        input_X = torch.cat([xt, x0[:,1:]], dim=1)

        # Predict the noise using the model
        predicted_noise = model(self.model_input(input_X), t, month)

        # Mean squared error over masked (water) pixels only. water_mask is [1, 1, H, W];
        # expand it to the batch size since boolean-mask indexing needs an exact shape
        # match, not just a broadcastable one.
        se = (predicted_noise - noise) ** 2
        loss = se[self.water_mask.expand_as(se)].mean()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, month):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)

        Parameterised through the predicted x_0 so it can be clamped to the valid
        [-1, 1] data range at every step; without that, small errors compound over
        T reverse steps and the sample drifts far outside the data range.

        Args:
            model: The U-Net model.
            x: Current image tensor of shape [B, C, H, W].
            t: Current timestep tensor of shape [B].
        """
        # No noise is added on the final step (t == 0), which is deterministic.
        z = torch.randn_like(x[:,0].unsqueeze(1)) if t[0] > 0 else torch.zeros_like(x[:,0].unsqueeze(1))

        #Reshape
        alphas_bar_t = self.alphas_bar[t].view(-1, 1, 1, 1)
        coef_x0 = self.posterior_mean_coef_x0[t].view(-1, 1, 1, 1)
        coef_xt = self.posterior_mean_coef_xt[t].view(-1, 1, 1, 1)
        variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)

        # Recover x_0 from the predicted noise and clamp it to the data range
        predicted_noise = model(self.model_input(x), t, month)
        x0_pred = (x[:,0].unsqueeze(1) - torch.sqrt(1 - alphas_bar_t) * predicted_noise) / torch.sqrt(alphas_bar_t)
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        # Posterior mean of q(x_{t-1} | x_t, x_0), plus noise
        mean = coef_x0 * x0_pred + coef_xt * x[:,0].unsqueeze(1)
        xt = mean + torch.sqrt(variance_t) * z

        return xt

    @torch.no_grad()
    def sample(self, context:torch.Tensor, month: int, model, target: Optional[torch.Tensor] = None):
        """Generate one sample, returned in data range [0, 1] with land set to NaN."""
        if target is not None: 
            loss_during_sampling = []
        
        if self.water_mask is not None:
            height, width = self.water_mask.shape[-2:]
        else:
            height = width = self.image_size

        # Land pixels are NaN in the raw context; zero them out before they hit the
        # conv layers, same as compute_loss does for x0 -- otherwise NaN spreads
        # through the receptive field and compounds over every reverse step.
        context = torch.nan_to_num(context, nan=0.0).unsqueeze(0)  # -> [1, C, H, W]
        xt = torch.randn((1, 1, height, width), device=self.device)
        month_tensor = torch.tensor([month], device=self.device).long()
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.tensor([t], device=self.device).long()
            # Re-assemble [xt, k, k-1] each step: the context channels stay fixed,
            # but xt must be the latest denoised estimate, not the initial noise.
            input_x = torch.cat([xt, context], dim=1)
            xt = self.p_sample(model, input_x, t_tensor, month_tensor)

            if target is not None: 
                se = (to_data_range(xt.clamp(-1.0, 1.0)) - target.nan_to_num(nan=0.0).unsqueeze(1)) ** 2
                loss = se[self.water_mask.expand_as(se)].mean()
                loss_during_sampling.append(loss.item())


        # Back from the model's [-1, 1] range to concentrations in [0, 1]
        xt = to_data_range(xt.clamp(-1.0, 1.0))

        if self.water_mask is not None:
            xt = xt.masked_fill(~self.water_mask, float("nan"))

        return xt if target is None else (xt, loss_during_sampling)