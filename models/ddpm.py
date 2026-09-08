import math
import os
from random import sample

import torch
from torch.optim import Adam
import torch.nn.functional as F

from config import DDPMConfig
from utils.dataloader import to_data_range


class DDPM:
    def __init__(self, timesteps: int = 1000, device: str = "mps", image_size: int = 128,
                 land_mask: torch.Tensor = None, schedule: str = None,
                 shift_ref_resolution: int = None):
        """
        Args:
            timesteps: number of diffusion steps T.
            device: torch device string.
            image_size: spatial size used by ``sample()`` when no land mask is given, and
                the resolution the noise schedule is shifted for. Must be set before the
                schedule is built.
            land_mask: optional static water/land mask of shape [1, H, W] or [1, 1, H, W]
                with 1.0 = water, 0.0 = land. When given it is fed to the model as a
                second input channel (so the network can tell land from open water,
                which are otherwise both encoded as a valid concentration) and used to
                mask land back to NaN in the generated samples.
            schedule: ``"shifted_cosine"`` or ``"linear"``; defaults to ``DDPMConfig``.
            shift_ref_resolution: reference resolution for the log-SNR shift; defaults to
                ``DDPMConfig``. Ignored by the linear schedule.
        """
        self.timesteps = timesteps
        self.device = device
        # Assigned before the schedule is built: the shift depends on it.
        self.image_size = image_size
        self.schedule = schedule if schedule is not None else DDPMConfig.schedule
        self.shift_ref_resolution = (
            shift_ref_resolution if shift_ref_resolution is not None
            else DDPMConfig.shift_ref_resolution
        )
        self.output_name = "" # Initialize output_name to an empty string

        # The land mask is resolved first: it is the authority on the field size, and the
        # noise schedule is shifted for that size.
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

        # Diffusion schedule. Precomputed once here rather than rebuilt on every
        # q_sample/p_sample call (which cost a linspace + cumprod per reverse step).
        #
        # Everything below is derived in float64 and cast to float32 only at the end.
        # The posterior coefficients all divide by (1 - alpha_bar), which at small t is a
        # subtraction of two nearly equal numbers: in float32 that cancellation is what
        # made posterior_mean_coef_x0[0] come out as 0.999834 instead of exactly 1.0.
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
        """Build the diffusion schedule, returning ``(betas, alphas, alphas_bar)``.

        The schedule is defined on ``alpha_bar`` rather than on ``betas``, because the
        quantity that actually matters -- how much of x0 survives to step t, at each
        spatial scale -- is a statement about alpha_bar. betas are then recovered from it.

        Returned in float64 on the CPU; ``__init__`` derives the posterior coefficients
        from these and casts everything to float32 at the end.
        """
        if self.schedule == "linear":
            betas = torch.linspace(
                DDPMConfig.beta_start, DDPMConfig.beta_end, timesteps, dtype=torch.float64
            )
        elif self.schedule == "shifted_cosine":
            alphas_bar = self._cosine_alphas_bar(timesteps)

            # Shift the whole log-SNR curve down by 2*log(ref/d). Averaging x_t over a
            # k x k block cuts the i.i.d. noise std by k while leaving a smooth x0
            # intact, so the SNR at scale k is sqrt(abar/(1-abar)) * k -- it grows with
            # image size. Shifting by the size ratio cancels exactly that growth, which
            # is what keeps q(x_T) close to N(0, I) at every scale up to the full domain.
            shift = 2.0 * math.log(self.shift_ref_resolution / self.image_size)
            log_snr = torch.log(alphas_bar / (1.0 - alphas_bar)) + shift
            alphas_bar = torch.sigmoid(log_snr)

            # betas from the ratio of consecutive alpha_bar, with alpha_bar_{-1} = 1.
            alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_bar[:-1]])
            betas = (1.0 - alphas_bar / alphas_bar_prev).clamp(0.0, 0.999)
        else:
            raise ValueError(
                f"unknown schedule {self.schedule!r}; expected 'shifted_cosine' or 'linear'"
            )

        # Recompute alpha_bar from the (possibly clamped) betas rather than reusing the
        # analytic one, so betas, alphas and alphas_bar are mutually consistent and the
        # posterior identities below hold exactly.
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        return betas, alphas, alphas_bar

    @staticmethod
    def _cosine_alphas_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine alpha_bar schedule (Nichol & Dhariwal 2021, eq. 17), in float64."""
        x = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        f = torch.cos((x + s) / (1.0 + s) * math.pi / 2.0) ** 2
        return (f[1:] / f[0]).clamp(1e-12, 1.0 - 1e-12)

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
        predicted_noise = model(self.model_input(xt), t)

        # Mean squared error over masked (water) pixels only
        se = (predicted_noise - noise) ** 2
        loss = se[mask].mean()

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

        # Bucket boundaries for per-timestep loss reporting. A single averaged loss
        # is not a useful diagnostic here: for most of t ~ U(0, T) the signal has
        # already decayed (sqrt(alpha_bar) < 0.3 beyond t/T ~ 0.5), so predicting the
        # noise is close to the identity map and the average is dominated by that
        # trivially easy regime. The low-t buckets are the ones that carry structure.
        n_buckets = 5
        for epoch in range(epochs):
            if epoch % 10 == 0 and epoch > 0:
                model.save_checkpoint(f"results/{experiment_name}/model_epoch_{epoch}.pth",
                                    optimizer=optimizer,
                                    epoch=epoch)

            bucket_sum = [0.0] * n_buckets
            bucket_count = [0] * n_buckets

            for step, batch in enumerate(dataloader):
                batch = batch.to(device)

                optimizer.zero_grad()

                # sample t from U(0,T)
                t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()

                loss = self.compute_loss(model, batch, t)

                print(f"Epoch: {epoch}, step: {step} -- Loss: {loss.item():.4f} ")#(t={t.tolist()})")

                bucket = min(int(t.float().mean().item() / timesteps * n_buckets), n_buckets - 1)
                bucket_sum[bucket] += loss.item()
                bucket_count[bucket] += 1

                loss.backward()
                optimizer.step()

            width = timesteps // n_buckets
            summary = " | ".join(
                f"t[{i * width}-{(i + 1) * width - 1}]: "
                + (f"{bucket_sum[i] / bucket_count[i]:.4f}" if bucket_count[i] else "n/a")
                for i in range(n_buckets)
            )
            print(f"Epoch {epoch} loss by timestep bucket -- {summary}")

        model.save_weights(f"results/{experiment_name}/final_{self.output_name}_batchsize{dataloader.batch_size}_t{timesteps}_epochs{epochs}_lr{lr}.pth")

