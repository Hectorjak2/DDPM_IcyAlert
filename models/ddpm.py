import math
import os
import pickle
from random import sample

import torch
from torch.optim import Adam
import torch.nn.functional as F
from typing import Optional
from config import DDPMConfig, TrainConfig
from utils.dataloader import to_data_range
from utils.progress import progress


def build_schedule(
    schedule: str,
    timesteps: int,
    image_size: int,
    beta_start: float = DDPMConfig.beta_start,
    beta_end: float = DDPMConfig.beta_end,
    cosine_s: float = DDPMConfig.cosine_s,
    cosine_shift_res: int = DDPMConfig.cosine_shift_res,
    cosine_shift_res_t0: int | None = DDPMConfig.cosine_shift_res_t0,
    laplace_mu: float = DDPMConfig.laplace_mu,
    laplace_b: float = DDPMConfig.laplace_b,
):
    """
    Build a diffusion schedule, returning ``(betas, alphas, alphas_bar)``.

    Parameters default to DDPMConfig, so training uses the config; plotting overrides them
    to compare several variants of one schedule. Only the parameters of ``schedule`` are read.
    """
    if schedule not in DDPM.SCHEDULES:
        raise ValueError(
            f"unknown schedule {schedule!r}; expected one of {DDPM.SCHEDULES}"
        )

    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

    else:
        if schedule in ("cosine", "shifted_cosine", "interpolated_cosine"):
            s = cosine_s
            x = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
            f = torch.cos((x + s) / (1.0 + s) * math.pi / 2.0) ** 2
            alphas_bar = (f[1:] / f[0]).clamp(1e-12, 1.0 - 1e-12)

            if schedule == "shifted_cosine":
                shift = 2.0 * math.log(cosine_shift_res / image_size)
                log_snr = torch.log(alphas_bar / (1.0 - alphas_bar)) + shift
                alphas_bar = torch.sigmoid(log_snr)

            elif schedule == "interpolated_cosine":
                # Hoogeboom et al. (2023, simple diffusion): the log-SNR shift moves linearly in t
                # from 2 log(cosine_shift_res_t0 / image_size) at t=0 to 2 log(cosine_shift_res /
                # image_size) at t=T. The noisy end keeps the shifted schedule's large-scale
                # behaviour; the clean end reaches a near noise-free x_t, so the last x0 estimate
                # can commit to the bound (solid ice, open water) instead of averaging below it.
                res_t0 = image_size if cosine_shift_res_t0 is None else cosine_shift_res_t0
                shift_t0 = 2.0 * math.log(res_t0 / image_size)
                shift_T = 2.0 * math.log(cosine_shift_res / image_size)
                frac = x[1:]  # t / T for t = 1..T, the same grid alphas_bar is built on
                log_snr = torch.log(alphas_bar / (1.0 - alphas_bar)) + (1.0 - frac) * shift_t0 + frac * shift_T
                alphas_bar = torch.sigmoid(log_snr).clamp(1e-12, 1.0 - 1e-12)

        elif schedule == "laplace":
            mu, b = laplace_mu, laplace_b
            t = (torch.arange(timesteps, dtype=torch.float64) + 0.5) / timesteps

            log_snr = mu - b * torch.sign(0.5 - t) * torch.log(1.0 - 2.0 * (t - 0.5).abs())
            alphas_bar = torch.sigmoid(log_snr).clamp(1e-12, 1.0 - 1e-12)

        # Shared derivation for all non-linear schedules
        alphas_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_bar[:-1]])
        betas = (1.0 - alphas_bar / alphas_bar_prev).clamp(0.0, 0.999)
        alphas = 1.0 - betas

    return betas, alphas, alphas_bar


class DDPM:
    # Noise schedules accepted by build_schedule(); plotting iterates over this.
    SCHEDULES = ("linear", "shifted_cosine", "interpolated_cosine", "cosine", "laplace")

    def __init__(
        self,
        timesteps: int = 1000,
        device: str = "mps",
        image_size: int = 896,
        land_mask: Optional[torch.Tensor] = None,
        schedule: str = None,
        cosine_shift_res: int = None,
        cosine_shift_res_t0: int | None = DDPMConfig.cosine_shift_res_t0,
    ):
        """
        DDPM wrapper.

        Args:
            schedule: one of DDPM.SCHEDULES
            cosine_shift_res: reference resolution of the shifted cosine schedule (for
                interpolated_cosine: at t=T). Must be passed explicitly to take a config
                instance's value; build_schedule's default is the class-level
                DDPMConfig.cosine_shift_res, fixed at import time.
            cosine_shift_res_t0: reference resolution at t=0 of interpolated_cosine; None means
                image_size (no shift). Same caveat: pass the config instance's value explicitly.
        """
        self.timesteps = timesteps
        self.device = device
        self.image_size = image_size
        self.schedule = schedule if schedule is not None else DDPMConfig.schedule
        self.cosine_shift_res = cosine_shift_res if cosine_shift_res is not None else DDPMConfig.cosine_shift_res
        self.cosine_shift_res_t0 = cosine_shift_res_t0
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
        Build the diffusion schedule from DDPMConfig, returning ``(betas, alphas, alphas_bar)``.
        """
        return build_schedule(self.schedule, timesteps, self.image_size,
                              cosine_shift_res=self.cosine_shift_res,
                              cosine_shift_res_t0=self.cosine_shift_res_t0)


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

    # Number of timestep buckets for per-t loss reporting: how well the model does at high vs low t.
    N_BUCKETS = 5

    def _batch_loss(self, model, batch, device, timesteps: int):
        """Loss of one dataloader batch at t ~ U(0, T), as in training. Returns (loss, t)."""
        # Conditional datasets yield (context, month); others a bare tensor.
        month = None
        if isinstance(batch, (list, tuple)):
            batch, month = batch
            month = month.to(device)
        batch = batch.to(device)
        t = torch.randint(0, timesteps, (batch.shape[0],), device=device).long()
        loss = (self.compute_loss(model, batch, t) if month is None
                else self.compute_loss(model, batch, t, month))
        return loss, t

    @classmethod
    def _bucket(cls, t: torch.Tensor, timesteps: int) -> int:
        return min(int(t.float().mean().item() / timesteps * cls.N_BUCKETS), cls.N_BUCKETS - 1)

    @classmethod
    def _bucket_summary(cls, bucket_means: list, timesteps: int) -> str:
        width = timesteps // cls.N_BUCKETS
        return " | ".join(
            f"t[{i * width}-{(i + 1) * width - 1}]: " + ("n/a" if m is None else f"{m:.4f}")
            for i, m in enumerate(bucket_means)
        )

    @staticmethod
    def _bucket_means(bucket_sum: list, bucket_count: list) -> list:
        return [s / c if c else None for s, c in zip(bucket_sum, bucket_count)]

    def evaluate(self, model, dataloader, device, timesteps: int, seed: int = 0, desc: str = "val") -> tuple[float, list]:
        """Training loss on `dataloader` without gradients: (mean loss, mean per timestep bucket).

        Dropout is off (eval mode, as when sampling), and so is DOSE (compute_loss checks
        model.training). t and the noise are drawn from `seed` in a forked RNG, so every call
        scores the same draws, which makes epochs comparable, and the training RNG stream is
        left untouched.
        """
        was_training = model.training
        model.eval()
        device_type = torch.device(device).type
        # fork_rng always forks the CPU generator; also fork the accelerator's if there is one.
        fork_kwargs = {"devices": []} if device_type == "cpu" else {"device_type": device_type}
        total, n = 0.0, 0
        bucket_sum = [0.0] * self.N_BUCKETS
        bucket_count = [0] * self.N_BUCKETS
        with torch.no_grad(), torch.random.fork_rng(**fork_kwargs):
            torch.manual_seed(seed)
            for batch in progress(dataloader, desc):
                loss, t = self._batch_loss(model, batch, device, timesteps)
                # Weighted by batch size, so a short last batch does not count as a full one.
                total += loss.item() * t.shape[0]
                n += t.shape[0]
                bucket = self._bucket(t, timesteps)
                bucket_sum[bucket] += loss.item()
                bucket_count[bucket] += 1
        model.train(was_training)
        return total / n, self._bucket_means(bucket_sum, bucket_count)

    def train(self, model, dataloader, experiment_name: str, device: str, timesteps: int, epochs: int, lr: float = 1e-3, train_config = TrainConfig(), val_dataloader = None):
        self.output_name = f"{model._get_name()}_{dataloader.dataset.name}"

        n_params = sum(p.numel() for p in model.parameters())
        print(f"parameters: {n_params / 1e6:.1f}M")
        out_dir = f"{train_config.output_dir}/{experiment_name}"
        os.makedirs(out_dir, exist_ok=True)

        model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr=lr)

        # Per-epoch losses, pickled after every epoch so a crashed run keeps its curve.
        # timesteps is kept so plotting can label the buckets (plotting/evaluation.py, loss_history).
        # "model" tells plotting/evaluation.py which loss this is.
        history = {"model": "ddpm", "train": [], "val": [], "train_buckets": [], "val_buckets": [], "timesteps": timesteps}
        best_val = float("inf")

        # Every 2 epochs: with epochs=10 (~10h total, ~1h/epoch) the old "% 10" cadence
        # never fired inside range(epochs), so a crash lost the whole run with no
        # fallback. This trades a bit of I/O for an actual safety net.
        checkpoint_every = 20
        for epoch in range(epochs):
            if epoch % checkpoint_every == 0 and epoch > 0:
                model.save_checkpoint(f"results/{experiment_name}/model_epoch_{epoch}.pth",
                                    optimizer=optimizer,
                                    epoch=epoch)

            bucket_sum = [0.0] * self.N_BUCKETS
            bucket_count = [0] * self.N_BUCKETS
            total, n = 0.0, 0

            pbar = progress(dataloader, f"Epoch {epoch} train")
            for batch in pbar:
                optimizer.zero_grad()

                loss, t = self._batch_loss(model, batch, device, timesteps)

                total += loss.item() * t.shape[0]
                n += t.shape[0]
                # Running epoch mean; refresh=False so it is shown at the bar's own refresh rate.
                pbar.set_postfix(loss=f"{total / n:.4f}", refresh=False)
                bucket = self._bucket(t, timesteps)
                bucket_sum[bucket] += loss.item()
                bucket_count[bucket] += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_loss = total / n
            train_buckets = self._bucket_means(bucket_sum, bucket_count)
            history["train"].append(train_loss)
            history["train_buckets"].append(train_buckets)
            print(f"Epoch {epoch} train loss by timestep bucket -- {self._bucket_summary(train_buckets, timesteps)}")

            if val_dataloader is not None:
                val_loss, val_buckets = self.evaluate(model, val_dataloader, device, timesteps,
                                                      seed=train_config.validation_seed, desc=f"Epoch {epoch} val")
                history["val"].append(val_loss)
                history["val_buckets"].append(val_buckets)
                print(f"Epoch {epoch} val loss by timestep bucket -- {self._bucket_summary(val_buckets, timesteps)}")
                print(f"Epoch {epoch} -- train: {train_loss:.4f} | val: {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    model.save_weights(f"{out_dir}/best_weights.pth")
                    print(f"Epoch {epoch} -- new best val loss, saved best_weights.pth")
            else:
                print(f"Epoch {epoch} -- train: {train_loss:.4f}")

            with open(f"{out_dir}/loss_history.pkl", "wb") as f:
                pickle.dump(history, f)

        model.save_weights(f"{out_dir}/final_weights.pth")

class ConditionalDDPM(DDPM):
    SAMPLERS = ("ddpm", "ddim")

    def __init__(self, *args, forecast_period: slice, batch_size: int, unfolded:bool, dose:int, residual_std: float, clamp_x0: bool = True, residual_context: bool = False,
                 sampler: str = "ddpm", ddim_steps: int = 50, ddim_eta: float = 0.0,
                 input_perturbation: float | None = None, **kwargs):
        """
        residual_std: std of the residual k+1 - k in model units [-1, 1] over the training set
            (CARRA2Forecast.residual_scale, from utils/residual_stats.json). Both residuals, the
            diffused target and the context channel, are divided by it, so the diffusion runs on
            unit-scale data as the noise schedule assumes.
        residual_context: see DDPMConfig.residual_context.
        sampler, ddim_steps, ddim_eta: see DDPMConfig.sampler.
        input_perturbation: gamma of DDPM-IP, see TrainConfig.input_perturbation.
        """
        super().__init__(*args, **kwargs)
        if sampler not in self.SAMPLERS:
            raise ValueError(f"unknown sampler {sampler!r}; expected one of {self.SAMPLERS}")
        self.sampler = sampler
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.input_perturbation = input_perturbation
        assert residual_std is not None, "no residual std for this training set; run python -m utils.residual_stats"
        self.residual_std = residual_std
        self.clamp_x0 = clamp_x0  # see DDPMConfig.clamp_x0
        self.residual_context = residual_context
        # x_t + context (k, k-1, and optionally the residual) + land mask
        #xt, (k, k-1), optionally one less if residual context, land mask
        self.in_channels = 1 + 2 - int(residual_context) + int(self.mask_channel is not None)
        self.forecast_period = forecast_period
        self.batch_size = batch_size
        self.unfolded = unfolded
        self.dose = dose

    def build_context(self, context: torch.Tensor) -> torch.Tensor:
        """Conditioning channels from context [B, 2, H, W] = (k, k-1), NaN-free.

        Returns [k, k-1], or [(k - (k-1)) / residual_std, k, k-1] with residual_context.
        """
        if not self.residual_context:
            return context
        context_residual = ((context[:, 0] - context[:, 1]) / self.residual_std).unsqueeze(1)
        return context_residual

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
        s = self.residual_std
        x0_residual = (x0[:,0] - x0[:,1]) / s #k+1 - k residual, standardized

        selected_context = self.build_context(x0[:, 1:])  # see DDPMConfig.residual_context

        # Sample x_t using the forward diffusion process, keeping the noise target
        noise = torch.randn_like(x0_residual.unsqueeze(1))
        # Input perturbation (TrainConfig.input_perturbation): x_t from slightly perturbed noise, so the
        # network learns to recover x_0 from the off-process x_t the sampling chain produces. Training
        # only, like DOSE. The target is x_0, so the perturbed noise is never a target.
        if self.input_perturbation and model.training:
            noise = noise + self.input_perturbation * torch.randn_like(noise)
        xt = self.q_sample(x0_residual.unsqueeze(1), t, noise) #Only noising the residual

        #reconstructing the input to be [xt, k - (k-1)]
        # DOSE only while training: the validation loss should measure the real task.
        if self.dose and model.training and torch.rand(()).item() < self.dose:
            xtr = torch.randn_like(xt)
            input_X = torch.cat([xtr, selected_context], dim=1)
        else: 
            input_X = torch.cat([xt, selected_context], dim=1)

        k = x0[:, 1:2]
        predicted_x0_residual = model(self.model_input(input_X), t, month)

        #Loss weighitng 
        snr = self.alphas_bar[t] / (1 - self.alphas_bar[t])
        w = snr.clamp(min=1.0, max=5.0).view(-1, 1, 1, 1)   # truncated-SNR, capped as in min-SNR-γ


        # Mean squared error over masked (water) pixels only. water_mask is [1, 1, H, W];        
        if self.unfolded and (t > 0).all():
            #Simulating one step of the chain: 
            z = torch.randn_like(xt[:,0].unsqueeze(1))
            coef_x0 = self.posterior_mean_coef_x0[t].view(-1, 1, 1, 1)
            coef_xt = self.posterior_mean_coef_xt[t].view(-1, 1, 1, 1)
            variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)

            # Simulate the same step as p_sample, so clamp exactly when sampling does.
            x0_step = torch.clamp(predicted_x0_residual, (-1.0 - k) / s, (1.0 - k) / s) if self.clamp_x0 else predicted_x0_residual
            mean = coef_x0 * x0_step + coef_xt * xt
            xtm1 = mean + torch.sqrt(variance_t) * z
            
            input_Xm1 = torch.cat([xtm1, selected_context], dim=1)
            unfolded_x0_residual = model(self.model_input(input_Xm1), t-1, month)

            se = w * 0.5 * (predicted_x0_residual - x0_residual.unsqueeze(1)) ** 2 + w * 0.5 * (unfolded_x0_residual - x0_residual.unsqueeze(1)) ** 2
        else: 
            se = w * (predicted_x0_residual - x0_residual.unsqueeze(1)) ** 2

        #se = (predicted_x0_residual - noise) ** 2
        loss = se[self.water_mask.expand_as(se)].mean()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, month, base: torch.Tensor):
        """
        Reverse diffusion process: p(x_{t-1} | x_t)

        Parameterised through the predicted residual x_0 (standardized, i.e. divided by
        residual_std) so it can be clamped at every step such that base + residual_std * x_0
        stays in the valid [-1, 1] data range; without that,
        small errors compound over T reverse steps and the sample drifts far outside it.

        Args:
            model: The U-Net model.
            x: [B, C, H, W] = (x_t, *build_context(k, k-1)).
            t: Current timestep tensor of shape [B].
            base: Field the residual is added to (day k), [B, 1, H, W]. Required, since x
                holds only the context residual, not k itself.
        """
        # No noise is added on the final step (t == 0), which is deterministic.
        z = torch.randn_like(x[:,0].unsqueeze(1)) if t[0] > 0 else torch.zeros_like(x[:,0].unsqueeze(1))

        #Reshape
        coef_x0 = self.posterior_mean_coef_x0[t].view(-1, 1, 1, 1)
        coef_xt = self.posterior_mean_coef_xt[t].view(-1, 1, 1, 1)
        variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)

        predicted_x0_residual = self.predict_x0(model, x, t, month, base)

        # Posterior mean of q(x_{t-1} | x_t, x_0), plus noise
        mean = coef_x0 * predicted_x0_residual + coef_xt * x[:,0].unsqueeze(1)
        xt = mean + torch.sqrt(variance_t) * z

        return xt

    def predict_x0(self, model, x, t, month, base: torch.Tensor) -> torch.Tensor:
        """The network's standardized residual x_0, clamped (DDPMConfig.clamp_x0) so that
        base + residual_std * x_0 stays in [-1, 1]. x and base as in p_sample."""
        predicted_x0_residual = model(self.model_input(x), t, month)
        if self.clamp_x0:
            k, s = base, self.residual_std
            predicted_x0_residual = torch.clamp(predicted_x0_residual, (-1.0 - k) / s, (1.0 - k) / s)
        return predicted_x0_residual

    def ddim_timesteps(self) -> list[int]:
        """Descending DDIM steps: ddim_steps evenly spaced values from T-1 to 0, both included,
        so the chain starts at the same noise level as DDPM and ends on the t=0 prediction."""
        steps = torch.linspace(self.timesteps - 1, 0, min(self.ddim_steps, self.timesteps)).round().long()
        return list(dict.fromkeys(steps.tolist()))  # drop duplicates from rounding, keep order

    @torch.no_grad()
    def ddim_step(self, model, x, t, t_prev: int, month, base: torch.Tensor):
        """
        One DDIM step x_t -> x_{t_prev} (Song et al. 2021, eq. 12), t_prev < t, or t_prev = -1
        for the final step, where alpha_bar = 1 and the result is the x_0 prediction itself.
        x, base as in p_sample.
        """
        xt = x[:, 0:1]
        x0 = self.predict_x0(model, x, t, month, base)
        ab_t = self.alphas_bar[t].view(-1, 1, 1, 1)
        ab_prev = self.alphas_bar[t_prev] if t_prev >= 0 else torch.ones((), device=xt.device)

        # The noise implied by x_t and the (clamped) x_0, so the two stay consistent
        eps = (xt - ab_t.sqrt() * x0) / (1.0 - ab_t).sqrt()
        sigma = self.ddim_eta * ((1.0 - ab_prev) / (1.0 - ab_t) * (1.0 - ab_t / ab_prev)).sqrt()
        z = torch.randn_like(xt) if t_prev >= 0 and self.ddim_eta > 0 else torch.zeros_like(xt)
        return ab_prev.sqrt() * x0 + (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps + sigma * z

    @torch.no_grad()
    def sample(self, context:torch.Tensor, month: int, model, target: Optional[torch.Tensor] = None,
               base: Optional[torch.Tensor] = None):
        """
        Generate one sample, returned in data range [0, 1] with land set to NaN.

        base: [1, H, W] field the predicted residual is added to. Defaults to day k
        (context[0]); pass the true k with a wrong context to test how much the
        network relies on its conditioning, separately from the reconstruction.
        """
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
        base = context[:, 0:1] if base is None else torch.nan_to_num(base, nan=0.0).unsqueeze(0)  # -> [1, 1, H, W]
        selected_context = self.build_context(context)  # see DDPMConfig.residual_context

        xt = torch.randn((1, 1, height, width), device=self.device)
        month_tensor = torch.tensor([month], device=self.device).long()
        # DDPM visits every t; DDIM a strided subset, each step jumping to the next one in the list
        steps = list(reversed(range(self.timesteps))) if self.sampler == "ddpm" else self.ddim_timesteps()
        for i, t in enumerate(steps):
            t_tensor = torch.tensor([t], device=self.device).long()
            # Re-assemble [xt, k - (k-1)] each step: the context channel stays fixed,
            # but xt must be the latest denoised estimate, not the initial noise.
            input_x = torch.cat([xt, selected_context], dim=1)
            if self.sampler == "ddpm":
                xt = self.p_sample(model, input_x, t_tensor, month_tensor, base)
            else:
                t_prev = steps[i + 1] if i + 1 < len(steps) else -1
                xt = self.ddim_step(model, input_x, t_tensor, t_prev, month_tensor, base)

            tmp_rec = base + self.residual_std * xt #temporary reconstruction for loss
            if target is not None: 
                se = (to_data_range(tmp_rec.clamp(-1.0, 1.0)) - target.nan_to_num(nan=0.0).unsqueeze(1)) ** 2
                loss = se[self.water_mask.expand_as(se)].mean()
                loss_during_sampling.append(loss.item())


        # Back from the model's [-1, 1] range to concentrations in [0, 1]
        xt_rec = base + self.residual_std * xt  # un-standardize the residual, add day k back
        xt_rec = to_data_range(xt_rec.clamp(-1.0, 1.0))

        if self.water_mask is not None:
            xt_rec = xt_rec.masked_fill(~self.water_mask, float("nan"))

        return xt_rec if target is None else (xt_rec, loss_during_sampling)