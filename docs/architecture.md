# Architecture & Design Decisions

## Module Map

- **[main.py](../main.py)** — Entry point. `run(carra=False, fashion=False, ...)` builds the dataset/dataloader, instantiates a UNet and DDPM, and trains. `download_samples(ddpm, n_of_samples)` runs inference and pickles outputs to `results/{ddpm.output_name}/samples.pkl`.
- **[models/ddpm.py](../models/ddpm.py)** — `DDPM` class: orchestrates diffusion training and sampling. The schedule tensors (`betas`, `alphas`, `alphas_bar`, `alphas_bar_prev`, posterior coefficients) are precomputed once in `__init__`. Takes an optional `land_mask` (see Land Mask as Input Channel below). Methods:
  - `beta_schedule()` — linear schedule from `start` to `end` over `timesteps`.
  - `get_alphas()` — derives `alpha` and `alpha_bar` from betas.
  - `model_input(xt)` — assembles the network input, appending the static land-mask channel when one was given.
  - `q_sample(x0, t, noise)` — forward diffusion process, adds noise to an image at timestep `t`.
  - `compute_loss(model, x0, t)` — loss function with NaN-masking for land pixels (see Design Decisions below).
  - `p_sample(model, x, t)` — reverse diffusion process, one denoising step, parameterised through a clamped predicted `x0`.
  - `sample(model)` — generates a sample from pure noise by iterating `p_sample` backwards; returns data-range `[0, 1]` with land set to NaN.
  - `train(model, dataloader, device, timesteps, epochs, lr)` — training loop with periodic checkpointing and per-timestep-bucket loss reporting.
- **[models/unet.py](../models/unet.py)** — UNet architectures.
  - `Unet` (general-purpose, resolution-agnostic) — configurable via `base_channels`, `channel_mult`, `num_res_blocks`, `attention_levels`, etc. Defaults: `base_channels=128`, `channel_mult=(1,2,2,2,4)`, `num_res_blocks=2`, `attention_levels=(3,)` (~78.7M params).
  - `UnetSmall` (fixed for small images) — compact version for FashionMNIST (28×28).
- **[utils/dataloader.py](../utils/dataloader.py)** — Dataset classes and range conversion.
  - `to_model_range(x)` / `to_data_range(x)` — map `[0, 1]` ↔ `[-1, 1]` (see Data Normalization below).
  - `CARRA2(selected_variable, device, time_slice, WEST, TEST, batch_dim)` — loads CARRA2 reanalysis data from a zarr store. `siconc` is sea-ice concentration. Exposes `finite_mask` (`[1, H, W]`, 1.0 = water) derived once from the first time step.
  - `FashionMNIST(train, device, batch_dim)` — wraps torchvision's FashionMNIST, auto-downloads. `finite_mask` is `None`.
- **[utils/device_utils.py](../utils/device_utils.py)** — Device detection.
  - `get_device()` — returns a device string (`"cuda"`, `"mps"`, or `"cpu"`) based on availability.
  - `get_device_info()` — diagnostic info (VRAM, compute capability, etc.).
  - `device_diagnostics(device)` — prints device details.
- **[utils/download_data.py](../utils/download_data.py)** — CDS (Climate Data Store) API client for downloading CARRA2 monthly data.
- **[utils/visualization.py](../utils/visualization.py)** — Plotting helpers (`visualize_sic_sample`, `visualize_fashion`).
- **[data/](../data/)** (gitignored) — `CARRA2_MONTHLY/dataset.zarr` (zarr store, large) and `FashionMNIST/raw` (auto-downloaded by torchvision).
- **[results/](../results/)** (gitignored) — Checkpoints and sampled outputs, organized by `{model_name}_{dataset_name}/`.
- **[run.sh](../run.sh)** — DTU HPC batch script (LSF/BSUB) that loads cuda/12.4.1, sets PyTorch config, and invokes `python main.py`.

## Design Decisions

### NaN-Masking for Land Pixels

**Location:** [models/ddpm.py](../models/ddpm.py) `compute_loss()`, lines 68–84.

The CARRA2 sea-ice concentration field (`siconc`) is defined only over water; land pixels are NaN. During loss computation:

1. `mask = torch.isfinite(x0)` — extract a boolean mask of non-NaN (water) pixels.
2. `x0 = torch.nan_to_num(x0, nan=0.0)` — replace NaNs with 0 before passing through convolutions, so NaNs don't propagate and contaminate predictions at valid water pixels.
3. `se = (predicted_noise - noise) ** 2; loss = se[mask].mean()` — compute squared error over all pixels, but average only over the masked (water) pixels.

**Rationale:** The model must never learn to predict land. By zeroing the loss gradient over land, we ensure the model's parameters are updated only based on water-pixel errors, so the model never "learns" to output values at land locations.

**Note:** This masks land out of the *loss* only. On its own it gives the network no way to know *where* land is — see Land Mask as Input Channel below.

### Data Normalization to [-1, 1]

**Location:** [utils/dataloader.py](../utils/dataloader.py) `to_model_range()` / `to_data_range()`, applied in both `__getitem__` methods and in `DDPM.sample()`.

`siconc` is a fraction in `[0, 1]` (mean ≈ 0.36, std ≈ 0.45 over CARRA2-WEST); FashionMNIST is `[0, 1]` after `ToTensor()`. The DDPM forward process `x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε` assumes roughly zero-mean, unit-scale `x_0`, so both datasets are shifted to `[-1, 1]` with `2x - 1` on load, and `sample()` maps back with `(x+1)/2`.

**Why it matters:** Training on un-normalized `[0, 1]` data mismatches the noise schedule's assumed signal scale. An earlier run without this produced samples with mean 1.28 and range `[-11, 25]` — entirely outside the valid concentration range, and clipped to near-uniform white by the `vmin=0, vmax=1` plotting in [utils/visualization.py](../utils/visualization.py).

**Consequence:** Checkpoints trained before this change are not compatible — they learned a different input distribution.

### Land Mask as Input Channel

**Location:** [models/ddpm.py](../models/ddpm.py) `DDPM.__init__()` and `model_input()`; wired up in [main.py](../main.py) `run()`.

`compute_loss()` replaces land NaNs with `0.0` before the convolutions. In `[-1, 1]` space `0.0` is a *legal* concentration (SIC = 0.5), so from the network's point of view land is indistinguishable from ordinary water. Masking the loss stops it from being penalised on land, but never tells it where the coastline is — and at sampling time there is no `x0` to derive a mask from at all, so an unconditional sample has no way to place Greenland.

The static water/land mask is therefore passed to `DDPM` and concatenated as a second input channel (`+1` water, `-1` land, matching the image channel's scale). This makes `UnetConfig.in_channels = 2` for CARRA2; `out_channels` stays 1 (the predicted noise). `sample()` re-applies the mask, setting land back to NaN.

The mask is time-invariant in CARRA2 (verified across the full 288-month record), so `CARRA2.__init__` derives it once from the first time step rather than per sample. FashionMNIST passes `land_mask=None`, in which case `model_input()` is a pass-through and `UnetSmall` stays 1-channel.

### Clamped x0 Parameterization in `p_sample`

**Location:** [models/ddpm.py](../models/ddpm.py) `p_sample()`.

The reverse step is computed via the predicted `x_0` rather than directly from the noise:

1. `x0_pred = (x_t - sqrt(1-ᾱ_t)·ε_θ) / sqrt(ᾱ_t)`, then **clamped to `[-1, 1]`**.
2. `mean = coef_x0(t)·x0_pred + coef_xt(t)·x_t`, the posterior mean of `q(x_{t-1} | x_t, x_0)` (Ho et al. eq. 7), with variance `β̃_t = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)`.

Without the clamp this is algebraically identical to the previous direct form `1/sqrt(α_t)·(x_t - (1-α_t)/sqrt(1-ᾱ_t)·ε_θ)` (verified numerically to ~1e-8). The clamp is the point: over T=1000 reverse steps, unconstrained errors compound and drive the sample far outside the data range.

Also fixed here: noise was previously added whenever `t > 1`, so the final step was stochastic. It is now `t > 0`, making the last step deterministic as intended.

### Per-Timestep Loss Reporting

**Location:** [models/ddpm.py](../models/ddpm.py) `train()`.

**A single averaged loss is not a useful diagnostic for this model.** With linear betas 1e-4 → 0.02 over T=1000, `sqrt(ᾱ_t)` is 0.28 by t=500 and 0.006 by t=999, so for most of `t ~ U(0,T)` the input is already nearly pure noise and predicting ε collapses to close to the identity map. The CARRA2 field makes this worse: 55% of water pixels are exactly 0, and over a locally constant patch ε-prediction reduces to an affine map a conv net fits within a handful of gradient steps.

Measured on the 2-epoch run, per timestep: 0.45 at t=5, 0.042 at t=150, 0.0095 at t=500, 0.0045 at t=900. The scalar "loss ≈ 0.0075" that looked like convergence was just the high-t bucket; the low-t regime that actually generates structure was no better than predicting zero.

`train()` therefore accumulates loss into 5 timestep buckets and prints a per-bucket summary each epoch. Watch the low-t buckets; a healthy run flattens the profile.

### UNet Downsizing for CARRA2

**Location:** [main.py](../main.py) `run()`, lines 42–49.

The `Unet` class's own defaults are `base_channels=128, channel_mult=(1,2,2,2,4), num_res_blocks=2, attention_levels=(3,)`, which yields ~78.7M parameters and high memory usage. For CARRA2 training on DTU HPC (64GB system memory, 1×80GB GPU per [run.sh](../run.sh)), we use a smaller configuration:

- `in_channels=2` (image + land mask; see Land Mask as Input Channel)
- `base_channels=64` (halved)
- `channel_mult=(1,2,2,2)` (removed the 4x level)
- `num_res_blocks=1` (halved)
- `attention_levels=()` (no self-attention blocks)

This yields ~10M parameters — ~8× smaller, fitting comfortably into HPC memory budgets while still maintaining sufficient capacity to learn the diffusion task. FashionMNIST training uses `UnetSmall()` instead, a separate hand-tuned architecture for 28×28 images.

### Region Split: WEST vs. TEST

**Location:** [utils/dataloader.py](../utils/dataloader.py) `CARRA2.__init__()`, lines 38–46.

The CARRA2 zarr store covers the entire European/Atlantic domain. We define two spatial regions:

- **WEST** (default, real training region): `y=300:1516, x=884:2100` — covers the North Atlantic and western European shelf, the region of interest for sea-ice forecasting.
- **TEST** (small region, local dev): `y=1272:1400, x=800:928` — 128×128 crop, used for fast smoke-testing on a local Mac before committing to a full HPC run. Takes precedence over WEST when both flags are set.

### Device Handling: Local Mac vs. HPC

**Location:** [utils/device_utils.py](../utils/device_utils.py) `get_device()`.

The same codebase runs in two environments:
- **Local (Mac):** Falls back to `mps` (Metal Performance Shaders, Apple's GPU) or CPU if MPS unavailable.
- **HPC (DTU cluster):** Detects CUDA and uses it; [run.sh](../run.sh) loads `cuda/12.4.1` and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to allow dynamic memory allocation.

Tensors are moved to the device *inside* `Dataset.__getitem__()` (before returning from the dataloader), not in the training loop, so device-specific operations are localized.

### Known Inconsistency: Default Hyperparameters — OPEN

**Location:** [config.py](../config.py) presets, and the comments describing them in [main.py](../main.py)'s `__main__` block.

**This is not yet fixed — it is the outstanding item.** The presets and their descriptions disagree, and the values look swapped:

- `HPC_RUN = TrainConfig(batch_size=2, epochs=2, lr=1e-3)` — but `main.py`'s comment claims it is `batch_size=16, epochs=100`.
- `SMOKE_TEST = TrainConfig(batch_size=256, epochs=2, lr=1e-3)` — but the comment claims `batch_size=2, epochs=2`.

`run.sh` invokes `python main.py` with no CLI args, so whatever `__main__` selects is what runs on HPC. `train_cfg = HPC_RUN` currently means **288 optimizer steps in total** (288 monthly fields ÷ batch 2 × 2 epochs). DDPMs typically need 10⁵–10⁶ steps, so any run under this preset is severely undertrained regardless of the fixes above.

Also worth noting: 288 full 1216×1216 fields is a very small dataset for an unconditional generative model. Training on random crops is the natural next step.

**Before the next HPC submission**, decide the real values, fix the presets, and update the comments to match.

---

**Last Updated:** When architecture changes, update this file and regenerate [../config.py](../config.py) to reflect the new module map or design decisions.
