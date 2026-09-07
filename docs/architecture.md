# Architecture & Design Decisions

## Module Map

- **[main.py](../main.py)** — Entry point. `run(carra=False, fashion=False, ...)` builds the dataset/dataloader, instantiates a UNet and DDPM, and trains. `download_samples(ddpm, n_of_samples)` runs inference and pickles outputs to `results/{ddpm.output_name}/samples.pkl`.
- **[models/ddpm.py](../models/ddpm.py)** — `DDPM` class: orchestrates diffusion training and sampling. Methods:
  - `beta_schedule()` — linear schedule from `start` to `end` over `timesteps`.
  - `get_alphas()` — derives `alpha` and `alpha_bar` from betas.
  - `q_sample(x0, t, noise)` — forward diffusion process, adds noise to an image at timestep `t`.
  - `compute_loss(model, x0, t)` — loss function with NaN-masking for land pixels (see Design Decisions below).
  - `p_sample(model, x, t)` — reverse diffusion process, one denoising step.
  - `sample(model)` — generates a sample from pure noise by iterating `p_sample` backwards.
  - `train(model, dataloader, device, timesteps, epochs, lr)` — training loop with periodic checkpointing.
- **[models/unet.py](../models/unet.py)** — UNet architectures.
  - `Unet` (general-purpose, resolution-agnostic) — configurable via `base_channels`, `channel_mult`, `num_res_blocks`, `attention_levels`, etc. Defaults: `base_channels=128`, `channel_mult=(1,2,2,2,4)`, `num_res_blocks=2`, `attention_levels=(3,)` (~78.7M params).
  - `UnetSmall` (fixed for small images) — compact version for FashionMNIST (28×28).
- **[utils/dataloader.py](../utils/dataloader.py)** — Dataset classes.
  - `CARRA2(selected_variable, device, time_slice, WEST, TEST, batch_dim)` — loads CARRA2 reanalysis data from a zarr store. `siconc` is sea-ice concentration.
  - `FashionMNIST(train, device, batch_dim)` — wraps torchvision's FashionMNIST, auto-downloads.
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

### UNet Downsizing for CARRA2

**Location:** [main.py](../main.py) `run()`, lines 42–49.

The `Unet` class's own defaults are `base_channels=128, channel_mult=(1,2,2,2,4), num_res_blocks=2, attention_levels=(3,)`, which yields ~78.7M parameters and high memory usage. For CARRA2 training on DTU HPC (64GB system memory, 1×80GB GPU per [run.sh](../run.sh)), we use a smaller configuration:

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

### Known Inconsistency: Default Hyperparameters

**Location:** Discrepancy between [main.py](../main.py) function defaults and the `__main__` block.

- `run()` function signature (line 10-17) has defaults: `timesteps=100, batch_size=16, epochs=100, lr=1e-3`.
- `__main__` block (lines 72-77) actually calls `run(carra=True, timesteps=1000, batch_size=2, epochs=2)` — **very different values**.

The `__main__` block values (`batch_size=2, epochs=2`) are what get deployed to HPC via `run.sh` (which just invokes `python main.py` with no CLI args). This looks like a leftover smoke-test configuration rather than a genuine full-scale HPC run. See [config.py](../config.py) for the canonicalized configuration.

---

**Last Updated:** When architecture changes, update this file and regenerate [../config.py](../config.py) to reflect the new module map or design decisions.
