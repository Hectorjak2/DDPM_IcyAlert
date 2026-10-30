# Architecture & Design Decisions

## Module Map

- **[main.py](../main.py)** — Entry point. `run(carra=False, fashion=False, ...)` builds the dataset/dataloader, instantiates a UNet and DDPM, and trains. `download_samples(ddpm, n_of_samples)` runs inference and pickles outputs to `{TrainConfig.samples_dir}/{experiment_name}/samples.pkl`.
- **[ensemble.py](../ensemble.py)** — Ensemble forecasts from a trained `.pth`, configured by `EnsembleConfig` in [config.py](../config.py). `method="dropout"`: MC dropout through the residual UNet baseline. It uses `model.train()` during inference, which is safe because `SimpleUnet` has only GroupNorm and so `train()` toggles nothing but dropout. `method="ddpm"`: the conditional DDPM sampled `n_members` times, each member re-seeded (`torch.manual_seed(member)`) so its initial noise is distinct and reproducible. `method="deep"`: a deep ensemble of residual UNets, one per seed in `ENSEMBLE_SEEDS`, each trained by `train_ensemble` (one member per call, so one per HPC job) into `{EnsembleConfig.members_dir}/member_{seed}/`. A job claims its seed by atomically creating `member_{seed}/claimed` (`O_EXCL`) when it starts, so jobs that pick a seed automatically skip seeds that are trained (`final_weights.pth` exists) or running. A crashed member keeps its claim and is rerun by forcing it with `--seed` / `ENSEMBLE_SEED`. `run_unet(initialization_seed=...)` seeds the global RNG before building the model, so the seed sets the initial weights and also the batch order and the dropout masks during training, as in Lakshminarayanan et al. 2017. Members are sampled with `eval()` (dropout off), so all spread comes from the different training runs. Dates are drawn with `TrainConfig.sample_seed`, so they match the single-forecast runs. Sampling itself is in [utils/sampling.py](../utils/sampling.py); `sample_ensemble` only picks the forecaster. Writes `ensemble_samples.pkl` as `(samples, contexts, targets, times, member_seeds)`, with each `samples[i]` shaped `[n_members, lead_months, H, W]`. Scored by [plotting/ensemble_evaluation.py](../plotting/ensemble_evaluation.py); `plotting/evaluation.py` expects one forecast per date.
- **[utils/sampling.py](../utils/sampling.py)** — All sampling. `Forecaster` adapters (`UnetForecaster`, `DDPMForecaster`) map a context `(k, k-1)` in SIC to a forecast of k+1 in SIC; factories `dropout_forecaster` / `deep_forecaster` / `ddpm_forecaster` build them from weights. `forecast()` is the one loop for single runs (`member_seeds=[0]`, writes `samples.pkl`), ensembles and roll-outs (see Sampling & Roll-out below). Also holds `download_samples` (unconditional) and `analysis_sampling` (conditioning diagnostics, `loss_stats.pkl`). [utils/helper_functions.py](../utils/helper_functions.py) keeps only run bookkeeping (`dump_config_snapshot`, `schedule_summary`).
- **[models/ddpm.py](../models/ddpm.py)** — `DDPM` class: orchestrates diffusion training and sampling. The schedule tensors (`betas`, `alphas`, `alphas_bar`, `alphas_bar_prev`, posterior coefficients) are precomputed once in `__init__`. Takes an optional `land_mask` (see Land Mask as Input Channel below). Methods:
  - `beta_schedule()` — linear schedule from `start` to `end` over `timesteps`.
  - `get_alphas()` — derives `alpha` and `alpha_bar` from betas.
  - `model_input(xt)` — assembles the network input, appending the static land-mask channel when one was given.
  - `q_sample(x0, t, noise)` — forward diffusion process, adds noise to an image at timestep `t`.
  - `compute_loss(model, x0, t)` — loss function with NaN-masking for land pixels (see Design Decisions below).
  - `p_sample(model, x, t)` — reverse diffusion process, one denoising step, parameterised through a clamped predicted `x0`.
  - `sample(model)` — generates a sample from pure noise by iterating `p_sample` backwards; returns data-range `[0, 1]` with land set to NaN.
  - `train(model, dataloader, device, timesteps, epochs, lr)` — training loop with periodic checkpointing and per-timestep-bucket loss reporting.
- **[models/climatology.py](../models/climatology.py)** — Climatology baseline: forecasts k+1 as the mean SIC of its calendar month over `TRAINING_SLICE` (training years only, so the test dates never leak in). Fitting reads every training day, so it is run once on HPC (`python -m models.climatology --area CARRA_WEST`) and cached as `climatology_<AREA>.npy` under `CARRA2_DATA_ROOT`. [plotting/evaluation.py](../plotting/evaluation.py) `print_stats` loads the cache whose grid matches the samples (896×896 samples use `climatology_cropped.npy`, cropped to the `CROPPED_Y`/`CROPPED_X` window from [utils/dataloader.py](../utils/dataloader.py) if it was fitted on full WEST; a per-pixel mean crops exactly) and prints its MSE next to the persistence baseline (k+1 = k).
- **[models/unet.py](../models/unet.py)** — UNet architectures.
  - `Unet` (general-purpose, resolution-agnostic) — configurable via `base_channels`, `channel_mult`, `num_res_blocks`, `attention_levels`, `mid_attention`, etc. Defaults: `base_channels=128`, `channel_mult=(1,2,2,2,4)`, `num_res_blocks=2`, `attention_levels=(3,)`, `mid_attention=True` (~78.7M params). Note that `attention_levels` governs the down/up paths only — the bottleneck is controlled separately by `mid_attention` (see Attention Placement).
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
- **[plotting/](../plotting/)** — All plotting code, split out of `utils/`. A regular package; run from the repo root so imports resolve as `plotting.*`.
  - [plotting/visualization.py](../plotting/visualization.py) — Sample plotting helpers (`visualize_sic_sample`, `visualize_fashion`, `plot_samples_grid`).
  - [plotting/conditional_visualization.py](../plotting/conditional_visualization.py) — Conditional forecast plots (context, ground truth, prediction).
  - [plotting/schedules.py](../plotting/schedules.py) — `Schedule` spec (kind + parameters + label) for the noise-schedule figures [plotting/schedule_comparison.py](../plotting/schedule_comparison.py), [plotting/logsnr_distributions.py](../plotting/logsnr_distributions.py) and [plotting/forward_process.py](../plotting/forward_process.py). Each script lists its schedules in a settings block at the top instead of reading `DDPMConfig`, so one figure can show several variants (e.g. shifted cosines at different `cosine_shift_res`). Parameters left out fall back to `DDPMConfig`. Curves come from the same `models.ddpm.build_schedule` that training uses.
  - [plotting/ensemble_evaluation.py](../plotting/ensemble_evaluation.py) — Rank histogram and spread-skill for `ensemble_samples.pkl`, both over the marginal ice zone only: pixels with 0 < SIC < 1 on day k. Open water and pack ice were dropped (an "all water" panel existed earlier): there the truth ties with nearly every member, random tie-breaking spreads those pixels evenly over the ranks, and as they are most of the ocean they make the histogram near-flat by construction. The zone is selected on day k, not on the target, because conditioning on the observation biases the histogram. Ranks are per pixel, with the remaining ties broken at random (Hamill 2001); counting only members strictly below the truth would put every tie in rank 0 and fake a U shape. `spread_skill_map.pdf` shows spread, RMSE of the ensemble mean and their ratio per pixel (rows: cool and melt season, since the ice edge moves between them). Unlike the histogram, it uses all ocean pixels: the map shows where the ice is by itself, and a per-date day-k mask would give each pixel a different number of dates. The ratio is masked where spread < `MIN_MAP_SPREAD` (shown white; land is grey), because there both are ~0 and the ratio is noise. The same mask restricts the printed per-pixel spread–RMSE correlation, so the trivially predictable ocean does not inflate it. Spread and RMSE share one colour scale; the ratio uses a log norm centred at 1.
  - [plotting/model_comparison.py](../plotting/model_comparison.py) — `calibration` and `mse_by_lead` for two `samples.pkl` (e.g. UNet vs DDPM) in one figure each, with climatology in `mse_by_lead`. Persistence is left out of `mse_by_lead`: it grows far past both models with lead time, flattening them against the axis. Reuses the loaders and statistics of [plotting/evaluation.py](../plotting/evaluation.py), which stays the per-model evaluation. The climatology baseline comes from the first set, so both sets should be sampled on the same dates (same `sample_seed`); it warns otherwise and plots only the lead times both sets have.
  - [plotting/plot_config.py](../plotting/plot_config.py) — Thesis figure style (10pt serif, `usetex`) plus `savefig` and `TEXTWIDTH_IN` (thesis `\textwidth` = 448.13pt). Thesis figures are saved as vector PDFs to `figures/`; use the `/thesis-plot` skill. Needs a local LaTeX install, so keep it out of HPC code paths.
  - `finite_TEST.pt` / `finite_WEST.pt` — Saved land/water masks used by the visualization scripts (WEST-shaped vs TEST-shaped, so they are not interchangeable).
- **[data/](../data/)** (gitignored) — `CARRA2_MONTHLY/dataset.zarr` (zarr store, large) and `FashionMNIST/raw` (auto-downloaded by torchvision).
- **[results/](../results/)** (gitignored) — Checkpoints and sampled outputs, organized by `{model_name}_{dataset_name}/`.
- **[run.sh](../run.sh)** — DTU HPC batch script (LSF/BSUB) that loads cuda/12.4.1, sets PyTorch config, and invokes `python main.py`.

## Design Decisions

### Sampling & Roll-out

**Location:** [utils/sampling.py](../utils/sampling.py), [utils/dataloader.py](../utils/dataloader.py) `CARRA2Forecast(lead_months=)` / `get_rollout`, `TrainConfig.lead_months`.

- **Forecasters work in SIC at the boundary.** Each converts to its own range internally (`to_model_range` for the DDPM, training mean/std for the UNet), so a prediction can be fed straight back as the next context without the loop knowing the model. The test set's own scaling is undone with `CARRA2.to_sic`, so one test set serves every model. `ConditionalDDPM.sample` stays in the model; `DDPMForecaster` is only an adapter.
- **Roll-out is autoregressive**: from `(k, k-1)` predict k+1, then `(k+1_pred, k)` → k+2, …, month advancing each step. Models are trained only on one-month steps, so there is no direct multi-month alternative without retraining.
- **One seed per trajectory, not per step** (`torch.manual_seed(member)` before each member's roll-out). A member's whole path is reproducible, and the lead-1 slice of a roll-out is bit-identical to a `lead_months=1` forecast on the same dates (verified for deep, dropout, ddpm and single UNet). The refactor was also verified bit-identical to the pre-refactor `run_ensemble` / `download_unet_samples` outputs.
- **`lead_months > 1` changes the date set.** Only anchors with ground truth at every lead are kept (the last `lead_months-1` of each contiguous block drop out), so `sample_seed` draws different dates than at lead 1. Compare models and leads only within runs that used the same `lead_months`.
- **Output layout**: `samples[i]` `[n_members, lead_months, H, W]`, `targets[i]` `[lead_months, H, W]`, `times[i]` `(k+1, …, k+lead_months, k, k-1)`. At lead 1 this is the old ensemble layout. Both evaluators take `--lead L` and write to `lead<L>/`. Single forecasts no longer store the DDPM's running sampling loss; use `analysis_sampling` for that diagnostic.

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

**Why it matters:** Training on un-normalized `[0, 1]` data mismatches the noise schedule's assumed signal scale. An earlier run without this produced samples with mean 1.28 and range `[-11, 25]` — entirely outside the valid concentration range, and clipped to near-uniform white by the `vmin=0, vmax=1` plotting in [plotting/visualization.py](../plotting/visualization.py).

**Consequence:** Checkpoints trained before this change are not compatible — they learned a different input distribution.

### Land Mask as Input Channel

**Location:** [models/ddpm.py](../models/ddpm.py) `DDPM.__init__()` and `model_input()`; wired up in [main.py](../main.py) `run()`.

`compute_loss()` replaces land NaNs with `0.0` before the convolutions. In `[-1, 1]` space `0.0` is a *legal* concentration (SIC = 0.5), so from the network's point of view land is indistinguishable from ordinary water. Masking the loss stops it from being penalised on land, but never tells it where the coastline is — and at sampling time there is no `x0` to derive a mask from at all, so an unconditional sample has no way to place Greenland.

The static water/land mask is therefore passed to `DDPM` and concatenated as a second input channel (`+1` water, `-1` land, matching the image channel's scale). This makes `UnetConfig.in_channels = 2` for CARRA2; `out_channels` stays 1 (the predicted noise). `sample()` re-applies the mask, setting land back to NaN.

The mask is time-invariant in CARRA2 (verified across the full 288-month record), so `CARRA2.__init__` derives it once from the first time step rather than per sample. FashionMNIST passes `land_mask=None`, in which case `model_input()` is a pass-through and `UnetSmall` stays 1-channel.

### Clamped x0 Parameterization in `p_sample`

**Location:** [models/ddpm.py](../models/ddpm.py) `p_sample()`.

The reverse step is computed via the predicted `x_0` rather than directly from the noise:

1. `x0_pred = (x_t - sqrt(1-ᾱ_t)·ε_θ) / sqrt(ᾱ_t)`, then **clamped to `[-1, 1]`**. In `ConditionalDDPM`, `x0` is the residual, so it is clamped to `[-1 - k, 1 - k]` instead (see *Residual Prediction for Forecasting*).
2. `mean = coef_x0(t)·x0_pred + coef_xt(t)·x_t`, the posterior mean of `q(x_{t-1} | x_t, x_0)` (Ho et al. eq. 7), with variance `β̃_t = β_t(1-ᾱ_{t-1})/(1-ᾱ_t)`.

Without the clamp this is algebraically identical to the previous direct form `1/sqrt(α_t)·(x_t - (1-α_t)/sqrt(1-ᾱ_t)·ε_θ)` (verified numerically to ~1e-8). The clamp is the point: over T=1000 reverse steps, unconstrained errors compound and drive the sample far outside the data range.

Also fixed here: noise was previously added whenever `t > 1`, so the final step was stochastic. It is now `t > 0`, making the last step deterministic as intended.

### Residual Prediction for Forecasting

**Location:** [models/ddpm.py](../models/ddpm.py) `ConditionalDDPM`, [models/hec_unet.py](../models/hec_unet.py) `train_network`, [utils/sampling.py](../utils/sampling.py) `UnetForecaster`.

Both forecasters predict the month-to-month change `k+1 − k` rather than the field `k+1`, and reconstruct by adding day `k` back. Land is NaN→0 in both fields, so its residual is 0 ("no change").

- **ConditionalDDPM** diffuses the residual (model units, range `[-2, 2]`). The `x0_pred` clamp is `[-1 − k, 1 − k]`, so that `k + x0_pred` stays in `[-1, 1]`. A `±1` clamp would cap the change at ±0.5 SIC; month-apart changes reach the full ±1 SIC (measured: max |residual| = 2.0 for 2000-03-15 → 2000-04-15). `sample(..., base=)` sets the field the residual is added to, and its clamp bounds. `analysis_sampling` passes the true `k` with a wrong context, so the wrong-context loss measures reliance on conditioning only, not a wrong baseline.
- **UNet baseline** (standardized units) outputs the residual `(k+1 − k)/std`; the mean cancels. The loss is taken on the reconstruction `k + pred`, which equals the loss on the residual, so the persistence ratio printed in training stays directly comparable.
- **`SimpleUnet.final_conv` is zero-initialised**, so at init the baseline is exactly persistence (ratio 1.0) and the DDPM predicts zero noise. This is shared by both models; the same choice as Dhariwal & Nichol's output layer.

**Open: residual scale vs. noise schedule.** The schedule was tuned for full fields. The residual has a smaller std (0.34 vs. 0.92 in model units for the date pair above, near the ice maximum; melt/freeze-up months will differ), which shifts the effective log-SNR by `2·log(σ_r/σ_x) ≈ −2.0` on top of the schedule's own shift. Not yet resolved: either divide the residual by a training-set `σ_r`, or retune `cosine_shift_res`.

### Noise Schedule: Shifted Cosine, Resolution-Aware

**Location:** [models/ddpm.py](../models/ddpm.py) module-level `build_schedule()` (takes the parameters explicitly; defaults come from `DDPMConfig`, and `DDPM.build_schedule` calls it with those defaults), [config.py](../config.py) `DDPMConfig`.

**The linear schedule does not destroy large-scale structure at 1216×1216.** DDPM sampling starts from `x_T ~ N(0,I)`, which is valid only if `q(x_T)` is genuinely close to `N(0,I)`. Noise is i.i.d. per pixel, so averaging `x_t` over a `k×k` block cuts the noise std by `k` while a spatially smooth `x_0` keeps its magnitude — the amplitude SNR at scale `k` is `sqrt(ᾱ/(1-ᾱ))·k`. Under `β = 1e-4 → 0.02` at t=999:

| pooling scale k | 1 | 16 | 64 | 152 (bottleneck) | 1216 (domain) |
|---|---|---|---|---|---|
| analytic SNR | 0.006 | 0.10 | 0.41 | 0.97 | **7.73** |
| measured corr(pool(x_T), pool(x_0)) | 0.005 | 0.07 | **0.26** | **0.50** | **0.73** |

(measured over 96 real fields × 5 noise seeds; see `utils/check_schedule.py`)

So `x_999` still reports which field it came from. The network reads the global layout off its input at every `t` and is never required to generate one — the bottleneck `AttentionBlock` sits at 152×152, exactly the scale where SNR ≈ 1. At sampling time `x_T` is a true `N(0,I)` draw, pooling it gives ≈0, and the field collapses to the prior mean. This is the measured failure: samples had mean 0.52 over water (= 0.0 in model range) against a data mean of 0.271, spatial std 0.074 against ~0.40, and 0.00002% of water pixels near zero against 62%.

**The leak grows with linear image size** — 0.41 at 64², 7.73 at 1216² — which is why an absolute `β` range does not transfer between resolutions. (Chen 2023, *On the Importance of Noise Scheduling*; Hoogeboom et al., *simple diffusion*.)

The schedule is now defined on `ᾱ` rather than on `β`: a cosine schedule (Nichol & Dhariwal) whose log-SNR is shifted down by `2·log(shift_ref_resolution / image_size)`, with `β` recovered as `1 - ᾱ_t/ᾱ_{t-1}` clamped to ≤0.999 and `ᾱ` then **recomputed from the clamped β** so the two stay mutually consistent. Measured: domain-scale SNR 7.73 → 0.0032, and the number of steps that actually destroy 64px-scale structure rises from 94/1000 to 186/1000.

Two things to know:

- **Plain cosine is not the fix.** It closes the endpoint leak but holds `ᾱ` high for most of the trajectory and crashes at the end, leaving only 11/1000 such steps — worse than linear's 94. The *shift* is the operative part. `ref=64` is the *simple diffusion* prescription; `ref=128` is the conservative fallback.
- **`β` is non-monotone under this schedule, by design.** It is U-shaped: ~0.030 at t=25, a minimum of ~0.0062 at t=494, rising to the clamp at t=T. A constant log-SNR shift leaves `dλ/dt` unchanged but moves the point where `ᾱ = 0.5` much earlier, and `β ≈ -(1-σ(λ))·dλ/dt` is large at both ends of the cosine's steep log-SNR curve. Monotone `β` is a convention of the linear/plain-cosine schedules, not a requirement — the derivation needs only `0 < β < 1`, `ᾱ` strictly decreasing, and non-negative posterior variance.

**The clamp in `p_sample` is load-bearing for this schedule.** The shift drives `ᾱ_T` to 6.7e-12, so recovering `x̂₀` divides by as little as 2.6e-6. The quantity that matters is the end-to-end sensitivity `posterior_mean_coef_x0 × 1/sqrt(ᾱ)`, which stays ≤0.04 at every `t` except t=999 where it is 31.6 — and there the `clamp(-1,1)` bounds the contribution to `coef_x0 × 1 = 8e-5`. Without the clamp this schedule would require v-prediction. Do not remove it.

**Schedule construction is float64**, cast to float32 only at the end, including the posterior coefficients. All of them divide by `1 - ᾱ`, which at small `t` subtracts two nearly equal numbers; in float32 that cancellation made `posterior_mean_coef_x0[0]` come out as 0.999834 instead of exactly 1.0. It is now 1.000000.

`utils/check_schedule.py` verifies both the invariants and the leakage empirically. Note that the domain-scale row has a standard error of roughly `1/sqrt(n_fields)` — with too few fields it swings by ±0.4 and is not interpretable, and the fields must span the seasonal cycle or there is no between-field variance for the correlation to detect.

**Laplace schedule (`schedule="laplace"`).** The valid schedule names live in `DDPM.SCHEDULES`, which the plotting scripts iterate over. Laplace (Hang et al. 2024) sets `λ(t) = μ − b·sgn(0.5−t)·log(1−2|t−0.5|)`, the inverse CDF of Laplace(μ, b), evaluated at step midpoints `t = (i+0.5)/T`, with `ᾱ = σ(λ)` and β recovered the same way as for shifted cosine. **It has no resolution shift**, and with the paper's defaults (μ=0, b=0.5) the log-SNR only spans about ±3.5 over 1000 steps: `ᾱ_0 = 0.969`, `ᾱ_T = 0.031`. At 1216² that leaves a domain-scale SNR of about 217 at t=T, far above the linear schedule's 7.73. Before training on WEST with it, move μ down (e.g. to the shift `2·log(64/1216) ≈ −5.9`) and/or raise b.

### Per-Timestep Loss Reporting

**Location:** [models/ddpm.py](../models/ddpm.py) `train()`.

**A single averaged loss is not a useful diagnostic for this model.** With linear betas 1e-4 → 0.02 over T=1000, `sqrt(ᾱ_t)` is 0.28 by t=500 and 0.006 by t=999, so for most of `t ~ U(0,T)` the input is already nearly pure noise and predicting ε collapses to close to the identity map. The CARRA2 field makes this worse: 55% of water pixels are exactly 0, and over a locally constant patch ε-prediction reduces to an affine map a conv net fits within a handful of gradient steps.

Measured on the 2-epoch run, per timestep: 0.45 at t=5, 0.042 at t=150, 0.0095 at t=500, 0.0045 at t=900. The scalar "loss ≈ 0.0075" that looked like convergence was just the high-t bucket; the low-t regime that actually generates structure was no better than predicting zero.

`train()` therefore accumulates loss into 5 timestep buckets and prints a per-bucket summary each epoch. Watch the low-t buckets; a healthy run flattens the profile.

**Caveat after the shifted-cosine change — the ε-loss became *less* informative, not more.** The shortcut available to the network is `ε̂ = x_t`, whose MSE is approximately `ᾱ_t · E[x_0²]`. Since the shifted schedule has lower `ᾱ` at every `t`, that shortcut gets cheaper everywhere:

| bucket | linear | shifted cosine |
|---|---|---|
| t[0–199] | 5.34e-01 | 1.23e-01 |
| t[400–599] | 5.43e-02 | 1.75e-03 |
| t[800–999] | 2.63e-04 | 5.66e-05 |

So bucket losses will drop across the board after the schedule change **for reasons unrelated to sample quality**. Do not read that as improvement. This is a general property of ε-prediction at low SNR and is the main argument for moving to v-prediction, whose loss is naturally SNR-balanced. Until then, the only trustworthy signal is sample statistics compared against the data: mean over water (0.271), spatial std (~0.40), fraction of water pixels below 0.05 (62%), and pairwise correlation between samples.

### UNet Downsizing for CARRA2

**Location:** [main.py](../main.py) `run()`, lines 42–49.

The `Unet` class's own defaults are `base_channels=128, channel_mult=(1,2,2,2,4), num_res_blocks=2, attention_levels=(3,)`, which yields ~78.7M parameters and high memory usage. For CARRA2 training on DTU HPC (64GB system memory, 1×80GB GPU per [run.sh](../run.sh)), we use a smaller configuration:

- `in_channels=2` (image + land mask; see Land Mask as Input Channel)
- `base_channels=64` (halved)
- `channel_mult=(1,2,2,2)` (removed the 4x level)
- `num_res_blocks=1` (halved)
- `attention_levels=()` (no self-attention in the down/up paths)
- `mid_attention=False` (no self-attention in the bottleneck either; see Attention Placement)

This yields ~5.8M parameters, fitting comfortably into HPC memory budgets while still maintaining sufficient capacity to learn the diffusion task. FashionMNIST training uses `UnetSmall()` instead, a separate hand-tuned architecture for 28×28 images.

### Attention Placement

**Location:** [models/unet.py](../models/unet.py) `Unet.__init__()`, bottleneck construction.

`attention_levels` only ever governed the down/up paths. The bottleneck `AttentionBlock` was unconditional, so `attention_levels=()` — documented here and in [config.py](../config.py) as "no attention" — still built one attention block at 1/8 resolution. At the WEST resolution that block sits at 152×152 = 23,104 tokens, where attention is quadratic in the token count. On `mps`, and on CUDA whenever SDPA falls back to the math backend, this materialises a `[B, heads, N, N]` matrix: ~34 GB at batch 4 in fp32. It was the single largest memory item in the network and the reason batch 4 would not fit at 1216×1216.

`mid_attention` now controls it explicitly, rather than being inferred from an empty `attention_levels`. The two are kept separate because attention in the bottleneck and attention at a given down/up level are genuinely different design choices — the standard Ho et al. configuration uses both, and wanting one without the other is reasonable. The default stays `mid_attention=True` so the class still reproduces Ho et al. and so pre-2026-09-09 checkpoints (whose state dicts carry `mid.layers.1.qkv.*` keys) load unchanged. The CARRA2 config sets it to `False`.

Profiling both this network and a colleague's attention-free `SuperResUNet` ([models/clara_unet.py](../models/clara_unet.py)) showed their convolutional activation costs are within 3% of each other at every resolution level, so attention placement and input resolution — not channel widths or block counts — are what actually drive memory here. At batch 4, activations scale as H·W: ~40 GB at 1000×1000 versus ~61 GB at 1216×1216.

### Axial Attention in `SimpleUnet`'s Bottleneck

**Location:** [models/hec_unet.py](../models/hec_unet.py) `AttentionBlock`, used between the two bottleneck ResBlocks of `SimpleUnet`.

Full self-attention at the 152×152 bottleneck is infeasible (see above), so `SimpleUnet` uses factorised (axial) attention instead: each pixel attends to every pixel in its row, and separately to every pixel in its column, and the two results are summed. That is O(H·W·(H+W)) rather than O((H·W)²). At `[4, 128, 152, 152]` it costs ~2 GB and ~0.3 s forward+backward on `mps`. Two layers of axial attention would give every pixel a path to every other pixel; one layer reaches only its row and column.

- **Layout:** the input is permuted to channels-last `[B, H, W, C]` before reshaping into sequences. Reshaping `[B, C, H, W]` directly cuts memory in channel-major order, so the "sequences" end up as chunks of single channels rather than rows of pixels. An earlier version had exactly this bug: shapes were correct, nothing crashed, and the semantics were wrong.
- **Projections:** q/k/v come from a full-channel `Linear(C, 3C)` and are split into heads afterwards, so each head learns its own projection. Rows and columns have separate q/k/v projections (`qkv_row`, `qkv_col`), since the two grid axes need not behave alike.
- **Output projection:** `proj_out` is zero-initialised, so the block is the identity at init and does not perturb the residual stream early in training.

Checked against a naive per-row/per-column loop (max diff ~7e-7); with the GroupNorm bypassed, perturbing one pixel changes only its row and column.

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

Output paths are split the same way. Weights and `config.txt` go to `{TrainConfig.output_dir}/{experiment_name}/` (`results/` in both environments, so `EnsembleConfig.weights` / `members_dir` resolve the same everywhere). Samples (`samples.pkl`, `ensemble_samples.pkl`, `loss_stats.pkl`) go to `{TrainConfig.samples_dir}/{experiment_name}/`, which [config.py](../config.py) reads from the `TEMP_RESULT_STORAGE` env var (default `results`). [run.sh](../run.sh) points it at `/work3` because sample pickles are too large for the home quota. `dump_config_snapshot` writes `config.txt` to both dirs, so a sample set on `/work3` can be traced to its config without the weights folder. Ensemble sampling trains nothing, so `dump_ensemble_snapshot` writes its `config.txt` (with `EnsembleConfig`, and the schedule for `method="ddpm"`) only next to `ensemble_samples.pkl`. The env var is read only in config.py, so every entry point (`main.py`, `ensemble.py`, the runners) writes to the same place; scripts should not override `output_dir` to reroute samples.

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
