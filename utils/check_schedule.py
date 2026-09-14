"""Does the forward process actually destroy spatial structure, at every step, or only
by t = T?

The question that matters for sampling is not "is x_T noise by the end" but "is x_t
noise *early enough* that the network is forced to learn to generate structure, rather
than read it off its own input." This checks that directly, at a handful of spatial
scales, across the whole 0..T trajectory:

    x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise      (q_sample)

Noise is i.i.d. per pixel, so averaging x_t over a k x k block cuts the noise std by k,
while a spatially smooth x0 keeps its magnitude roughly unchanged. So:

  - Analytic amplitude SNR at scale k:   sqrt(alpha_bar_t / (1 - alpha_bar_t)) * k
  - Empirical check of the same thing: pool many real fields' x_t and their own x0 over
    k x k blocks, and correlate the two across fields. High correlation at scale k means
    x_t still tells you which field it came from, at that scale -- the forward process
    has not destroyed the image there yet.

A schedule that has genuinely destroyed the image drives *both* curves to ~0 well
before t = T, at every scale up to the full domain (k = image size). If they only reach
~0 in the last few steps, or never do at the domain scale, the network can shortcut
training by reading the leaked signal instead of learning to generate one -- see
docs/architecture.md, "Noise Schedule: Shifted Cosine, Resolution-Aware".

Run directly:  python -m utils.check_schedule
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.ddpm import DDPM

# Pooling block sizes: a mid-range local scale, the UNet bottleneck (152 x 152 at the
# WEST resolution), and the full domain (one number per field -- the sharpest test,
# since it asks "does x_t still say which field/month this is").
SCALES = (64, 152, 1216)


def check_invariants(ddpm: DDPM) -> None:
    """Assert the schedule and posterior coefficients are well formed."""
    tensors = {
        "betas": ddpm.betas,
        "alphas": ddpm.alphas,
        "alphas_bar": ddpm.alphas_bar,
        "posterior_mean_coef_x0": ddpm.posterior_mean_coef_x0,
        "posterior_mean_coef_xt": ddpm.posterior_mean_coef_xt,
        "posterior_variance": ddpm.posterior_variance,
    }
    for name, tensor in tensors.items():
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or inf"

    assert (ddpm.betas > 0).all() and (ddpm.betas <= 0.999).all(), "betas outside (0, 0.999]"
    assert (ddpm.alphas_bar[1:] < ddpm.alphas_bar[:-1]).all(), "alphas_bar not strictly decreasing"
    assert (ddpm.posterior_variance >= 0).all(), "negative posterior variance"

    # At t=0 the posterior must reduce to x0 exactly: mean = 1 * x0_pred + 0 * x_t.
    assert torch.isclose(ddpm.posterior_mean_coef_x0[0], torch.tensor(1.0), atol=1e-6), \
        f"coef_x0[0] = {ddpm.posterior_mean_coef_x0[0]:.6f}, expected 1.0"
    assert ddpm.posterior_mean_coef_xt[0].abs() < 1e-6, "coef_xt[0] != 0"
    assert ddpm.posterior_variance[0].abs() < 1e-6, "posterior_variance[0] != 0"

    print(f"  invariants OK  (coef_x0[0]={ddpm.posterior_mean_coef_x0[0]:.6f})")


def analytic_snr(ddpm: DDPM, t: int, k: int) -> float:
    """Amplitude SNR of x_t after average-pooling over k x k blocks, at timestep t."""
    abar = ddpm.alphas_bar[t]
    return ((abar / (1.0 - abar)).sqrt() * k).item()


def empirical_corr(ddpm: DDPM, x0: torch.Tensor, t: int, k: int) -> float:
    """corr(pool_k(x_t), pool_k(x0)) across the batch of fields in x0.

    x0: real fields, shape [B, 1, H, W], land already filled with 0. B > 1 is required
    at k = image size, where pooling reduces each field to a single number and the
    correlation is taken *across* fields (does x_t still say which field this is?).
    """
    t_batch = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
    xt, _ = ddpm.q_sample(x0, t_batch)

    a = F.avg_pool2d(x0, k).flatten()
    b = F.avg_pool2d(xt, k).flatten()
    if a.numel() < 2:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def leakage_over_time(ddpm: DDPM, x0: torch.Tensor, scales=SCALES, n_t: int = 25) -> dict:
    """Sweep t across 0..T-1 and return, per scale, the analytic and empirical curves.

    Returns {k: {"t": [...], "analytic": [...], "empirical": [...]}}.
    """
    t_values = torch.linspace(0, ddpm.timesteps - 1, n_t).round().long().unique().tolist()
    out = {k: {"t": t_values, "analytic": [], "empirical": []} for k in scales}
    for t in t_values:
        for k in scales:
            out[k]["analytic"].append(analytic_snr(ddpm, t, k))
            out[k]["empirical"].append(empirical_corr(ddpm, x0, t, k))
    return out


def plot_leakage(results: dict, save_path: str = "utils/schedule_leakage.png") -> None:
    """One row per schedule, one line per scale: empirical corr(pool(x_t), pool(x0)) vs t.

    This is the plot that answers the question directly: for each schedule, at what t
    does each spatial scale actually go to ~0? A flat line near 1 that only drops at the
    very end means the network can shortcut training at that scale for most of t.
    """
    schedules = list(results.keys())
    fig, axes = plt.subplots(1, len(schedules), figsize=(6 * len(schedules), 4.5), sharey=True)
    if len(schedules) == 1:
        axes = [axes]

    for ax, schedule in zip(axes, schedules):
        for k, curves in results[schedule].items():
            ax.plot(curves["t"], curves["empirical"], marker="o", markersize=3, label=f"k={k}")
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(schedule)
        ax.set_xlabel("timestep t")
        ax.set_ylim(-0.2, 1.05)
        ax.legend(title="pool size")

    axes[0].set_ylabel("corr(pool(x_t), pool(x0))  across fields")
    fig.suptitle("Does q_sample destroy spatial structure before t = T?")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  saved plot to {save_path}")


def _load_fields(n: int = 96, device: str = "cpu"):
    """Real CARRA2-WEST fields in model range, land filled the way training fills it."""
    from utils.dataloader import CARRA2

    ds = CARRA2("siconc", device, WEST=True, batch_dim=False)
    # Spread evenly across the whole record so the fields span the seasonal cycle. This
    # matters at k = image size: correlation measures discrimination *between* fields,
    # so it needs fields that actually differ in domain-mean ice, not 96 copies of the
    # same season.
    idx = torch.linspace(0, len(ds) - 1, min(n, len(ds))).round().long().tolist()
    x0 = torch.stack([ds[i] for i in idx])
    return torch.nan_to_num(x0, nan=0.0), ds.finite_mask


if __name__ == "__main__":
    device = "cpu"
    torch.manual_seed(0)
    x0, mask = _load_fields(device=device)
    n = x0.shape[0]
    print(f"fields: {tuple(x0.shape)}   (corr standard error ~{1 / n ** 0.5:.2f})\n")

    all_results = {}
    for schedule in ("linear", "shifted_cosine"):
        ddpm = DDPM(timesteps=1000, device=device, land_mask=mask, schedule=schedule)
        print(f"{schedule}  (image_size={ddpm.image_size}, ref={ddpm.shift_ref_resolution})")
        check_invariants(ddpm)

        curves = leakage_over_time(ddpm, x0)
        all_results[schedule] = curves

        # Print a coarse summary: at t=T, and the first t where each scale's empirical
        # correlation drops below 0.1 (a soon-to-be-noise threshold, not a hard rule).
        print(f"  {'scale k':>9} {'corr @ t=0':>12} {'corr @ t=T-1':>14} {'first t with corr<0.1':>24}")
        for k in SCALES:
            c = curves[k]
            first_below = next((t for t, v in zip(c["t"], c["empirical"]) if v == v and v < 0.1), None)
            first_str = str(first_below) if first_below is not None else f"never (by t={c['t'][-1]})"
            print(f"  {k:>9} {c['empirical'][0]:>12.4f} {c['empirical'][-1]:>14.4f} {first_str:>24}")
        print()

    plot_leakage(all_results)
