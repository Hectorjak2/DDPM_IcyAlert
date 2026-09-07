"""Checks on the diffusion noise schedule.

Two things are verified here:

1. ``check_invariants`` -- the schedule and the posterior coefficients derived from it are
   finite, ordered, and reduce to x0 at t=0.
2. ``check_leakage`` -- the reason the schedule was changed. Noise is i.i.d. per pixel, so
   averaging x_t over a k x k block cuts the noise std by k while a spatially smooth x0
   keeps its magnitude. If the correlation between pooled x_T and pooled x0 is still high
   at large k, the forward process has not actually destroyed the image: the network can
   read the global layout off its input instead of learning to generate one, and samples
   started from a true N(0, I) collapse to the prior mean.

Run directly:  python -m utils.check_schedule
"""

import torch
import torch.nn.functional as F

from models.ddpm import DDPM

SCALES = (1, 16, 64, 152, 1216)


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

    # Monotone betas are a property of the linear and unshifted-cosine schedules, not a
    # requirement of the derivation -- which needs only 0 < beta < 1, alpha_bar strictly
    # decreasing, and non-negative posterior variance, all asserted above.
    #
    # The shifted cosine is deliberately non-monotone: it is U-shaped, peaking early
    # (~0.030 at t=25), dipping mid-trajectory (~0.0062 at t=494) and rising to the clamp
    # at t=T. A constant log-SNR shift leaves d(logSNR)/dt untouched but moves the point
    # where alpha_bar = 0.5 much earlier, and beta ~ -(1 - sigmoid(logSNR)) * d(logSNR)/dt
    # is then large at both ends of the cosine's steep log-SNR curve. Not a bug.
    if ddpm.schedule == "linear":
        assert (ddpm.betas[1:] >= ddpm.betas[:-1] - 1e-6).all(), "linear betas not monotone"

    # At t=0 the posterior must reduce to x0 exactly: mean = 1 * x0_pred + 0 * x_t, no noise.
    assert torch.isclose(ddpm.posterior_mean_coef_x0[0], torch.tensor(1.0), atol=1e-6), \
        f"coef_x0[0] = {ddpm.posterior_mean_coef_x0[0]:.6f}, expected 1.0"
    assert ddpm.posterior_mean_coef_xt[0].abs() < 1e-6, "coef_xt[0] != 0"
    assert ddpm.posterior_variance[0].abs() < 1e-6, "posterior_variance[0] != 0"

    print(f"  invariants OK  (coef_x0[0]={ddpm.posterior_mean_coef_x0[0]:.6f})")


def pooled_snr(ddpm: DDPM, t: int, scales=SCALES) -> dict:
    """Analytic amplitude SNR of x_t after average-pooling over k x k blocks."""
    abar = ddpm.alphas_bar[t]
    snr = (abar / (1.0 - abar)).sqrt().item()
    return {k: snr * k for k in scales}


def check_leakage(ddpm: DDPM, x0: torch.Tensor, t: int = None, scales=SCALES) -> dict:
    """Correlate pooled x_t against pooled x0 at several spatial scales.

    Args:
        ddpm: the diffusion process under test.
        x0: real fields of shape [B, 1, H, W] in model range, land already filled. Use
            B > 1 so the domain-scale (k = H) correlation is taken across fields -- that
            is the sharpest form of the test: does x_T still report which month it came
            from?
        t: timestep to diffuse to (default: the last one).
        scales: pooling block sizes.

    Returns a dict mapping block size -> correlation coefficient. Values near zero mean
    the forward process has genuinely destroyed structure at that scale.
    """
    if t is None:
        t = ddpm.timesteps - 1

    t_batch = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
    xt, _ = ddpm.q_sample(x0, t_batch)

    correlations = {}
    for k in scales:
        if k > min(x0.shape[-2:]):
            continue
        a = F.avg_pool2d(x0, k).flatten()
        b = F.avg_pool2d(xt, k).flatten()
        if a.numel() < 2:
            correlations[k] = float("nan")
            continue
        correlations[k] = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    return correlations


def _load_fields(n: int = 96, device: str = "cpu"):
    """Real CARRA2-WEST fields in model range, land filled the way training fills it."""
    from utils.dataloader import CARRA2

    ds = CARRA2("siconc", device, batch_dim=False)
    # Spread evenly across the whole record so the fields span the seasonal cycle. This
    # matters for the domain-scale (k = H) row: correlation measures discrimination
    # *between* fields, so it is only informative if the fields actually differ in
    # domain-mean ice. Sampling one fixed month per year gives almost no spread and makes
    # that row look clean even under a leaky schedule.
    idx = torch.linspace(0, len(ds) - 1, min(n, len(ds))).round().long().tolist()
    x0 = torch.stack([ds[i] for i in idx])
    return torch.nan_to_num(x0, nan=0.0), ds.finite_mask


if __name__ == "__main__":
    device = "cpu"
    torch.manual_seed(0)
    x0, mask = _load_fields(device=device)
    n = x0.shape[0]
    # Correlation from n fields has a standard error of roughly 1/sqrt(n), and the
    # largest scale is the noisiest because it reduces each field to a single number.
    # With n=24 the k=H row swings by +-0.4 between seeds and is not interpretable.
    print(f"fields: {tuple(x0.shape)}   (corr standard error ~{1 / n ** 0.5:.2f})\n")

    for schedule in ("linear", "shifted_cosine"):
        ddpm = DDPM(timesteps=1000, device=device, land_mask=mask, schedule=schedule)
        print(f"{schedule}  (image_size={ddpm.image_size}, ref={ddpm.shift_ref_resolution})")
        check_invariants(ddpm)

        snr = pooled_snr(ddpm, ddpm.timesteps - 1)
        corr = check_leakage(ddpm, x0)
        # The two columns answer slightly different questions. "analytic SNR" is the
        # pooled signal against the noise in absolute terms -- can x_T tell you the
        # overall ice level at all. The correlation is empirical and relative -- can x_T
        # tell these particular fields apart. Both should be near zero for a schedule
        # that has genuinely destroyed the image.
        print(f"  {'scale k':>9} {'analytic SNR':>14} {'corr(pool(x_T), pool(x0))':>28}")
        for k in SCALES:
            c = corr.get(k)
            c_str = "n/a (single value)" if c is None or c != c else f"{c:+.4f}"
            print(f"  {k:>9} {snr[k]:>14.4f} {c_str:>28}")

        n_destroyed = int(((ddpm.alphas_bar / (1 - ddpm.alphas_bar)).sqrt() * 64 < 1).sum())
        print(f"  steps with 64px-scale SNR < 1: {n_destroyed}/{ddpm.timesteps}\n")
