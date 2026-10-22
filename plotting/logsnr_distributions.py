import math
import torch
import matplotlib.pyplot as plt
from config import DDPMConfig
from plotting.config import savefig, TEXTWIDTH_IN, COLORS
from plotting.schedule_comparison import schedule_label

mu = DDPMConfig().laplace_mu
b = DDPMConfig().laplace_b
s = DDPMConfig().cosine_s
shift_ref = DDPMConfig().cosine_shift_res
image_size = 1216

delta = 2 * math.log(shift_ref / image_size)

lambda_vals = torch.linspace(-15, 10, 1000, dtype=torch.float64)

p_linear = torch.exp(lambda_vals) / (1 + torch.exp(lambda_vals))**2
p_laplace = (1 / (2 * b)) * torch.exp(-torch.abs(lambda_vals - mu) / b)


def cosine_logsnr_density(s: float):
    """log-SNR lambda(t) and its density p(lambda) = 1 / |d lambda / dt| for t ~ U(0, 1).

    Uses alpha_bar(t) = f(t) / f(0), f(t) = cos^2((t + s) / (1 + s) * pi / 2), as in models/ddpm.py.
    For s = 0 this is 1 / (2 pi cosh(lambda / 2)); s > 0 thins the high-SNR tail into exp(-lambda).
    """
    # Log-spaced near t = 0, where lambda -> inf, so the tail is resolved up to the plot edge
    t = torch.cat([torch.logspace(-14, -2, 2000, dtype=torch.float64),
                   torch.linspace(1e-2, 1 - 1e-6, 20000, dtype=torch.float64)[1:]])
    du_dt = math.pi / (2 * (1 + s))
    u = (t + s) * du_dt
    f0 = math.cos(s * du_dt) ** 2
    alphas_bar = torch.cos(u) ** 2 / f0
    log_snr = torch.log(alphas_bar / (1 - alphas_bar))
    dalphas_bar_dt = -torch.sin(2 * u) * du_dt / f0
    dlog_snr_dt = dalphas_bar_dt / (alphas_bar * (1 - alphas_bar))
    return log_snr, 1 / dlog_snr_dt.abs()


lambda_cosine, p_cosine = cosine_logsnr_density(s)

# (lambda, p(lambda)) per schedule; the shift moves the cosine density by delta without changing its shape
densities = {
    schedule_label("cosine", image_size): (lambda_cosine, p_cosine),
    schedule_label("shifted_cosine", image_size): (lambda_cosine + delta, p_cosine),
    schedule_label("laplace", image_size): (lambda_vals, p_laplace),
}

fig, ax = plt.subplots()
for (label, (lam, p)), color in zip(densities.items(), COLORS):
    ax.plot(lam, p, color=color, label=label)

ax.set_xlabel(r"log-SNR $\lambda$")
ax.set_ylabel(r"$p(\lambda)$")
ax.set_xlim(lambda_vals[0].item(), lambda_vals[-1].item())
ax.set_ylim(bottom=0)
ax.legend(loc="upper left")

savefig(fig, "figures", "logsnr_distributions", width=TEXTWIDTH_IN, height=2.8)
