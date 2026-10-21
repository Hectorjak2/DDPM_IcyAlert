import matplotlib.pyplot as plt
import torch

from models.ddpm import DDPM
from plotting.config import savefig, TEXTWIDTH_IN

# Readable, LaTeX-safe labels for DDPM.SCHEDULES
SCHEDULE_LABELS = {"linear": "Linear", "cosine": "Cosine", "laplace": "Laplace", "shifted_cosine": "Shifted cosine"}


def plot_schedule(betas, alphas, alphas_bar, name: str = "schedule", output_dir: str = "figures"):
    """Plot betas, alphas, and alphas_bar of one schedule against the diffusion timestep."""
    t = torch.arange(len(betas))

    fig, axes = plt.subplots(1, 3)
    for ax, values, title in zip(
        axes,
        (betas, alphas, alphas_bar),
        (r"$\beta_t$", r"$\alpha_t$", r"$\bar{\alpha}_t$"),
    ):
        ax.plot(t, values.cpu())
        ax.set_title(title)
        ax.set_xlabel("timestep $t$")

    savefig(fig, output_dir, name, width=TEXTWIDTH_IN, height=2.4)


def plot_schedule_comparison(schedules: dict, name: str = "schedule_comparison", output_dir: str = "figures"):
    """Overlay several named schedules: alphas_bar and log-SNR = log(alphas_bar / (1 - alphas_bar))."""
    fig, axes = plt.subplots(1, 2, sharex=True)

    for label, (_, _, alphas_bar) in schedules.items():
        alphas_bar = alphas_bar.cpu()
        t = torch.arange(len(alphas_bar))
        axes[0].plot(t, alphas_bar, label=label)
        axes[1].plot(t, torch.log(alphas_bar / (1.0 - alphas_bar)), label=label)

    axes[0].set_ylabel(r"$\bar{\alpha}_t$")
    axes[1].set_ylabel(r"$\log \mathrm{SNR}(t)$")
    # alphas_bar is clamped to [1e-12, 1 - 1e-12], so the endpoints spike to about +-27; clip to keep the shape readable
    axes[1].set_ylim(-12, 12)
    for ax in axes:
        ax.set_xlabel("timestep $t$")
    axes[0].legend()

    savefig(fig, output_dir, name, width=TEXTWIDTH_IN, height=2.8)


if __name__ == "__main__":
    timesteps = 1000

    # The schedule depends only on DDPMConfig and the number of timesteps, so no data is needed.
    schedules = {}
    for schedule in DDPM.SCHEDULES:
        ddpm = DDPM(timesteps=timesteps, device="cpu", schedule=schedule)
        schedules[SCHEDULE_LABELS[schedule]] = ddpm.build_schedule(timesteps)

    plot_schedule_comparison(schedules)
