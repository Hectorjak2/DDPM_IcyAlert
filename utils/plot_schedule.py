import matplotlib.pyplot as plt
import torch

from utils import CARRA2
from models.ddpm import DDPM
from config import DDPMConfig


def plot_schedule(betas, alphas, alphas_bar):
    """Plot betas, alphas, and alphas_bar against the diffusion timestep."""
    betas = betas.cpu()
    alphas = alphas.cpu()
    alphas_bar = alphas_bar.cpu()
    t = torch.arange(len(betas))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(t, betas)
    axes[0].set_title(r"$\beta_t$")
    axes[0].set_xlabel("timestep")

    axes[1].plot(t, alphas)
    axes[1].set_title(r"$\alpha_t$")
    axes[1].set_xlabel("timestep")

    axes[2].plot(t, alphas_bar)
    axes[2].set_title(r"$\bar{\alpha}_t$")
    axes[2].set_xlabel("timestep")

    plt.tight_layout()
    plt.show()


def plot_schedule_comparison(schedules: dict):
    """Overlay multiple named schedules' alphas_bar on one plot."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for name, (_, _, alphas_bar) in schedules.items():
        alphas_bar = alphas_bar.cpu()
        t = torch.arange(len(alphas_bar))
        ax.plot(t, alphas_bar, label=name)

    ax.set_title(r"$\bar{\alpha}_t$")
    ax.set_xlabel("timestep")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = "mps"
    timesteps = 1000
    train_ds = CARRA2("siconc", device, WEST=True, batch_dim=False)

    schedules = {}
    for name in ["linear", "shifted_cosine"]:
        ddpm = DDPM(
            timesteps=timesteps,
            device=device,
            image_size=train_ds[0][0].shape[-1],
            land_mask=train_ds.finite_mask,
            schedule=name,
            shift_ref_resolution=DDPMConfig.shift_ref_resolution,
        )
        schedules[name] = ddpm.build_schedule(timesteps)

    plot_schedule_comparison(schedules)
