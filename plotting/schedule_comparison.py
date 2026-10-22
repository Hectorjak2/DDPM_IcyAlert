import math

import matplotlib.pyplot as plt
import torch

from config import DDPMConfig
from models.ddpm import DDPM
from plotting.config import savefig, TEXTWIDTH_IN

# Readable, LaTeX-safe labels for DDPM.SCHEDULES
SCHEDULE_LABELS = {"linear": "Linear", "cosine": "Cosine", "laplace": "Laplace", "shifted_cosine": "Shifted cosine"}
# Reference resolution of the shifted cosine, matching models/ddpm.py


def schedule_label(schedule: str, image_size: int) -> str:
    """Legend label of a schedule, with the DDPMConfig parameters it is built from."""
    c = DDPMConfig
    if schedule == "linear":
        params = rf"$\beta_1 = {c.beta_start:g}$, $\beta_T = {c.beta_end:g}$"
    elif schedule == "cosine":
        params = rf"$s = {c.cosine_s:g}$"
    elif schedule == "shifted_cosine":
        delta = 2 * math.log(c.cosine_shift_res / image_size)
        params = rf"$s = {c.cosine_s:g}$, $\delta = 2\log({c.cosine_shift_res}/{image_size}) \approx {delta:.2f}$"
    elif schedule == "laplace":
        params = rf"$\mu = {c.laplace_mu:g}$, $b = {c.laplace_b:g}$"
    return f"{SCHEDULE_LABELS[schedule]} ({params})"


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
    # The parameter labels are too long for one panel, so the legend spans both above them
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))

    savefig(fig, output_dir, name, width=TEXTWIDTH_IN, height=2.8)


if __name__ == "__main__":
    from models.models import DatasetChoice
    from utils.dataloader import CARRA2

    timesteps = 1000

    # The shifted cosine depends on the resolution, so use the CARRA2-WEST size we train on
    # (DDPM takes it from the land mask during training). Linear, cosine and Laplace ignore it.
    image_size = max(CARRA2("siconc", "cpu", time_slice=slice("2000-03-15", "2000-03-15"),
                            area=DatasetChoice.CARRA_WEST)[0].shape[-2:])

    schedules = {}
    for schedule in DDPM.SCHEDULES:
        ddpm = DDPM(timesteps=timesteps, device="cpu", image_size=image_size, schedule=schedule)
        schedules[schedule_label(schedule, image_size)] = ddpm.build_schedule(timesteps)

    plot_schedule_comparison(schedules)
