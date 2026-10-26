import matplotlib.pyplot as plt
import torch

from plotting.config import savefig, TEXTWIDTH_IN
from plotting.schedules import Schedule
from config import DDPMConfig
ddpm_conf = DDPMConfig()

SCHEDULES = [
    Schedule("linear", beta_start=ddpm_conf.beta_start, beta_end=ddpm_conf.beta_end),
    Schedule("cosine"),
    Schedule("laplace", laplace_mu=0),
    Schedule("shifted_cosine", cosine_shift_res=64),
    Schedule("shifted_cosine", cosine_shift_res=32),
]

TIMESTEPS = 1000
IMAGE_SIZE = 1216  # CARRA2-WEST; sets the shifted cosine's delta, the others ignore it

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
    plot_schedule_comparison({s.label(IMAGE_SIZE): s.build(TIMESTEPS, IMAGE_SIZE) for s in SCHEDULES})
