"""Forward diffusion q(x_t | x_0) of one CARRA2-WEST SIC residual (k+1 - k, a month apart): one row per noise schedule."""
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import _clean_image_axis
from plotting.schedules import Schedule
from utils.dataloader import CARRA2

SCHEDULES = [
    Schedule("linear"),
    Schedule("cosine"),
    Schedule("laplace", laplace_mu=0),
    Schedule("shifted_cosine", cosine_shift_res=64),
    Schedule("shifted_cosine", cosine_shift_res=128),
]

TIMESTEPS = 1000
T_SHOWN = list(range(100, TIMESTEPS + 1, 100))  # 1-indexed steps; schedule index is t - 1
DATE = "2000-03-15"  # near the annual sea-ice maximum
MONTH_APART = "2000-04-15"
# Plot every STRIDE-th pixel: keeps the PDF small. Striding (not pooling) keeps per-pixel noise variance.
STRIDE = 8
SEED = 0
ROW_HEIGHT = 0.65  # inches per schedule row


# Load each date on its own: a slice spanning both would hold every timestep in between.
k = CARRA2("siconc", "cpu", time_slice=slice(DATE, DATE), area=DatasetChoice.CARRA_WEST)[0]  # [1, 1, H, W] in [-1, 1], land NaN
kp1 = CARRA2("siconc", "cpu", time_slice=slice(MONTH_APART, MONTH_APART), area=DatasetChoice.CARRA_WEST)[0]
# Residual target of ConditionalDDPM, in [-2, 2]. Land NaN -> 0 in both, so land is
# "no change", as in compute_loss, and is noised like the rest of the field.
x0 = torch.nan_to_num(kp1, nan=0.0) - torch.nan_to_num(k, nan=0.0)

image_size = max(k.shape[-2:])

# Same noise for every schedule and timestep, so rows differ only through alpha_bar_t
noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(SEED))


def to_image(x):
    """[1, 1, H, W] model-range residual -> strided [h, w] SIC change in [-1, 1]."""
    return (x / 2.0)[0, 0, ::STRIDE, ::STRIDE].numpy()


def _show_field(ax, field):
    """Diverging map centred on 0 (no change); noisy steps outside [-1, 1] saturate."""
    return ax.imshow(field, cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower",
                     interpolation="none")


n_cols = 1 + len(T_SHOWN)
fig = plt.figure()
# Extra narrow column for the shared colorbar, as in evaluation.plot_forecast_grid
gs = fig.add_gridspec(
    len(SCHEDULES), n_cols + 1, width_ratios=[1] * n_cols + [0.06], wspace=0.04, hspace=0.06,
)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(len(SCHEDULES))])

for row, schedule in zip(axes, SCHEDULES):
    _, _, alphas_bar = schedule.build(TIMESTEPS, image_size)
    alphas_bar = alphas_bar.float()
    _show_field(row[0], to_image(x0))
    for ax, t in zip(row[1:], T_SHOWN):
        # q(x_t | x_0), as DDPM.q_sample
        xt = alphas_bar[t - 1].sqrt() * x0 + (1 - alphas_bar[t - 1]).sqrt() * noise
        img = _show_field(ax, to_image(xt))
    for ax in row:
        _clean_image_axis(ax)
    row[0].set_ylabel(schedule.name.replace(" ", "\n"))

axes[0, 0].set_title("$x_0$")
for ax, t in zip(axes[0, 1:], T_SHOWN):
    ax.set_title(f"${t}$")

# extend="both": noisy steps go beyond [-1, 1] and are shown saturated
cbar = fig.colorbar(img, cax=fig.add_subplot(gs[:, n_cols]), extend="both", ticks=[-1, 0, 1])
cbar.set_label("SIC change", labelpad=4)

savefig(fig, "figures", "forward_process", width=TEXTWIDTH_IN, height=ROW_HEIGHT * len(SCHEDULES))
