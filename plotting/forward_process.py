"""Forward diffusion q(x_t | x_0) of one CARRA2-WEST SIC field: one row per noise schedule."""
import matplotlib.pyplot as plt
import torch

from config import DDPMConfig
from models.ddpm import DDPM
from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import _clean_image_axis, _show_field
from plotting.plot_schedule import SCHEDULE_LABELS
from utils.dataloader import CARRA2, to_data_range

DATE = "2000-03-15"  # near the annual sea-ice maximum
T_SHOWN = list(range(100, DDPMConfig.timesteps + 1, 100))  # 1-indexed steps; DDPM index is t - 1
# Plot every STRIDE-th pixel: keeps the PDF small. Striding (not pooling) keeps per-pixel noise variance.
STRIDE = 8
SEED = 0

ds = CARRA2("siconc", "cpu", time_slice=slice(DATE, DATE), area=DatasetChoice.CARRA_WEST)
x0 = ds[0]  # [1, 1, H, W] in [-1, 1], land NaN
x0 = torch.nan_to_num(x0, nan=-1.0)  # land as SIC 0, noised like the rest of the field

# Same noise for every schedule and timestep, so rows differ only through alpha_bar_t
noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(SEED))


def to_image(x):
    """[1, 1, H, W] model-range field -> strided [h, w] SIC field."""
    return to_data_range(x)[0, 0, ::STRIDE, ::STRIDE].numpy()


n_cols = 1 + len(T_SHOWN)
fig, axes = plt.subplots(
    len(DDPM.SCHEDULES), n_cols, gridspec_kw={"wspace": 0.04, "hspace": 0.06},
)

for row, schedule in zip(axes, DDPM.SCHEDULES):
    ddpm = DDPM(timesteps=DDPMConfig.timesteps, device="cpu",
                image_size=x0.shape[-1], schedule=schedule)
    _show_field(row[0], to_image(x0))
    for ax, t in zip(row[1:], T_SHOWN):
        xt = ddpm.q_sample(x0, torch.tensor([t - 1]), noise)
        _show_field(ax, to_image(xt))
    for ax in row:
        _clean_image_axis(ax)
    row[0].set_ylabel(SCHEDULE_LABELS[schedule].replace(" ", "\n"))

axes[0, 0].set_title("$x_0$")
for ax, t in zip(axes[0, 1:], T_SHOWN):
    ax.set_title(f"${t}$")

savefig(fig, "figures", "forward_process", width=TEXTWIDTH_IN, height=2.6)
