"""Forward diffusion q(x_t | x_0) of one CARRA2-WEST SIC field: one row per noise schedule."""
import matplotlib.pyplot as plt
import torch

from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import _clean_image_axis, _show_field
from plotting.schedules import Schedule
from utils.dataloader import CARRA2, to_data_range

SCHEDULES = [
    Schedule("linear"),
    Schedule("cosine"),
    Schedule("shifted_cosine", cosine_shift_res=64),
    Schedule("shifted_cosine", cosine_shift_res=128),
    Schedule("shifted_cosine", cosine_shift_res=256),
]
TIMESTEPS = 1000
T_SHOWN = list(range(100, TIMESTEPS + 1, 100))  # 1-indexed steps; schedule index is t - 1
DATE = "2000-03-15"  # near the annual sea-ice maximum
# Plot every STRIDE-th pixel: keeps the PDF small. Striding (not pooling) keeps per-pixel noise variance.
STRIDE = 8
SEED = 0
ROW_HEIGHT = 0.65  # inches per schedule row


ds = CARRA2("siconc", "cpu", time_slice=slice(DATE, DATE), area=DatasetChoice.CARRA_WEST)
x0 = ds[0]  # [1, 1, H, W] in [-1, 1], land NaN
x0 = torch.nan_to_num(x0, nan=-1.0)  # land as SIC 0, noised like the rest of the field
image_size = max(x0.shape[-2:])

# Same noise for every schedule and timestep, so rows differ only through alpha_bar_t
noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(SEED))


def to_image(x):
    """[1, 1, H, W] model-range field -> strided [h, w] SIC field."""
    return to_data_range(x)[0, 0, ::STRIDE, ::STRIDE].numpy()


n_cols = 1 + len(T_SHOWN)
fig, axes = plt.subplots(
    len(SCHEDULES), n_cols, gridspec_kw={"wspace": 0.04, "hspace": 0.06}, squeeze=False,
)

for row, schedule in zip(axes, SCHEDULES):
    _, _, alphas_bar = schedule.build(TIMESTEPS, image_size)
    alphas_bar = alphas_bar.float()
    _show_field(row[0], to_image(x0))
    for ax, t in zip(row[1:], T_SHOWN):
        # q(x_t | x_0), as DDPM.q_sample
        xt = alphas_bar[t - 1].sqrt() * x0 + (1 - alphas_bar[t - 1]).sqrt() * noise
        _show_field(ax, to_image(xt))
    for ax in row:
        _clean_image_axis(ax)
    row[0].set_ylabel(schedule.name.replace(" ", "\n"))

axes[0, 0].set_title("$x_0$")
for ax, t in zip(axes[0, 1:], T_SHOWN):
    ax.set_title(f"${t}$")

savefig(fig, "figures", "forward_process", width=TEXTWIDTH_IN, height=ROW_HEIGHT * len(SCHEDULES))
