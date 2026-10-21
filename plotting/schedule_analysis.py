from utils.dataloader import CARRA2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MARKERS, MUTED, REF_LINESTYLE

from models.models import DatasetChoice
from models.ddpm import DDPM
from config import DDPMConfig

dataset = DatasetChoice.CARRA_WEST
train_ds = CARRA2("siconc", "mps", area=dataset, batch_dim=False)
selected_vars = torch.linspace(0, 100, 60, dtype=torch.int)

x0 = train_ds[selected_vars]
print(x0.shape)
x0 = torch.nan_to_num(x0, nan=0.0)

scales = [64, 128, 256, 512]
timesteps = 1000
t_values = torch.linspace(0, timesteps-1, 30, dtype=torch.int)

fig, axes = plt.subplots(1, 2, sharey=True)

water = train_ds.finite_mask  # [1, H, W], 1=water, 0=land
def masked_pool(x, k):
    pooled = F.avg_pool2d(x * water, k)
    frac = F.avg_pool2d(water.expand_as(x), k)
    return (pooled / frac.clamp(min=1e-6))[frac > 0.5].flatten()

for ax, schedule in zip(axes, ["linear", "shifted_cosine"]):
    ddpm = DDPM(
        timesteps=timesteps,
        device="mps",
        image_size=train_ds[0][0].shape[-1],
        land_mask=train_ds.finite_mask,
        schedule=schedule,
    )

    for k, color, marker in zip(scales, COLORS, MARKERS):
        a = masked_pool(x0, k)
        corr_per_t = []
        for t in t_values:
            xt = ddpm.q_sample(x0=x0, t=int(t))[0].reshape(x0.shape)
            b = masked_pool(xt, k)
            corr_per_t.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
        ax.plot(t_values, corr_per_t, color=color, marker=marker, label=f"k={k}")

    ax.set_title(schedule.replace("_", " ").capitalize())
    ax.set_xlabel("timestep t")
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)
    ax.legend()

axes[0].set_ylabel(r"corr($\mathrm{pool}(x_t)$, $\mathrm{pool}(x_0)$)")

savefig(fig, "figures", "schedule_analysis", width=TEXTWIDTH_IN, height=2.8)
