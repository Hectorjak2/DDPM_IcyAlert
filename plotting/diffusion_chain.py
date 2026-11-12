"""Schematic of the DDPM Markov chain (after Ho et al. 2020, Fig. 2), drawn with a real CARRA2-WEST SIC residual.

Top row: forward process q, the residual x_0 noised to x_1, x_t and x_T with the training schedule.
Bottom row: reverse process p_theta, the same panels read right to left. It is a schematic: the
panels are forward samples, not a model trajectory, so no checkpoint is needed.
"""
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import ConnectionPatch

from config import DDPMConfig
from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN, INK, MUTED, REF_LINESTYLE
from plotting.evaluation import _clean_image_axis
from plotting.schedules import Schedule
from utils.dataloader import CARRA2

SCHEDULE = Schedule(DDPMConfig.schedule)  # the schedule the model is trained with
TIMESTEPS = DDPMConfig.timesteps
T_MID = 20  # 1-indexed step shown as x_t; the shifted cosine on 1216^2 is near pure noise by t ~ 100
DATE = "2000-09-15"  # day k
NEXT_DATE = "2001-04-15"  # day k+1; freeze-up gives a clear residual
STRIDE = 8  # plot every STRIDE-th pixel to keep the PDF small (striding keeps the noise variance)
SEED = 0

# (1-indexed step, panel label); step 0 is the clean residual
PANELS = [(0, r"$\mathbf{x}_0$"), (1, r"$\mathbf{x}_1$"), (T_MID, r"$\mathbf{x}_t$"), (TIMESTEPS, r"$\mathbf{x}_T$")]
# Role of the two ends of the chain, under the panel label: (forward row, reverse row)
ROLES = {0: ("data", "generated sample"), len(PANELS) - 1: ("noise", "sampled noise")}
# Arrow labels between consecutive panels; True marks skipped steps, drawn as \cdots
FORWARD = [(r"$q(\mathbf{x}_1 \mid \mathbf{x}_0)$", False),
           (r"$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$", True),
           (r"$q(\mathbf{x}_T \mid \mathbf{x}_{T-1})$", True)]
REVERSE = [(r"$p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$", False),
           (r"$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$", True),
           (r"$p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T)$", True)]

# Each date loaded on its own: a slice spanning both would hold every day in between
k = CARRA2("siconc", "cpu", time_slice=slice(DATE, DATE), area=DatasetChoice.CARRA_WEST)[0]  # [1, 1, H, W], land NaN
kp1 = CARRA2("siconc", "cpu", time_slice=slice(NEXT_DATE, NEXT_DATE), area=DatasetChoice.CARRA_WEST)[0]
# Residual target of ConditionalDDPM in [-2, 2]; land NaN -> 0 ("no change"), as in compute_loss
x0 = torch.nan_to_num(kp1, nan=0.0) - torch.nan_to_num(k, nan=0.0)

_, _, alphas_bar = SCHEDULE.build(TIMESTEPS, max(x0.shape[-2:]))
alphas_bar = alphas_bar.float()
noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(SEED))


def noised(step):
    """q(x_t | x_0) at a 1-indexed step, as DDPM.q_sample; step 0 is x_0 itself."""
    if step == 0:
        return x0
    a = alphas_bar[step - 1]
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def to_image(x):
    """[1, 1, H, W] residual -> strided [h, w] SIC change in [-1, 1]."""
    return (x / 2.0)[0, 0, ::STRIDE, ::STRIDE].numpy()


def draw_arrow(ax, label, skipped, leftward, label_below):
    """Arrow across a spacer axis, with its transition label and \\cdots for skipped steps."""
    ax.axis("off")
    start, end = (0.92, 0.08) if leftward else (0.08, 0.92)
    if skipped:  # dots on the side the chain comes from, arrow over the rest
        dots_x = 0.88 if leftward else 0.12
        ax.text(dots_x, 0.5, r"$\cdots$", ha="center", va="center", transform=ax.transAxes)
        start = 0.72 if leftward else 0.28
    ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(0.5, 0.36 if label_below else 0.64, label, ha="center", va="top" if label_below else "bottom",
            transform=ax.transAxes)


images = [to_image(noised(step)) for step, _ in PANELS]

n = len(PANELS)
fig = plt.figure()
# Columns alternate image / arrow; middle row is the gap holding the dashed links
gs = fig.add_gridspec(3, 2 * n - 1, width_ratios=[1, 1.15] * (n - 1) + [1], height_ratios=[1, 0.45, 1],
                      wspace=0.0, hspace=0.0)
rows = {}
for r, (gs_row, arrows, leftward) in enumerate([(0, FORWARD, False), (2, REVERSE, True)]):
    img_axes = []
    for i, (img, (_, name)) in enumerate(zip(images, PANELS)):
        ax = fig.add_subplot(gs[gs_row, 2 * i])
        ax.imshow(img, cmap="RdBu", vmin=-1.0, vmax=1.0, origin="lower", interpolation="none")
        _clean_image_axis(ax)
        ax.set_aspect("equal")
        role = ROLES.get(i, (None, None))[r]
        # Centred tabular with a small extra gap between the bold x and its caption
        # (usetex ignores matplotlib's linespacing, and a bare \\ breaks the centring).
        # Below the images, x_1 and x_t get an invisible caption row: matplotlib places a
        # one-row and a two-row tabular differently, so the four x's would not line up.
        if role is not None:
            rows_tex = rf"{name}\\[1pt]({role})"
        elif gs_row == 2:
            rows_tex = rf"{name}\\[1pt]\phantom{{(x)}}"
        else:
            rows_tex = name
        label = rf"\setlength{{\tabcolsep}}{{0pt}}\begin{{tabular}}{{c}}{rows_tex}\end{{tabular}}"
        if gs_row == 0:
            ax.set_title(label, pad=3)
        else:
            ax.set_xlabel(label, labelpad=3)
        img_axes.append(ax)
    arrow_axes = []
    for i, (label, skipped) in enumerate(arrows):
        ax = fig.add_subplot(gs[gs_row, 2 * i + 1])
        draw_arrow(ax, label, skipped, leftward, label_below=leftward)
        arrow_axes.append(ax)
    rows[gs_row] = (img_axes, arrow_axes)

top_imgs, top_arrows = rows[0]
bottom_imgs, bottom_arrows = rows[2]

# Captions on the outer side of each row, over the middle arrow
top_arrows[1].text(0.5, 1.25, r"\textbf{Forward process} (adding noise)", ha="center", va="bottom",
                   transform=top_arrows[1].transAxes)
bottom_arrows[1].text(0.5, -0.36, r"\textbf{Reverse process} (denoising with $p_\theta$)", ha="center", va="top",
                      transform=bottom_arrows[1].transAxes)

# Dashed links between the two ends of the chain; drawn at render time, so they follow the final layout
for top, bottom in [(top_imgs[0], bottom_imgs[0]), (top_imgs[-1], bottom_imgs[-1])]:
    fig.add_artist(ConnectionPatch(xyA=(0.5, -0.05), coordsA=top.transAxes, xyB=(0.5, 1.05), coordsB=bottom.transAxes,
                                   color=MUTED, lw=0.8, linestyle=REF_LINESTYLE))

savefig(fig, "figures", "diffusion_chain", width=TEXTWIDTH_IN, height=2.9)
