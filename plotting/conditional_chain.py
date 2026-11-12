"""Graphical model of the conditional reverse chain, with the conditioning c drawn as real CARRA2-WEST SIC fields.

The reverse chain x_T -> ... -> x_0 runs right to left; every step is conditioned on c, the SIC fields of
days k and k-1 (ConditionalDDPM.build_context). The forward process is left out on purpose: it is fixed
noising and does not see c, so only the conditioned reverse direction is drawn.
"""
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Circle, FancyBboxPatch

from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN, INK, REF_LINESTYLE
from plotting.evaluation import SIC_CMAP
from utils.dataloader import CARRA2

DATE = "2000-10-15"  # day k
PREV_DATE = "2000-10-14"  # day k-1
STRIDE = 8  # plot every STRIDE-th pixel to keep the PDF small

# Diagram colours: conditioning in blue, reverse chain in brown
COND = "#4A5DB8"
COND_FILL = "#B8C0E4"
CHAIN = "#9A4A1E"
CHAIN_FILL = "#F7EEE7"

# Layout in data units (1 unit ~ 0.44 in at full width)
Y_CHAIN, R = 1.2, 0.42  # node row and node radius
Y_COND = 3.4  # dashed conditioning line
Y_REV = -0.2  # bottom direction arrow
NODES = {r"$\mathbf{x}_0$": 0.9, r"$\mathbf{x}_t$": 7.0, r"$\mathbf{x}_T$": 13.1}
# Reverse-step arrows (start, end, label centre, label up to the c); \cdots sits in the gaps between them
STEPS = [
    (12.68, 10.55, 11.5, r"p_\theta(\mathbf{x}_{T-1} \mid \mathbf{x}_T,"),
    (9.75, 7.42, 8.75, r"p_\theta(\mathbf{x}_t \mid \mathbf{x}_{t+1},"),
    (6.58, 4.4, 5.35, r"p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t,"),
    (3.6, 1.32, 2.5, r"p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1,"),
]
DOTS_X = [10.15, 4.0]


def sic_image(date):
    """SIC of one day as a strided [h, w] array in [0, 1], land NaN (grey in SIC_CMAP)."""
    x = CARRA2("siconc", "cpu", time_slice=slice(date, date), area=DatasetChoice.CARRA_WEST)[0]  # [1, 1, H, W] in [-1, 1]
    return ((x + 1.0) / 2.0)[0, 0, ::STRIDE, ::STRIDE].numpy()


def step_label(ax, x, y, head):
    """'p_theta(... | ..., c)' with only c in the conditioning colour.

    usetex drops \\textcolor, so the label is three text pieces packed on a common baseline.
    Each piece is cropped to its ink, so the space after the comma is an explicit 2pt gap.
    """
    head, c, close = (TextArea(text, textprops=dict(color=color))
                      for text, color in [(rf"${head}$", INK), (r"$\mathbf{c}$", COND), (r"$)$", INK)])
    packed = HPacker(children=[head, DrawingArea(2.0, 0.0), c, close], align="baseline", pad=0, sep=0)
    ax.add_artist(AnnotationBbox(packed, (x, y), xycoords="data", box_alignment=(0.5, 1.0), frameon=False, pad=0))


def arrow(ax, x0, y0, x1, y1, color, lw=1.0, head=9):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="data",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0,
                                mutation_scale=head))


fig, ax = plt.subplots()
ax.set_axis_off()

# Conditioning c: day k-1 behind day k, on a rounded blue card as in the node style
ax.add_patch(FancyBboxPatch((0.15, 2.55), 1.55, 1.55, boxstyle="round,pad=0.08,rounding_size=0.12",
                            facecolor=COND_FILL, edgecolor="none", zorder=1))
# Each layer's frame sits between its own image and the next one, so the back frame is hidden by day k
for layer, (date, (x, y)) in enumerate([(PREV_DATE, (0.4, 2.8)), (DATE, (0.15, 2.55))]):
    ax.imshow(sic_image(date), cmap=SIC_CMAP, vmin=0.0, vmax=1.0, origin="lower", interpolation="none",
              extent=(x, x + 1.3, y, y + 1.3), zorder=2 + 2 * layer)
    ax.add_patch(plt.Rectangle((x, y), 1.3, 1.3, fill=False, edgecolor="white", lw=0.8, zorder=3 + 2 * layer))
ax.text(0.93, 4.3, r"$\mathbf{c}$", ha="center", va="bottom", color=COND)

# Dashed conditioning line with a drop onto every reverse step
drops = [(start + end) / 2 for start, end, _, _ in STEPS]
ax.plot([1.85, max(drops)], [Y_COND, Y_COND], color=COND, lw=1.0, linestyle=REF_LINESTYLE)
ax.text((1.85 + max(drops)) / 2, Y_COND + 0.12, r"Conditioning for predicting $\mathbf{x}_0$",
        ha="center", va="bottom", color=COND)
for x in drops:
    ax.plot([x, x], [Y_COND, Y_CHAIN + 0.2], color=COND, lw=1.0, linestyle=REF_LINESTYLE, zorder=1)
    arrow(ax, x, Y_CHAIN + 0.22, x, Y_CHAIN + 0.02, COND, head=7)  # solid head: a dashed one looks broken

# Reverse chain
for name, x in NODES.items():
    ax.add_patch(Circle((x, Y_CHAIN), R, facecolor=CHAIN_FILL, edgecolor=CHAIN, lw=0.8))
    ax.text(x, Y_CHAIN, name, ha="center", va="center", color=CHAIN)
for start, end, label_x, head in STEPS:
    arrow(ax, start, Y_CHAIN, end, Y_CHAIN, CHAIN)
    step_label(ax, label_x, Y_CHAIN - 0.15, head)
for x in DOTS_X:
    ax.text(x, Y_CHAIN, r"$\cdots$", ha="center", va="center", color=CHAIN)

# Direction of the reverse process: noise removed right to left
left, right = NODES[r"$\mathbf{x}_0$"], NODES[r"$\mathbf{x}_T$"]
label_box = dict(facecolor="white", edgecolor="none", pad=2)
arrow(ax, right, Y_REV, left, Y_REV, CHAIN)
ax.text((left + right) / 2, Y_REV, r"Reverse process $p_\theta$ (denoising)", ha="center", va="center",
        color=CHAIN, bbox=label_box)

ax.set_xlim(0.0, 13.7)
ax.set_ylim(-0.45, 4.7)
ax.set_aspect("equal")

savefig(fig, "figures", "conditional_chain", width=TEXTWIDTH_IN, height=2.4)
