from matplotlib import pyplot as plt
from cycler import cycler
import os

# LaTeX \textwidth of the thesis (448.13pt, 72.27pt per inch)
TEXTWIDTH_IN = 448.13 / 72.27
# Thesis body font size (10pt)
FONTSIZE = 10

# Palette: teal, amber, steel blue, coral, charcoal, light gray, periwinkle, plum
# (also the default colour cycle; ordered so the first few series stay well separated)
COLORS = ["#00AB86", "#FF8D00", "#F86748","#4E7693", "#B7B7B7", "#69747D", "#8A99B0", "#9C5B9E"]
# Per-series markers so curves stay distinguishable in grayscale / for colour-blind readers
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Soft neutrals
INK = "#3A3F47"     # text, ticks
GRID = "#E3E6EA"    # grid lines
MUTED = "#9AA1AB"   # spines, reference lines (use with linestyle=REF_LINESTYLE)
REF_LINESTYLE = (0, (4, 3))

def savefig(fig, output_dir: str, name: str, width: float = TEXTWIDTH_IN, height: float = 4):
    fig.set_size_inches(width, height)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches="tight")

rcParams = {
    "font.family": "serif",
    "text.usetex": True,
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "axes.labelpad": 1,
    "axes.axisbelow": True,
    # colours
    "axes.prop_cycle": cycler(color=COLORS),
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": MUTED,
    # frame and grid
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    # lines and legend
    "lines.linewidth": 1.6,
    "lines.markersize": 3,
    "lines.markeredgewidth": 0,
    "legend.frameon": False,
    "legend.labelcolor": INK,
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join(
        [
            r"\usepackage{url}",
            r"\usepackage{amsmath,amssymb}",
        ]
    ),
}
plt.rcParams.update(rcParams)