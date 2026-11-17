"""State-of-the-art overview: one lane per research line, one card per paper, and an optional
"This work" box underneath that all lanes point into.

All text lives in LANES and THIS_WORK below; card heights adapt to the text automatically.
The text is typeset by LaTeX (usetex), so escape & % _ # as \\& \\% \\_ \\#, write en dashes as --,
and use $\\rightarrow$ etc. instead of unicode arrows.

Run from the repo root:  python -m plotting.sota_overview   # -> figures/sota_overview.pdf
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, INK, MUTED, muted

# ---------------------------------------------------------------------------------------------
# Text: edit freely. Each paper: name (bold first line; None to skip), authors (italic second
# line; None to skip), summary (wrapped body text).
# ---------------------------------------------------------------------------------------------
LANES = [
    dict(title="Deep learning for SIC forecasting", color=COLORS[0], papers=[
        dict(name="IceNet",
             authors="Andersson, Hosking, P{\'e}rez-Ortiz et al. (2021)",
             summary="U-Net ensemble forecasting SIC"),
        dict(name="LSTM vs.\\ CNN vs.\\ U-Net",
             authors="Kim et al. (2025)",
             summary="U-Net consistent performance compared to LSTM and CNN on SIC prediction task"),
        dict(name="IceMamba",
             authors="Wang et al. (2025)",
             summary="State-space model for SIC forecasting competing with "
                     "established deep learning baselines."),
    ]),
    dict(title="Diffusion models for sea ice", color=COLORS[1], papers=[
        dict(name="ViT forecast + diffusion downscaling",
             authors="Xu et al. (2024)",
             summary="Vision transformer forecasts SIC at 25 km; a guided diffusion model "
                     "downscales the forecast to 2.5 km."),
        dict(name="Diffusion downscaling of SIT",
             authors="Brajard et al. (2026)",
             summary="Diffusion model that downscales sea-ice thickness fields to a finer grid."),
    ]),
    dict(title="Diffusion for regression (speech)", color=COLORS[2], papers=[
        dict(name="Conditional diffusion for regression",
             authors="Tai et al. (2023)",
             summary="Uses diffusion models for a regression task in speech processing, varying " \
             "the training objective"),
        dict(name="Extending diffusion for regression",
             authors="Liu et al. (2026)",
             summary="Further extends training objectives for diffusion models on speech reconstruction tasks."),
    ]),
]

# Set THIS_WORK = None to drop the bottom box and the arrows.
THIS_WORK = None
# THIS_WORK = dict(
#     name="This work",
#     summary="Conditional DDPM forecasting monthly SIC directly on the 2.5 km CARRA2 grid.",
# )

# ---------------------------------------------------------------------------------------------
# Layout (inches; text is 10pt from plotting/config.py).
# ---------------------------------------------------------------------------------------------
WIDTH = TEXTWIDTH_IN
LANE_GAP = 0.12      # horizontal gap between lanes
CARD_GAP = 0.08      # vertical gap between cards in a lane
LANE_PAD = 0.07      # padding between lane edge and cards
CARD_PAD = 0.06      # padding between card edge and its text
HEADER_PAD = 0.06    # vertical padding around the (wrapped) lane titles
ARROW_H = 0.28       # vertical space taken by the arrows into THIS_WORK
SUMMARY_SIZE = r"\small"   # LaTeX size of the summary text ("" for 10pt)
AUTHOR_SIZE = r"\small"
LANE_FILL = 0.88     # share of white in the lane background
HEADER_FILL = 0.45   # share of white in the lane title band
ROUND = 0.04         # corner radius


def _card_tex(paper: dict, width_in: float, align: str = r"\raggedright") -> str:
    """One LaTeX parbox: bold name, italic authors, summary, wrapped to width_in."""
    lines = []
    if paper.get("name"):
        lines.append(rf"\textbf{{{paper['name']}}}")
    if paper.get("authors"):
        lines.append(rf"{{{AUTHOR_SIZE}\textit{{{paper['authors']}}}}}")
    if paper.get("summary"):
        lines.append(rf"{{{SUMMARY_SIZE} {paper['summary']}}}")
    body = r"\\[1pt]".join(lines)
    return rf"\parbox[t]{{{width_in:.3f}in}}{{{align} {body}}}"


def _box(ax, x, y_top, w, h, face, edge=None, lw=0.0):
    """Rounded box given its top-left corner (y grows downward as negative data y)."""
    ax.add_patch(FancyBboxPatch((x, -y_top - h), w, h, boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                facecolor=face, edgecolor=edge or face, linewidth=lw))


def _text_height(fig, text) -> float:
    """Rendered height of a Text in inches."""
    return text.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi


def plot_sota_overview(out_dir: str = "figures", name: str = "sota_overview"):
    fig = plt.figure()
    # The layout is measured in inches, so the width must be final before text is measured.
    # savefig sets the same size again at the end.
    fig.set_size_inches(WIDTH, 4)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    n = len(LANES)
    lane_w = (WIDTH - (n - 1) * LANE_GAP) / n
    card_w = lane_w - 2 * LANE_PAD
    text_w = card_w - 2 * CARD_PAD

    # Lane titles wrap inside the band; the band is as tall as the tallest title.
    titles = [ax.text(i * (lane_w + LANE_GAP) + lane_w / 2, 0,
                      rf"\parbox{{{text_w:.3f}in}}{{\centering\textbf{{{lane['title']}}}}}",
                      ha="center", va="center", color=INK, zorder=3) for i, lane in enumerate(LANES)]
    HEADER_H = max(_text_height(fig, t) for t in titles) + 2 * HEADER_PAD
    for t in titles:
        t.set_y(-HEADER_H / 2)

    # Pass 1: place the text of every card (y measured downward from the top, in inches).
    lanes = []
    for i, lane in enumerate(LANES):
        x = i * (lane_w + LANE_GAP)
        y = HEADER_H + LANE_PAD
        cards = []
        for paper in lane["papers"]:
            t = ax.text(x + LANE_PAD + CARD_PAD, -(y + CARD_PAD), _card_tex(paper, text_w),
                        ha="left", va="top", color=INK, zorder=3)
            h = _text_height(fig, t) + 2 * CARD_PAD
            cards.append((y, h))
            y += h + CARD_GAP
        lanes.append((x, cards, y - CARD_GAP + LANE_PAD))
    lane_h = max(bottom for _, _, bottom in lanes)

    # Pass 2: lane backgrounds, title bands and card boxes, all lanes the same height.
    for lane, (x, cards, _) in zip(LANES, lanes):
        c = lane["color"]
        _box(ax, x, 0, lane_w, lane_h, muted(c, LANE_FILL))
        _box(ax, x, 0, lane_w, HEADER_H, muted(c, HEADER_FILL))
        for y, h in cards:
            _box(ax, x + LANE_PAD, y, card_w, h, "white", edge=c, lw=0.8)

    total_h = lane_h
    if THIS_WORK is not None:
        y = lane_h + ARROW_H
        t = ax.text(WIDTH / 2, -(y + CARD_PAD), _card_tex(THIS_WORK, WIDTH - 2 * CARD_PAD - 0.2, r"\centering"),
                    ha="center", va="top", color=INK, zorder=3)
        h = _text_height(fig, t) + 2 * CARD_PAD
        _box(ax, 0, y, WIDTH, h, muted(COLORS[4], LANE_FILL), edge=COLORS[4], lw=0.8)
        for x, _, _ in lanes:
            ax.add_patch(FancyArrowPatch((x + lane_w / 2, -lane_h), (x + lane_w / 2, -y),
                                         arrowstyle="-|>", mutation_scale=8, color=MUTED,
                                         linewidth=1.0, shrinkA=1, shrinkB=1))
        total_h = y + h

    ax.set_xlim(0, WIDTH)
    ax.set_ylim(-total_h, 0)

    savefig(fig, out_dir, name, width=WIDTH, height=total_h)
    plt.close(fig)
    print(f"Wrote {out_dir}/{name}.pdf ({WIDTH:.2f} x {total_h:.2f} in)")


if __name__ == "__main__":
    plot_sota_overview()
