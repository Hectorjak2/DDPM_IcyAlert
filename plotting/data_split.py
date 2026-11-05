"""Chronological train / validation / test split as one bar, read from the slices in config.py.

The white gaps between segments are the two months dropped at each boundary, so that no
(k-1, k, k+1) window spans two sets. Training and test are drawn shorter than their true
length so the short validation segment has room for its label, so the bar is NOT to scale:
say so in the caption.

Run from the repo root:  python -m plotting.data_split
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

from config import TEST_SLICE, TRAINING_SLICE, VALIDATION_SLICE
from plotting.config import savefig, TEXTWIDTH_IN, COLORS, INK

MUTE = 0.45  # share of white blended into each palette colour
# Drawn length of each segment relative to its true length.
TRAINING_SCALE = 0.5
TEST_SCALE = 0.6


def _muted(color: str) -> tuple:
    return tuple((1 - MUTE) * np.array(to_rgb(color)) + MUTE)


SPLITS = [("Training", TRAINING_SLICE, _muted(COLORS[0]), TRAINING_SCALE),
          ("Validation", VALIDATION_SLICE, _muted(COLORS[3]), 1.0),
          ("Test", TEST_SLICE, _muted(COLORS[1]), TEST_SCALE)]


def _year(date: str, end: bool = False) -> float:
    """'2016-10-31' -> decimal year; an end date counts through the end of its day."""
    ts = pd.Timestamp(date) + pd.Timedelta(days=int(end))
    start = pd.Timestamp(year=ts.year, month=1, day=1)
    return ts.year + (ts - start).days / (366 if ts.is_leap_year else 365)


def plot_data_split(out_dir: str = "figures"):
    fig, ax = plt.subplots()
    x = 0.0
    for i, (name, s, color, scale) in enumerate(SPLITS):
        start, stop = _year(s.start), _year(s.stop, end=True)
        if i > 0:  # keep the real gap to the previous set
            x += start - _year(SPLITS[i - 1][1].stop, end=True)
        width = (stop - start) * scale
        ax.barh(0, width, left=x, height=1.0, color=color, linewidth=0)
        ax.text(x + width / 2, 0, name, ha="center", va="center", color=INK)
        ax.text(x + width / 2, -0.75, f"{pd.Timestamp(s.start).year}--{pd.Timestamp(s.stop).year}",
                ha="center", va="top", color=INK)
        x += width
    ax.set_ylim(-1.6, 0.5)
    ax.set_xlim(0, x)
    ax.axis("off")

    savefig(fig, out_dir, "data_split", width=TEXTWIDTH_IN, height=0.75)
    plt.close(fig)
    print(f"Wrote {out_dir}/data_split.pdf")


if __name__ == "__main__":
    plot_data_split()
