"""One forecast case side by side: true SIC, the runs in ``RUNS`` and climatology, in one row.

Edit ``TARGET`` (year, month of the k+1 target) and ``RUNS`` below, then run from the repo root:

    python -m plotting.case_comparison

Picks the sample whose lead-1 target falls in ``TARGET`` (the first one if there are
several, see ``DAY``), checks every run has that same target date, and writes
``<OUT_DIR>/case_comparison_<YYYY>_<MM>.pdf``. Prints the RMSE over water of every panel.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import _clean_image_axis, _load_climatology, _show_field, load_leads

TARGET = (2025, 10)  # (year, month) of the k+1 target
DAY = None           # day of month, if TARGET matches several samples; None = first match
# (folder under results/, panel title), in column order after the truth.
RUNS = [
    ("ddpm-corrected-land-256", "DDPM Shift (256)"),
    ("ensembles/member_5", "U-Net"),
]
OUT_DIR = "final_figures"


def _find(data):
    """Index of the sample whose target date matches TARGET (and DAY)."""
    year, month = TARGET
    for i, (target_date, _, _) in enumerate(data["times"]):
        t = pd.Timestamp(target_date)
        if t.year == year and t.month == month and (DAY is None or t.day == DAY):
            return i
    raise ValueError(f"No sample with target in {year}-{month:02d}"
                     + (f"-{DAY:02d}" if DAY else ""))


def _rmse(pred, target) -> float:
    water = np.isfinite(target) & np.isfinite(pred)
    return float(np.sqrt(np.mean((pred[water] - target[water]) ** 2)))


def main():
    panels = []  # (title, field)
    target = target_date = None
    for run, title in RUNS:
        _, leads = load_leads(run)
        data = leads[0]
        del leads
        i = _find(data)
        date = pd.Timestamp(data["times"][i][0])
        if target is None:
            target, target_date = data["targets"][i], date
            panels.append(("Truth", target))
        elif date != target_date:
            raise ValueError(f"{run}: target {date.date()} differs from {target_date.date()}")
        panels.append((title, data["predictions"][i]))
        print(f"{run}: sample {i}, target {date.date()}, "
              f"context k = {pd.Timestamp(data['times'][i][1]).date()}")

    climatology = _load_climatology(target.shape)
    if climatology is None:
        raise ValueError("No climatology fitted for this grid.")
    panels.append(("Climatology", climatology.predict(target_date.month)))

    print(f"\nRMSE over water, target {target_date.date()}:")
    for title, field in panels[1:]:
        print(f"  {title:<14} {_rmse(field, target):.4f}")

    n = len(panels)
    fig, axes = plt.subplots(1, n + 1, gridspec_kw={"width_ratios": [1] * n + [0.06]})
    for ax, (title, field) in zip(axes[:n], panels):
        img = _show_field(ax, field)
        ax.set_title(title)
        _clean_image_axis(ax)
    fig.colorbar(img, cax=axes[n]).set_label("SIC", labelpad=4)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"case_comparison_{TARGET[0]}_{TARGET[1]:02d}"
    savefig(fig, out_dir, name, width=TEXTWIDTH_IN, height=1.9)
    plt.close(fig)
    print(f"Wrote {out_dir / name}.pdf")


if __name__ == "__main__":
    main()
