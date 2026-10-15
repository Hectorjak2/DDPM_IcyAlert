---
name: thesis-plot
description: "Create thesis-quality matplotlib figures for the LaTeX report using plotting/plot_config.py (10pt serif, usetex, exact \\textwidth, vector PDF). Use when the user asks to plot/visualize something for the thesis or says 'use thesis-plot'."
---

# Thesis-quality plots

Figures for the thesis must match the LaTeX document (memoir, A4, 10pt, `\textwidth` = 448.13pt ≈ 6.20 in). All styling and saving goes through [plotting/plot_config.py](../../../plotting/plot_config.py). Reference implementation: [plotting/schedule_analysis.py](../../../plotting/schedule_analysis.py).

## Rules

1. **Import from plot_config.** Importing it applies the rcParams (serif, `usetex`, 10pt) as a side effect:
   ```python
   from plotting.plot_config import savefig, TEXTWIDTH_IN
   ```
2. **Do not set `figsize`, `dpi` or font sizes** on `plt.subplots(...)` or elsewhere. `savefig` sets the size.
3. **Save only with `savefig(fig, output_dir, name, width=..., height=...)`.** It writes `<output_dir>/<name>.pdf` (vector). No `plt.show()`, no PNG, no direct `fig.savefig`, no extra `fig.tight_layout()` (savefig calls it after setting the size).
4. **Width is relative to `TEXTWIDTH_IN`**, so the PDF is included in LaTeX at native size (font stays 10pt):
   - full width: `width=TEXTWIDTH_IN`
   - half width: `width=TEXTWIDTH_IN / 2`
   - Choose `height` for the aspect ratio (default 4; ~2.8-3.2 for a 1×2 row of panels).
5. **Never hardcode sizes in pt/in** other than height. Change `FONTSIZE` / `TEXTWIDTH_IN` in `plot_config.py` if the document changes.

## Colours and style

The house style lives in `plot_config.py` and is applied on import. Do not override it with ad-hoc colours or styling.

- **Palette:** teal `#3FA38F`, golden yellow `#E6B93D`, burnt orange `#D9692B`, slate gray `#6B7280` (`COLORS`). This is the default colour cycle, so plain `ax.plot(...)` calls already use it in this order. Don't pass `color=` with other hex codes, `tab:*` names or `"black"`.
- **Markers:** `MARKERS` (`o s ^ D`) are not applied by default. For a handful of comparable series, add them so curves stay distinguishable in grayscale: `zip(series, COLORS, MARKERS)`. Skip them for dense curves.
- **More than 4 series:** don't invent colours. Vary linestyle, or split into panels, and ask the user if unsure.
- **Reference lines** (zero line, thresholds): `color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE`, never black.
- **Already set by rcParams, don't repeat:** slate text/ticks (`INK`), grey spines, faint grid, no top/right spines, frameless legend, line width 1.6.
- Import what you need: `from plotting.plot_config import savefig, TEXTWIDTH_IN, COLORS, MARKERS, MUTED, REF_LINESTYLE`.
- To change the palette or style, edit `plot_config.py` only, so every figure stays consistent.

## LaTeX-safe text (`text.usetex=True`)

Plain strings are passed to LaTeX, so special characters break:
- Escape `_`, `%`, `&`, `#`: write `r"\_"` etc. Never pass raw identifiers like `shifted_cosine` as labels or titles; use readable labels ("Shifted cosine").
- Use raw strings for math: `r"$\bar{\alpha}_t$"`, `r"$\mathrm{pool}(x_t)$"`.
- Keep labels short; at 10pt, legends and y-labels take space.

## Workflow

1. Put plotting code in `plotting/`. Reuse existing helpers in `plotting/` before writing new ones.
2. Write the figure with the rules above.
3. Output always goes to `figures/` (repo root) unless the user names another directory. The user copies PDFs from there into the thesis `graphics/` folder.
4. Tell the user the output path and the width used. Mention how to include it: `\includegraphics{<name>.pdf}` with **no** `width=` option.
5. Do not run on HPC paths: `usetex` needs a local LaTeX install (`/Library/TeX/texbin/latex` on the Mac). Keep plotting out of `main.py`/training code.

## Template

```python
import matplotlib.pyplot as plt
from plotting.plot_config import savefig, TEXTWIDTH_IN, COLORS, MARKERS, MUTED, REF_LINESTYLE

fig, axes = plt.subplots(1, 2, sharey=True)
for ax, (title, series) in zip(axes, data.items()):
    for (label, y), color, marker in zip(series.items(), COLORS, MARKERS):
        ax.plot(x, y, color=color, marker=marker, label=label)
    ax.axhline(0.0, color=MUTED, linewidth=0.8, linestyle=REF_LINESTYLE)
    ax.set_title(title)
    ax.set_xlabel("timestep $t$")
    ax.legend()
axes[0].set_ylabel("correlation")

savefig(fig, "figures", "my_figure", width=TEXTWIDTH_IN, height=2.8)
```

## Verification

Running requires data and a LaTeX install. If you cannot run it, say so; do not claim it works. If you do run it, check the PDF exists and report any LaTeX errors verbatim.
