# 15-10-26:13:02

**Focus:** Building the thesis figure of the baseline `SimpleUnet` with PlotNeuralNet ([plotting/simple_unet_arch.py](../../plotting/simple_unet_arch.py)) and what it took to make it fit `\textwidth` at 10pt.

## Theory
- **Text size is a scaling problem, not a font problem.** The diagram is ~40 cm wide in TikZ units and is scaled to `TEXTWIDTH_IN` (scale ≈ 0.37). So text drawn at 10pt lands at ~3.7pt. The script injects `\fontsize{F}` into the tikzpicture and iterates (compile, measure width, `F = FONTSIZE / scale`) until it converges; it ends at F ≈ 27pt drawn → 10pt in the thesis. Labels change the bounding box, hence the loop rather than one pass.
- **Why labels overlapped:** PlotNeuralNet's `RightBandedBox` puts one `xlabel` under *each* sub-box (0.4 cm wide). At 10pt final size two 3-digit labels collide. Fix: one label per ResBlock pair (`n_filer=("", ch)`, output channels). The label array is evaluated by pgfmath, so it **only accepts numbers**: `\shortstack{256\\128}` for the concat blocks crashes with "Undefined control sequence".
- **"Part-orange" ResBlocks:** box fill opacity 0.4 and band opacity 0.6 differ, so one colour still looks two-toned. Setting both to 1 fixes it, but any shared alpha < 1 re-creates it, because the band is drawn *over* the box and stacks. Correct fix: shared alpha on the box, `bandopacity=0`.
- **Image orientation:** `to_input` draws on the zy plane. The CARRA2 grid has y increasing northwards (needs `flipud`) and PlotNeuralNet's z axis runs right-to-left (needs `fliplr`). Both are applied in [plotting/unet_io_images.py](../../plotting/unet_io_images.py). Sanity check: Greenland on the left, Iceland lower right.

## Design decisions
- **PlotNeuralNet is not modified.** The script copies its `layers/` into a temp dir and patches the `.sty` files there (spatial-size label moved to `pos=0.5`, `\small` dropped, opacity). Alternative rejected: editing the clone, which would silently break on re-clone.
- **Block types are explained by a legend, not captions.** In-block captions collided with channel labels. Legend is a nested `tikzpicture` inside a `\node` anchored at the midpoint of `current bounding box`, so it stays centred when the model changes.
- **Figure shows the baseline U-Net only:** 3 input channels (mask, k-1, k) → 1 output (k+1). An earlier version showed the diffusion variant (x_t, k, k-1, mask → noise); dropped on request. If a DDPM figure is wanted later, a `xt.png`/`eps.png` generator is the missing piece.
- **Palette comes from `COLORS` in [plotting/plot_config.py](../../plotting/plot_config.py)**: Conv teal, ResBlock orange, Down golden, Up slate. Global `OPACITY = 0.6`.

## Code design
- Output goes to `figures/simple_unet.pdf` at exactly `\textwidth`; include **without** `width=`.
- **Draw order matters:** TikZ paints in source order. The output arrow is emitted *before* the output image so the image covers its end. The legend arrow cannot be a `node {\copymidarrow}` (baseline sits below the line); it is a line plus a separately drawn `-Stealth` segment.
- Input and output spacing share one `IO_GAP` (5). Stack offset is `STACK_DX`. Named anchors (`out-west`) do not exist for canvas image nodes, so positions are computed with `$(r3-east)+(IO_GAP,0,0)$` (needs `\usetikzlibrary{calc}`, set in `COLORDEFS`).
- Without `plotting/assets/*.png` the script falls back to plain blocks. Assets are rendered from `data/CARRA2_DAILY/west.zarr`; only year 2000 exists locally, so dates are 5 Feb / 5 Mar / 5 Apr 2000.
- Rough edge: PDFs are previewed with `sips -s format png file.pdf` (macOS built-in; no poppler needed). `channels` / `sizes` in the script were edited by hand to WEST values; they are not read from `config.py`.

## Open threads
- The stack of 3 input images is mostly edges at `STACK_DX = 0.6`; may want more separation.
- Input arrow is drawn on top of the front image, the output arrow behind. Inconsistent; left as requested.
- Up-path blocks show only output channels; the doubled concat input is not shown (only the "Skip (concat)" legend entry).

## Promotion candidates
- PlotNeuralNet pipeline facts (scale-to-`\textwidth` loop, patched layers copy, label array accepts numbers only) → a new "Figures" section in `docs/architecture.md`.
