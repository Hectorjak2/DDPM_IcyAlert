"""
PlotNeuralNet diagram of SimpleUnet (models/hec_unet.py), with channels and n_res_blocks read from
UnetConfig (config.py), so the figure follows the network that actually trains.

Layout mirrors SimpleUnet: per level, n_res_blocks ResBlocks then a Downsample (none on the last
level); bottleneck of 2 ResBlocks; per decoder level, n_res_blocks + 1 ResBlocks then an Upsample
(none on level 0). Skips are drawn from the same push/pop order as SimpleUnet.forward: in_conv,
every encoder ResBlock and every Downsample push one; every decoder ResBlock pops one.

Usage (from the repo root):
    python3 plotting/simple_unet_arch.py   # -> figures/simple_unet.pdf

The diagram is compiled with pdflatex in a temp dir and then scaled to exactly
\\textwidth (TEXTWIDTH_IN from plot_config), so include it in the thesis with
\\includegraphics{simple_unet.pdf} and no width= option.

PlotNeuralNet is looked up in $PLOTNEURALNET, else ../../PlotNeuralNet
(i.e. cloned next to the DDPM_IcyAlert folder).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PNN = os.environ.get("PLOTNEURALNET", os.path.abspath(os.path.join(HERE, "..", "..", "PlotNeuralNet")))
if not os.path.isdir(os.path.join(PNN, "pycore")):
    sys.exit(f"PlotNeuralNet not found at {PNN}. Clone it or set PLOTNEURALNET.")
sys.path.append(PNN)
sys.path.append(os.path.abspath(os.path.join(HERE, "..")))
from pycore.tikzeng import *  # noqa: E402
from plotting.config import TEXTWIDTH_IN, FONTSIZE, COLORS  # noqa: E402
from config import UnetConfig  # noqa: E402

_cfg = UnetConfig()
channels = list(_cfg.channels)
N_RES = _cfg.n_res_blocks
sizes = [1216 // 2 ** i for i in range(len(channels))]  # spatial size at each level (1216x1216 WEST grid)
heights = [40, 32, 24, 18]      # drawn size, shrinking with spatial size
assert len(heights) == len(channels), "add a drawn height per level"
OPACITY = 0.6    # alpha of every block (and legend swatch)
GAP = "(2,0,0)"   # spacing between stages
STEP = "(1,0,0)"  # spacing between ResBlocks within a stage
NEAR = "(0,0,0)"  # spacing between a stage and its down/up block
RES_W = 3         # drawn width of a ResBlock box
UNIT = 0.2        # cm per height unit (IMG = 8cm is drawn as 40 units)
ZY = 0.385        # vertical rise per cm of depth in PlotNeuralNet's 3D view (TikZ default z vector)
# Skip arcs of level i run from SKIP_MARGIN to SKIP_MARGIN + SKIP_SPREAD above the highest far corner
# they start from or pass over; this stays below the arcs of level i-1 for the default heights.
SKIP_MARGIN, SKIP_SPREAD = 0.3, 0.4


def far_top(h):
    """Height (cm) of the far top edge of a box with height = depth = h units."""
    return h * UNIT / 2 * (1 + ZY)

# House palette from plot_config: conv teal, ResBlock orange, down golden, up slate
_CONV, _POOL, _RES, _UP = (c.lstrip("#") for c in (COLORS[0], COLORS[1], COLORS[2], COLORS[3]))
COLORDEFS = rf"""
\definecolor{{cConv}}{{HTML}}{{{_CONV}}}
\definecolor{{cPool}}{{HTML}}{{{_POOL}}}
\definecolor{{cRes}}{{HTML}}{{{_RES}}}
\definecolor{{cUp}}{{HTML}}{{{_UP}}}
\def\ConvColor{{cConv}}
\def\PoolColor{{cPool}}
\def\UnpoolColor{{cUp}}
\def\ResColor{{cRes}}
\usetikzlibrary{{calc}}
"""

arch = [to_head(PNN), to_cor(), COLORDEFS, to_begin()]

# Baseline U-Net input: 3 stacked 1216x1216 fields (land mask, day k-1, day k); output: day k+1,
# drawn as real images (see plotting/unet_io_images.py). Falls back to plain blocks without the PNGs.
ASSETS = os.path.join(HERE, "assets")
IN_IMAGES = ["mask", "km1", "k"]   # back -> front
OUT_IMAGE = "kp1"
HAVE_IMAGES = all(os.path.exists(os.path.join(ASSETS, f"{n}.png")) for n in IN_IMAGES + [OUT_IMAGE])
IMG = 8          # cm, = drawn depth of the 40-unit blocks
STACK_DX = 0.6   # x offset between stacked input channels
IO_GAP = 5       # x distance between the input stack / output image and the network (same on both sides)

if HAVE_IMAGES:
    for n, img in enumerate(IN_IMAGES):
        arch.append(to_input(os.path.join(ASSETS, f"{img}.png"), to=f"({-IO_GAP - STACK_DX * (len(IN_IMAGES) - 1 - n)},0,0)",
                             width=IMG, height=IMG, name=f"in{n}"))
    arch.append(rf"\node[anchor=north] at ({-IO_GAP - STACK_DX * (len(IN_IMAGES) - 1) / 2},-5.5) {{{len(IN_IMAGES)}$\times$1216$^2$}};")

def res_block(name, level, offset, to, label):
    """One ResBlock as a single box. Only one box per stage is labelled: labels under adjacent
    0.6cm-wide boxes would overlap at 10pt, and a stage shares size and channels."""
    h = heights[level]
    # channels only: the spatial size follows from the Downsample/Upsample blocks (in/out images give 1216^2)
    labels = f"xlabel={{{{{channels[level]}, }}}}," if label else ""
    return rf"""
\pic[shift={{{offset}}}] at {to}
    {{Box={{name={name}, caption= , {labels} fill=\ResColor, height={h}, width={RES_W}, depth={h}}}}};
"""


def skip(of, to, y):
    """Concat skip: straight up from `of`, across at height y (cm), down into `to`."""
    return (rf"\draw [copyconnection] ({of}-north) -- ({of}-north |- 0,{y:.2f}) "
            rf"-- node {{\copymidarrow}} ({to}-north |- 0,{y:.2f}) -- ({to}-north);")


# input conv
arch.append(to_Conv("inconv", "", channels[0], offset="(0,0,0)", to="(0,0,0)",
                    height=40, depth=40, width=2, caption=" "))
if HAVE_IMAGES:
    arch.append(rf"\draw [connection] ({-IO_GAP},0,0) -- (inconv-west);")
prev = "inconv"
skip_stack = [("inconv", 0)]  # (block, level whose resolution its output has), as in SimpleUnet.forward

# down path: N_RES x ResBlock, stride-2 conv downsample (not on the last level)
for i in range(len(channels)):
    for k in range(N_RES):
        name = f"d{i}_{k}"
        arch.append(res_block(name, i, GAP if k == 0 else STEP, f"({prev}-east)", label=k == N_RES - 1))
        arch.append(to_connection(prev, name))
        skip_stack.append((name, i))
        prev = name
    if i != len(channels) - 1:
        h = heights[i]
        arch.append(to_Pool(f"p{i}", offset=NEAR, to=f"({prev}-east)",
                            width=1, height=h - 4, depth=h - 4, caption=" ", opacity=OPACITY))
        skip_stack.append((f"p{i}", i + 1))  # its output is already at the next level's resolution
        prev = f"p{i}"

# bottleneck: 2 x ResBlock at the last level's resolution
for k in range(2):
    name = f"bn{k}"
    arch.append(res_block(name, len(channels) - 1, GAP if k == 0 else STEP, f"({prev}-east)", label=k == 1))
    arch.append(to_connection(prev, name))
    prev = name

# up path: (N_RES + 1) x [concat skip, ResBlock], then upsample (not on level 0)
for i in reversed(range(len(channels))):
    # highest box under this level's arcs: its own boxes, or the Downsample/in_conv the outer arc starts at
    clear = max(far_top(heights[i]), far_top(heights[i - 1] - 4) if i > 0 else far_top(40))
    for k in range(N_RES + 1):
        name = f"r{i}_{k}"
        # label the first box: the stage ends in an Upsample taller than the boxes, which would cover the labels
        arch.append(res_block(name, i, GAP if k == 0 else STEP, f"({prev}-east)", label=k == 0))
        arch.append(to_connection(prev, name))
        src, src_level = skip_stack.pop()
        assert src_level == i, f"skip {src} (level {src_level}) popped at level {i}"
        # first pop is the innermost source, so it gets the lowest arc
        arch.append(skip(src, name, clear + SKIP_MARGIN + k * SKIP_SPREAD / N_RES))
        prev = name
    if i != 0:
        h = heights[i - 1]  # mirrors the Downsample between levels i-1 and i
        arch.append(to_UnPool(f"u{i}", offset=NEAR, to=f"({prev}-east)",
                              width=1, height=h - 4, depth=h - 4, caption=" ", opacity=OPACITY))
        prev = f"u{i}"
assert not skip_stack, f"unused skips: {skip_stack}"

# output conv (GroupNorm + SiLU + 3x3 conv to 1 channel), mirrors the input conv
arch.append(to_Conv("outconv", "", 1, offset=GAP, to=f"({prev}-east)",
                    height=40, depth=40, width=2, caption=" "))
arch.append(to_connection(prev, "outconv"))
if HAVE_IMAGES:
    # arrow first, so the output image is drawn over its end and the line stays behind it
    arch.append(rf"\draw [connection] (outconv-east) -- ($(outconv-east)+({IO_GAP},0,0)$);")
    arch.append(to_input(os.path.join(ASSETS, f"{OUT_IMAGE}.png"), to=f"($(outconv-east)+({IO_GAP},0,0)$)", width=IMG, height=IMG, name="out"))
    arch.append(rf"\node[anchor=north] at ($(outconv-east)+({IO_GAP},-5.5,0)$) {{1$\times$1216$^2$}};")
# Legend: block types by colour (replaces per-block captions). Text size is set globally in main()
# so that it ends up at FONTSIZE after the figure is scaled to \\textwidth.
LEGEND = r"""
\coordinate (leg) at ([yshift=-5cm]inconv-south);
\coordinate (bbmid) at ($(current bounding box.west)!0.5!(current bounding box.east)$);
\node[anchor=north, inner sep=0] at (bbmid |- leg) {\begin{tikzpicture}
\node[fill=\ConvColor, fill opacity=OPA, minimum size=1.4cm, anchor=west] (l1) {};
\node[anchor=west, right=0.3cm of l1] (t1) {Conv};
\node[fill=\ResColor, fill opacity=OPA, minimum size=1.4cm, anchor=west, right=1.5cm of t1] (l2) {};
\node[anchor=west, right=0.3cm of l2] (t2) {ResBlock};
\node[fill=\PoolColor, fill opacity=OPA, minimum size=1.4cm, anchor=west, right=1.5cm of t2] (l3) {};
\node[anchor=west, right=0.3cm of l3] (t3) {Downsample};
\node[fill=\UnpoolColor, fill opacity=OPA, minimum size=1.4cm, anchor=west, right=1.5cm of t3] (l4) {};
\node[anchor=west, right=0.3cm of l4] (t4) {Upsample};
\draw[copyconnection] ([xshift=1.5cm]t4.east) coordinate (l5s) -- ++(3,0) coordinate (l5);
\draw[-Stealth, line width=0.8mm, draw={rgb:blue,4;red,1;green,1;black,3}, opacity=0.7] ([xshift=1.2cm]l5s) -- ++(0.3,0);
\node[anchor=west, right=0.3cm of l5] (t5) {Skip (concat)};
\end{tikzpicture}};
"""
arch.append(LEGEND.replace("OPA", str(OPACITY)))
arch.append(to_end())


def _pdflatex(tex_path: str):
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
                         cwd=os.path.dirname(tex_path), stdin=subprocess.DEVNULL,
                         capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(res.stdout[-3000:])


def _patched_layers(tmp: str) -> str:
    """Copy PlotNeuralNet's layers/ into tmp and set every block's opacity to OPACITY (PlotNeuralNet itself stays untouched)."""
    dst = os.path.join(tmp, "layers")
    shutil.copytree(os.path.join(PNN, "layers"), dst)
    path = os.path.join(dst, "Box.sty")
    text = open(path).read()
    open(path, "w").write(text.replace("opacity=0.4,", f"opacity={OPACITY},"))
    return tmp


def _natural_width_pt(pdf_path: str) -> float:
    """Width of the compiled PDF in pt, measured by LaTeX itself (the MediaBox can sit in a compressed object)."""
    probe = os.path.join(os.path.dirname(pdf_path), "probe.tex")
    with open(probe, "w") as f:
        f.write("\\documentclass{article}\\usepackage{graphicx}\\newsavebox\\probebox\\begin{document}"
                f"\\sbox\\probebox{{\\includegraphics{{{os.path.basename(pdf_path)}}}}}\\typeout{{PROBEWIDTH=\\the\\wd\\probebox}}\\end{{document}}\n")
    res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "probe.tex"],
                         cwd=os.path.dirname(pdf_path), stdin=subprocess.DEVNULL, capture_output=True, text=True)
    return float(re.search(r"PROBEWIDTH=([\d.]+)pt", res.stdout).group(1))


def main(output_dir: str = os.path.join(HERE, "..", "figures"), name: str = "simple_unet"):
    os.makedirs(output_dir, exist_ok=True)
    width_pt = TEXTWIDTH_IN * 72.27
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.tex")
        pieces = list(arch)
        # fix-cm: Computer Modern at any size. Without it LaTeX silently caps the ~30pt drawn font at
        # 24.88pt, and after scaling to \textwidth the text ends up well below FONTSIZE.
        pieces[0] = "\\RequirePackage{fix-cm}\n" + to_head(_patched_layers(tmp))
        to_generate(pieces, raw)
        tex = open(raw).read()

        # The diagram is scaled to \textwidth by `scale`, so text drawn at size F ends up at F*scale.
        # Pick F so the final text is FONTSIZE pt (like thesis-plot); iterate since labels affect the width.
        font = float(FONTSIZE)
        for _ in range(6):
            with open(raw, "w") as f:
                f.write(tex.replace("\\begin{tikzpicture}",
                                    f"\\begin{{tikzpicture}}\\fontsize{{{font:.2f}}}{{{1.2 * font:.2f}}}\\selectfont", 1))
            _pdflatex(raw)
            scale = width_pt / _natural_width_pt(os.path.join(tmp, "raw.pdf"))
            new_font = FONTSIZE / scale
            converged = abs(new_font - font) < 0.01 * font
            font = new_font
            if converged:
                break

        wrapper = os.path.join(tmp, name + ".tex")
        with open(wrapper, "w") as f:
            f.write("\\documentclass[border=0pt]{standalone}\n\\usepackage{graphicx}\n"
                    f"\\begin{{document}}\\includegraphics[width={width_pt:.2f}pt]{{raw.pdf}}\\end{{document}}\n")
        _pdflatex(wrapper)
        os.replace(os.path.join(tmp, name + ".pdf"), os.path.join(output_dir, name + ".pdf"))
    print(f"wrote {os.path.join(output_dir, name)}.pdf at width {TEXTWIDTH_IN:.2f} in; "
          f"scale {scale:.3f}, drawn font {font:.1f}pt -> {FONTSIZE}pt in the thesis")


if __name__ == "__main__":
    main()
