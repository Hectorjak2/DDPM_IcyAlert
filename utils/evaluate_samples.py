"""Evaluate generated SIC samples against the real CARRA2 distribution.

This is a *distributional* evaluation for the unconditional-generation task: we are
not comparing a sample to a specific target field, we are asking whether the set of
generated fields looks like the set of real fields.

Two failure modes have already burned us (see docs/diaries/):
  1. A scalar loss that "converged" while the samples were noise.
  2. Samples whose *mean* looked plausible (~0.52 water) but which were just the prior
     mean -- zero spatial structure, almost no open-water pixels.
Both are invisible to a point comparison and both are caught here, by looking at the
value distribution and the spatial structure rather than any single number.

Metrics (the priority set):
  1. Water-pixel concentration histogram + 1-D Wasserstein distance    -> value collapse
  2. Radially-averaged power spectrum (log-log)                        -> missing structure
  3. Ice-area / mean-concentration distribution across fields          -> set-level realism
  plus sanity checks (value range, land-mask agreement).

Data conventions (from utils/dataloader.py and models/ddpm.py):
  * samples.pkl is a pickled *list* of tensors, each [1,1,H,W], in data range [0,1]
    with land pixels = NaN.
  * CARRA2(...) returns [1,H,W] in the model range [-1,1] with land = NaN; we map it
    back to [0,1] with to_data_range so samples and data are compared in the same units.

Usage:
    python3 utils/evaluate_samples.py <experiment_name> [--n-real N] [--outdir DIR]
    python3 utils/evaluate_samples.py --samples path/to/samples.pkl [...]

Everything runs on CPU and writes a small JSON summary + a few PNGs; nothing here
touches the HPC or the training pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

# Headless: never assume a display is attached.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Make "utils.*" / "config" importable when run as a script from the repo root.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataloader import CARRA2, to_data_range  # noqa: E402


def wasserstein_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1-D Wasserstein (earth-mover) distance between two empirical samples.

    W1 = integral |F_u(x) - F_v(x)| dx, computed on the merged support. Self-contained
    so the script has no scipy dependency (scipy is not in the project venv).
    """
    u = np.sort(np.asarray(u, dtype=np.float64))
    v = np.sort(np.asarray(v, dtype=np.float64))
    if u.size == 0 or v.size == 0:
        return float("nan")
    grid = np.concatenate([u, v])
    grid.sort()
    deltas = np.diff(grid)
    # CDF of each sample evaluated just to the left of each grid point.
    cdf_u = np.searchsorted(u, grid[:-1], side="right") / u.size
    cdf_v = np.searchsorted(v, grid[:-1], side="right") / v.size
    return float(np.sum(np.abs(cdf_u - cdf_v) * deltas))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _to_hw(t) -> np.ndarray:
    """Coerce any [.,.,H,W]/[.,H,W]/[H,W] tensor-or-array to a 2-D float32 numpy array."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.asarray(t, dtype=np.float32)
    t = np.squeeze(t)
    if t.ndim != 2:
        raise ValueError(f"Expected a 2-D field after squeeze, got shape {t.shape}")
    return t


def load_samples(pickle_path: Path) -> np.ndarray:
    """Load samples.pkl -> stacked array [N, H, W] in [0,1] with land = NaN."""
    with open(pickle_path, "rb") as f:
        samples = pickle.load(f)
    if isinstance(samples, torch.Tensor):
        samples = [samples[i] for i in range(samples.shape[0])]
    fields = [_to_hw(s) for s in samples]
    shapes = {f.shape for f in fields}
    if len(shapes) != 1:
        raise ValueError(f"Samples have inconsistent shapes: {shapes}")
    return np.stack(fields, axis=0)


def load_real(n_real: int, west: bool, test: bool) -> np.ndarray:
    """Load real CARRA2 fields -> [N, H, W] in [0,1] with land = NaN.

    Sampled evenly across the time axis so the seasonal cycle is represented -- an
    all-January subset has almost no between-field variance and makes the spatial
    metrics lie (see diary 07-09-26_2213, "the leakage test is easy to make lie").
    """
    ds = CARRA2("siconc", device="cpu", WEST=west, TEST=test, batch_dim=False)
    total = len(ds)
    if n_real >= total:
        idx = range(total)
    else:
        idx = np.linspace(0, total - 1, n_real).round().astype(int)
        idx = sorted(set(int(i) for i in idx))
    fields = [_to_hw(to_data_range(ds[i])) for i in idx]
    return np.stack(fields, axis=0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def water_values(fields: np.ndarray) -> np.ndarray:
    """All finite (water) pixel values across every field, flattened."""
    v = fields.reshape(-1)
    return v[np.isfinite(v)]


def per_field_stats(fields: np.ndarray) -> dict:
    """Distribution of interpretable per-field summaries (set-level realism).

    * mean_conc : mean SIC over water   -- overall iciness
    * ice_frac  : fraction of water with SIC > 0.15 (the standard ice-edge threshold)
    * spatial_std : within-field std over water -- flat prior-mean fields score ~0 here
    """
    means, ice_fracs, stds = [], [], []
    for f in fields:
        w = f[np.isfinite(f)]
        if w.size == 0:
            continue
        means.append(float(w.mean()))
        ice_fracs.append(float((w > 0.15).mean()))
        stds.append(float(w.std()))
    return {
        "mean_conc": np.array(means),
        "ice_frac": np.array(ice_fracs),
        "spatial_std": np.array(stds),
    }


def radial_power_spectrum(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum of one field.

    Land/NaN is filled with the field's water-mean before the FFT (fills with 0 in the
    mean-subtracted field), so land contributes no spurious edge. Returns (k, power)
    with k in integer radial-frequency bins, excluding the DC term.
    """
    f = field.astype(np.float64)
    mask = np.isfinite(f)
    if not mask.any():
        raise ValueError("Field is entirely NaN.")
    f = f - f[mask].mean()
    f[~mask] = 0.0

    fft = np.fft.fftshift(np.fft.fft2(f))
    power = np.abs(fft) ** 2

    h, w = f.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)

    rmax = min(cy, cx)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / np.maximum(nr, 1)
    k = np.arange(1, rmax)  # drop DC (k=0)
    return k, radial[1:rmax]


def mean_radial_spectrum(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average the radial power spectrum over a set of fields (on a shared k-grid)."""
    specs = []
    k_ref = None
    for f in fields:
        k, p = radial_power_spectrum(f)
        if k_ref is None:
            k_ref = k
        specs.append(p[: len(k_ref)])
    return k_ref, np.mean(np.stack(specs, axis=0), axis=0)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_histogram(sample_vals, real_vals, out_path: Path, wass: float):
    bins = np.linspace(0.0, 1.0, 51)
    plt.figure(figsize=(7, 4))
    plt.hist(real_vals, bins=bins, density=True, alpha=0.5, label="real", color="tab:blue")
    plt.hist(sample_vals, bins=bins, density=True, alpha=0.5, label="samples", color="tab:orange")
    plt.xlabel("sea ice concentration (water pixels)")
    plt.ylabel("density")
    plt.title(f"Concentration distribution  (Wasserstein = {wass:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_spectrum(k_s, p_s, k_r, p_r, out_path: Path):
    plt.figure(figsize=(7, 4))
    plt.loglog(k_r, p_r, label="real", color="tab:blue")
    plt.loglog(k_s, p_s, label="samples", color="tab:orange")
    plt.xlabel("radial spatial frequency k")
    plt.ylabel("mean power")
    plt.title("Radially-averaged power spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()


def plot_field_stats(sstats, rstats, out_path: Path):
    keys = ["mean_conc", "ice_frac", "spatial_std"]
    titles = ["mean concentration", "ice fraction (>0.15)", "within-field spatial std"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, keys, titles):
        r, s = rstats[key], sstats[key]
        lo = float(min(r.min() if r.size else 0, s.min() if s.size else 0))
        hi = float(max(r.max() if r.size else 1, s.max() if s.size else 1))
        bins = np.linspace(lo, hi, 21) if hi > lo else 20
        ax.hist(r, bins=bins, density=True, alpha=0.5, label="real", color="tab:blue")
        ax.hist(s, bins=bins, density=True, alpha=0.5, label="samples", color="tab:orange")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Per-field summary distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def evaluate(samples_path: Path, outdir: Path, n_real: int, west: bool, test: bool,
             max_hist_pts: int = 500_000) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(samples_path)
    real = load_real(n_real, west=west, test=test)

    report: dict = {
        "samples_path": str(samples_path),
        "n_samples": int(samples.shape[0]),
        "sample_shape": list(samples.shape[1:]),
        "n_real": int(real.shape[0]),
        "real_shape": list(real.shape[1:]),
    }

    # --- sanity checks -----------------------------------------------------
    shape_match = samples.shape[1:] == real.shape[1:]
    report["shape_match"] = bool(shape_match)
    sv_all = samples.reshape(samples.shape[0], -1)
    finite_sv = sv_all[np.isfinite(sv_all)]
    report["sample_value_range"] = [float(finite_sv.min()), float(finite_sv.max())]
    report["sample_out_of_range_frac"] = float(
        ((finite_sv < 0.0) | (finite_sv > 1.0)).mean()
    )
    if shape_match:
        # Do samples and data mark the SAME pixels as land?
        sample_land = ~np.isfinite(samples[0])
        real_land = ~np.isfinite(real[0])
        report["land_mask_agreement"] = float((sample_land == real_land).mean())

    # --- 1. value distribution + Wasserstein -------------------------------
    sw = water_values(samples)
    rw = water_values(real)

    def _sub(a):
        if a.size > max_hist_pts:
            rng = np.random.default_rng(0)
            return a[rng.choice(a.size, max_hist_pts, replace=False)]
        return a

    wass = float(wasserstein_distance(_sub(sw), _sub(rw)))
    report["wasserstein_concentration"] = wass
    report["sample_water_mean"] = float(sw.mean())
    report["real_water_mean"] = float(rw.mean())
    report["sample_frac_below_0.05"] = float((sw < 0.05).mean())
    report["real_frac_below_0.05"] = float((rw < 0.05).mean())
    plot_histogram(_sub(sw), _sub(rw), outdir / "hist_concentration.png", wass)

    # --- 2. power spectrum -------------------------------------------------
    if shape_match:
        k_s, p_s = mean_radial_spectrum(samples)
        k_r, p_r = mean_radial_spectrum(real)
        n = min(len(k_s), len(k_r))
        # log-space L1 between the two mean spectra: scale-free structure mismatch.
        spec_l1 = float(np.mean(np.abs(np.log10(p_s[:n] + 1e-30) - np.log10(p_r[:n] + 1e-30))))
        report["spectrum_log_l1"] = spec_l1
        plot_spectrum(k_s, p_s, k_r, p_r, outdir / "power_spectrum.png")
    else:
        report["spectrum_log_l1"] = None
        report["_warning_shape"] = (
            "sample/real shapes differ; skipped spectrum & land-mask checks. "
            "Check the region flags (--west/--test) match how the samples were generated."
        )

    # --- 3. per-field summary distributions --------------------------------
    sstats = per_field_stats(samples)
    rstats = per_field_stats(real)
    for key in ("mean_conc", "ice_frac", "spatial_std"):
        s, r = sstats[key], rstats[key]
        report[f"field_{key}_sample_mean"] = float(s.mean()) if s.size else None
        report[f"field_{key}_real_mean"] = float(r.mean()) if r.size else None
        if s.size and r.size:
            report[f"field_{key}_wasserstein"] = float(wasserstein_distance(s, r))
    plot_field_stats(sstats, rstats, outdir / "field_stats.png")

    # --- write summary -----------------------------------------------------
    with open(outdir / "evaluation.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def _print_summary(report: dict) -> None:
    print("\n=== Sample evaluation ===")
    print(f"samples: {report['n_samples']} x {report['sample_shape']}   "
          f"real: {report['n_real']} x {report['real_shape']}")
    if report.get("_warning_shape"):
        print(f"!! {report['_warning_shape']}")
    print(f"value range (water): {report['sample_value_range']}  "
          f"out-of-[0,1] frac: {report['sample_out_of_range_frac']:.2e}")
    if "land_mask_agreement" in report:
        print(f"land-mask agreement: {report['land_mask_agreement']:.4f}")
    print("-- value distribution --")
    print(f"  water mean   sample {report['sample_water_mean']:.3f}  vs real {report['real_water_mean']:.3f}")
    print(f"  frac < 0.05  sample {report['sample_frac_below_0.05']:.3f}  vs real {report['real_frac_below_0.05']:.3f}")
    print(f"  Wasserstein(concentration): {report['wasserstein_concentration']:.4f}   (lower is better)")
    if report.get("spectrum_log_l1") is not None:
        print("-- spatial structure --")
        print(f"  spectrum log-L1: {report['spectrum_log_l1']:.4f}   (lower is better)")
    print("-- per-field summaries (Wasserstein, lower is better) --")
    for key, label in [("mean_conc", "mean conc"), ("ice_frac", "ice frac"), ("spatial_std", "spatial std")]:
        w = report.get(f"field_{key}_wasserstein")
        if w is not None:
            print(f"  {label:11s}: sample {report[f'field_{key}_sample_mean']:.3f} "
                  f"vs real {report[f'field_{key}_real_mean']:.3f}   W={w:.4f}")
    print("=========================\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment_name", nargs="?", default=None,
                    help="results/<experiment_name>/samples.pkl is evaluated; "
                         "output goes to results/<experiment_name>/evaluation/.")
    ap.add_argument("--samples", type=str, default=None,
                    help="Explicit path to a samples.pkl (overrides experiment_name).")
    ap.add_argument("--outdir", type=str, default=None, help="Where to write plots+json.")
    ap.add_argument("--n-real", type=int, default=96,
                    help="Number of real fields to compare against, spread across time "
                         "(default 96; the diary found n>=96 is needed for stable spatial stats).")
    ap.add_argument("--west", dest="west", action="store_true", default=True,
                    help="Compare against the CARRA2-WEST region (default).")
    ap.add_argument("--full", dest="west", action="store_false",
                    help="Compare against the full CARRA2 domain instead of WEST.")
    ap.add_argument("--test-region", dest="test", action="store_true", default=False,
                    help="Compare against the 128x128 TEST region.")
    args = ap.parse_args()

    if args.samples:
        samples_path = Path(args.samples)
        default_out = samples_path.parent / "evaluation"
    elif args.experiment_name:
        samples_path = Path("results") / args.experiment_name / "samples.pkl"
        default_out = Path("results") / args.experiment_name / "evaluation"
    else:
        ap.error("Provide either an experiment_name or --samples.")

    if not samples_path.exists():
        ap.error(f"samples file not found: {samples_path}")

    outdir = Path(args.outdir) if args.outdir else default_out
    report = evaluate(samples_path, outdir, n_real=args.n_real, west=args.west, test=args.test)
    _print_summary(report)
    print(f"Wrote {outdir}/evaluation.json and plots (hist_concentration.png, "
          f"power_spectrum.png, field_stats.png).")


if __name__ == "__main__":
    main()
