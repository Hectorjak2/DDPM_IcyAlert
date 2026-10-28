"""
Spectral analysis and noise synthesis for SIC fields.

Functions here operate purely in the frequency domain and are reusable both
during data preparation (estimating the noise PSD from the training set) and
during evaluation (verifying that degraded fields match the target PSD).
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

#Each carra2 pixel is 2.5km 
def radial_psd(
    field: np.ndarray,
    dx_km: float,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the azimuthally-averaged (radial) power spectral density of a 2-D field.

    A Hann window is applied before the FFT. Without it, the crop's sharp
    rectangular edges act as an implicit boxcar window, whose FFT leaks
    broadband power into every wavenumber bin (spectral leakage) — this masks
    real high-wavenumber differences between fields (e.g. it can make a
    heavily-blurred field's PSD look nearly identical to the unblurred one).

    Parameters
    ----------
    field  : 2-D SIC array.
    dx_km  : grid spacing [km].
    region : optional (y_start, y_end, x_start, x_end) indices that select a
             land-free rectangular sub-region before computing the FFT.
             Crop to ~400×400 points to avoid land artifacts and to fix the
             maximum resolved scale (400 × dx_km km).

    Returns
    -------
    k_centres : np.ndarray — wavenumber bin centres [cycles / km]
    psd       : np.ndarray — mean power in each bin
    """
    if region is not None:
        y0, y1, x0, x1 = region
        field = field[y0:y1, x0:x1]

    # Fill any residual NaNs (e.g. from Gaussian blur bleeding across land edges)
    # with the field mean so the FFT sees a continuous signal.
    if np.isnan(field).any():
        field = field.copy()
        field[np.isnan(field)] = np.nanmean(field)

    ny, nx = field.shape

    window = np.hanning(ny)[:, None] * np.hanning(nx)[None, :]
    window /= np.sqrt(np.mean(window ** 2))  # preserve overall power scale

    f2d = fft2((field - field.mean()) * window)
    power = (np.abs(f2d) ** 2) / (ny * nx)

    ky = fftfreq(ny, d=dx_km)
    kx = fftfreq(nx, d=dx_km)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_max = 0.5 / dx_km
    n_bins = min(ny, nx) // 2
    k_bins = np.linspace(0, k_max, n_bins + 1)
    psd = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.any():
            psd[i] = power[mask].mean()
            counts[i] = mask.sum()
    k_centres = 0.5 * (k_bins[:-1] + k_bins[1:])
    valid = counts > 0
    return k_centres[valid], psd[valid]


def find_landfree_region(
    field: np.ndarray,
    lat_arr: np.ndarray,
    size: int = 480,
    lat_min: float = 80.0,
) -> tuple[int, int, int, int] | None:
    """
    Find the first size×size sub-region that contains no NaN pixels and whose
    minimum latitude exceeds *lat_min*.

    The latitude constraint ensures the region is in the Arctic Ocean rather
    than lower-latitude seas that happen to be land-free.

    Parameters
    ----------
    field   : 2-D SIC array (one timestep is sufficient).
    lat_arr : 2-D latitude array matching field shape.
    size    : side length of the square window in grid points (default 400).
    lat_min : minimum latitude threshold in degrees (default 80°N).

    Returns
    -------
    (y0, y1, x0, x1) index tuple, or None if no valid region is found.
    """
    land = np.isnan(field).astype(np.int32)

    # 2-D prefix sum for O(1) NaN counting in any window
    cs = np.cumsum(np.cumsum(land, axis=0), axis=1)

    def nan_count(y: int, x: int) -> int:
        total = int(cs[y + size - 1, x + size - 1])
        if y > 0:
            total -= int(cs[y - 1, x + size - 1])
        if x > 0:
            total -= int(cs[y + size - 1, x - 1])
        if y > 0 and x > 0:
            total += int(cs[y - 1, x - 1])
        return total

    ny, nx = field.shape
    for y in range(ny - size):
        for x in range(nx - size):
            if nan_count(y, x) == 0:
                if lat_arr[y:y + size, x:x + size].min() >= lat_min:
                    return (y, y + size, x, x + size)
    return None
