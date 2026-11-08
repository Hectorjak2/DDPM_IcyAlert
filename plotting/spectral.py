"""
Spectral analysis and noise synthesis for SIC fields.

Functions here operate purely in the frequency domain and are reusable both
during data preparation (estimating the noise PSD from the training set) and
during evaluation (verifying that degraded fields match the target PSD).
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

def mean_spacing_km(
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
) -> float:
    """
    Mean true ground distance between neighbouring grid points [km].

    CARRA2 is on a polar-stereographic grid, which is conformal (pixels are
    locally square) but whose true spacing varies with latitude: ~2.34 km at
    61°N up to ~2.49 km at 84°N across CARRA_CROPPED. Within one PSD window
    the variation is well under 1%, so the window mean is used as dx_km.

    Parameters
    ----------
    lat_arr, lon_arr : 2-D latitude/longitude arrays [degrees] matching the field.
    region           : optional (y_start, y_end, x_start, x_end), as in radial_psd.
    """
    if region is not None:
        y0, y1, x0, x1 = region
        lat_arr, lon_arr = lat_arr[y0:y1, x0:x1], lon_arr[y0:y1, x0:x1]
    lat, lon = np.deg2rad(lat_arr), np.deg2rad(lon_arr)

    def haversine(lat1, lon1, lat2, lon2):
        a = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
        return 2 * 6371.0 * np.arcsin(np.sqrt(a))

    dx = haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    dy = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return float(np.mean(np.concatenate([dx.ravel(), dy.ravel()])))


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
    dx_km  : grid spacing [km]; use mean_spacing_km over the same region.
    region : optional (y_start, y_end, x_start, x_end) indices that select a
             land-free rectangular sub-region before computing the FFT, which
             avoids land artifacts and fixes the maximum resolved scale
             (size × dx_km km; ~320 km for a 128×128 window).

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

    # Rings of width df (the fundamental wavenumber, one cycle per window) centred on
    # df, 2 df, ..., Nyquist. Ring 0 is only the mean (k = 0), which is removed above and
    # has no wavelength, so it is dropped; the largest resolved wavelength is the window size.
    df = 1.0 / (min(ny, nx) * dx_km)
    n_bins = min(ny, nx) // 2
    ring = np.rint(K / df).astype(int).ravel()
    sums = np.bincount(ring, weights=power.ravel(), minlength=n_bins + 1)[1:n_bins + 1]
    counts = np.bincount(ring, minlength=n_bins + 1)[1:n_bins + 1]
    k_centres = df * np.arange(1, n_bins + 1)
    valid = counts > 0
    return k_centres[valid], sums[valid] / counts[valid]


def find_landfree_region(
    field: np.ndarray,
    lat_arr: np.ndarray,
    size: int = 128,
    lat_min: float = 67.0,
) -> tuple[int, int, int, int] | None:
    """
    Find the first size×size sub-region (raster order) that contains no NaN
    pixels and whose minimum latitude exceeds *lat_min*.

    The latitude constraint keeps the window north of Iceland (~63.4–66.6°N),
    out of the ice-free North Atlantic. On CARRA_CROPPED the land mask allows
    no land-free 256×256 window anywhere; 192 is the largest that fits. With
    the defaults the result is (166, 294, 742, 870) in crop coordinates, which
    is the Denmark Strait (67–70°N, 23–14°W) and has ice-edge structure on most
    days of the year.

    Parameters
    ----------
    field   : 2-D SIC array (one timestep is sufficient; the land mask is static).
    lat_arr : 2-D latitude array matching field shape.
    size    : side length of the square window in grid points (default 128).
    lat_min : minimum latitude threshold in degrees (default 67°N).

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
    for y in range(ny - size + 1):
        for x in range(nx - size + 1):
            if nan_count(y, x) == 0:
                if lat_arr[y:y + size, x:x + size].min() >= lat_min:
                    return (y, y + size, x, x + size)
    return None
