"""Noise-schedule specs for the schedule plots, defined in the plot scripts instead of read from config.

Each plot script lists the schedules it shows, e.g.

    SCHEDULES = [
        Schedule("cosine"),
        Schedule("shifted_cosine", cosine_shift_res=64),
        Schedule("shifted_cosine", cosine_shift_res=256),
        Schedule("laplace", laplace_mu=-2, laplace_b=0.5),
    ]

Parameter names match DDPMConfig / models.ddpm.build_schedule; any left out fall back to DDPMConfig.
"""
import math

import torch

from config import DDPMConfig
from models.ddpm import DDPM, build_schedule

# Readable, LaTeX-safe names for DDPM.SCHEDULES
SCHEDULE_NAMES = {"linear": "Linear", "cosine": "Cosine", "laplace": "Laplace", "shifted_cosine": "Shifted cosine"}

# The parameters each schedule reads
SCHEDULE_PARAMS = {
    "linear": ("beta_start", "beta_end"),
    "cosine": ("cosine_s",),
    "shifted_cosine": ("cosine_s", "cosine_shift_res"),
    "laplace": ("laplace_mu", "laplace_b"),
}


class Schedule:
    """One noise schedule to plot: its kind, parameters and label.

    Args:
        kind: one of DDPM.SCHEDULES
        name: short name (row labels, legend prefix); defaults to SCHEDULE_NAMES[kind].
            Set it when one figure shows several variants of the same kind.
        **params: schedule parameters, see SCHEDULE_PARAMS; missing ones come from DDPMConfig
    """

    def __init__(self, kind: str, name: str = None, **params):
        if kind not in DDPM.SCHEDULES:
            raise ValueError(f"unknown schedule {kind!r}; expected one of {DDPM.SCHEDULES}")
        unknown = set(params) - set(SCHEDULE_PARAMS[kind])
        if unknown:
            raise ValueError(f"{kind} takes {SCHEDULE_PARAMS[kind]}, got {sorted(unknown)}")

        self.kind = kind
        self.name = name if name is not None else SCHEDULE_NAMES[kind]
        self.params = {p: params.get(p, getattr(DDPMConfig, p)) for p in SCHEDULE_PARAMS[kind]}

    def build(self, timesteps: int, image_size: int):
        """``(betas, alphas, alphas_bar)``, exactly as DDPM builds them during training."""
        return build_schedule(self.kind, timesteps, image_size, **self.params)

    def shift(self, image_size: int) -> float:
        """log-SNR shift delta = 2 log(r / image_size) of the shifted cosine; 0 for other schedules."""
        if self.kind != "shifted_cosine":
            return 0.0
        return 2 * math.log(self.params["cosine_shift_res"] / image_size)

    def label(self, image_size: int) -> str:
        """Legend label: name plus the parameters the schedule is built from."""
        p = self.params
        if self.kind == "linear":
            params = rf"$\beta_1 = {p['beta_start']:g}$, $\beta_T = {p['beta_end']:g}$"
        elif self.kind == "cosine":
            params = rf"$s = {p['cosine_s']:g}$"
        elif self.kind == "shifted_cosine":
            params = rf"$s = {p['cosine_s']:g}$, $\delta = {self.shift(image_size):.2f}$"
        elif self.kind == "laplace":
            params = rf"$\mu = {p['laplace_mu']:g}$, $b = {p['laplace_b']:g}$"
        return f"{self.name} ({params})"

    def logsnr_density(self, image_size: int):
        """Continuous-time log-SNR lambda and its density p(lambda) for t ~ U(0, 1)."""
        if self.kind in ("cosine", "shifted_cosine"):
            lam, p = _cosine_logsnr_density(self.params["cosine_s"])
            # The shift moves the cosine density by delta without changing its shape
            return lam + self.shift(image_size), p
        if self.kind == "laplace":
            mu, b = self.params["laplace_mu"], self.params["laplace_b"]
            lam = torch.linspace(mu - 30 * b, mu + 30 * b, 4000, dtype=torch.float64)
            return lam, (1 / (2 * b)) * torch.exp(-torch.abs(lam - mu) / b)
        raise NotImplementedError(f"no continuous-time log-SNR density for {self.kind!r}")


def _cosine_logsnr_density(s: float):
    """log-SNR lambda(t) and its density p(lambda) = 1 / |d lambda / dt| for t ~ U(0, 1).

    Uses alpha_bar(t) = f(t) / f(0), f(t) = cos^2((t + s) / (1 + s) * pi / 2), as in models/ddpm.py.
    For s = 0 this is 1 / (2 pi cosh(lambda / 2)); s > 0 thins the high-SNR tail into exp(-lambda).
    """
    # Log-spaced near t = 0, where lambda -> inf, so the tail is resolved up to the plot edge
    t = torch.cat([torch.logspace(-14, -2, 2000, dtype=torch.float64),
                   torch.linspace(1e-2, 1 - 1e-6, 20000, dtype=torch.float64)[1:]])
    du_dt = math.pi / (2 * (1 + s))
    u = (t + s) * du_dt
    f0 = math.cos(s * du_dt) ** 2
    alphas_bar = torch.cos(u) ** 2 / f0
    log_snr = torch.log(alphas_bar / (1 - alphas_bar))
    dalphas_bar_dt = -torch.sin(2 * u) * du_dt / f0
    dlog_snr_dt = dalphas_bar_dt / (alphas_bar * (1 - alphas_bar))
    return log_snr, 1 / dlog_snr_dt.abs()
