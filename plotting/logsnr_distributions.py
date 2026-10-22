"""Density p(lambda) of the log-SNR lambda for t ~ U(0, 1), one curve per noise schedule."""
import matplotlib.pyplot as plt

from plotting.config import savefig, TEXTWIDTH_IN
from plotting.schedules import Schedule

SCHEDULES = [
    Schedule("cosine"),
    Schedule("shifted_cosine", cosine_shift_res=64),
    Schedule("shifted_cosine", cosine_shift_res=128),
    Schedule("shifted_cosine", cosine_shift_res=256),
]

IMAGE_SIZE = 1216  # CARRA2-WEST; sets the shifted cosine's delta
XLIM = (-15, 10)
YLIM = (0, 0.3)  # raise the top (e.g. (0, 0.4)) to make room for the legend; None = auto

fig, ax = plt.subplots()
for schedule in SCHEDULES:
    lam, p = schedule.logsnr_density(IMAGE_SIZE)
    ax.plot(lam, p, label=schedule.label(IMAGE_SIZE))

ax.set_xlabel(r"log-SNR $\lambda$")
ax.set_ylabel(r"$p(\lambda)$")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.legend(loc="upper left")

savefig(fig, "figures", "logsnr_distributions", width=TEXTWIDTH_IN, height=2.8)
