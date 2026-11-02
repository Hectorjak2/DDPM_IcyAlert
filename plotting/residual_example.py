"""One CARRA2-CROPPED example of the ConditionalDDPM target: SIC on day k, a month later (k+1), and the residual k+1 - k."""
import matplotlib.pyplot as plt

from models.models import DatasetChoice
from plotting.config import savefig, TEXTWIDTH_IN
from plotting.evaluation import SIC_CMAP, _clean_image_axis
from utils.dataloader import CARRA2

DATE = "2010-06-15"  # early melt season, so the month-ahead change is large
MONTH_APART = "2010-07-15"

RESIDUAL_CMAP = plt.cm.RdBu.copy()
RESIDUAL_CMAP.set_bad(SIC_CMAP.get_bad())  # land in the same grey as the SIC panels


def load_sic(date):
    """[H, W] SIC in [0, 1], land NaN. Each date loaded on its own, as in forward_process."""
    ds = CARRA2("siconc", "cpu", time_slice=slice(date, date), area=DatasetChoice.CARRA_CROPPED)
    return ds.to_sic(ds[0])[0, 0].numpy()


k = load_sic(DATE)
kp1 = load_sic(MONTH_APART)
residual = kp1 - k  # land stays NaN (shown grey) rather than the 0 used in training

fig, (ax_k, ax_kp1, ax_res) = plt.subplots(1, 3)

sic_img = ax_k.imshow(k, cmap=SIC_CMAP, vmin=0.0, vmax=1.0, origin="lower", interpolation="none")
ax_kp1.imshow(kp1, cmap=SIC_CMAP, vmin=0.0, vmax=1.0, origin="lower", interpolation="none")
res_img = ax_res.imshow(residual, cmap=RESIDUAL_CMAP, vmin=-1.0, vmax=1.0, origin="lower",
                        interpolation="none")

for ax, title in zip((ax_k, ax_kp1, ax_res), (f"$k$ ({DATE})", f"$k+1$ ({MONTH_APART})", "$k+1 - k$")):
    _clean_image_axis(ax)
    ax.set_title(title)

# Colorbars inset next to their maps, so they match the image height
for ax, img, ticks, label in ((ax_kp1, sic_img, [0, 0.5, 1], "SIC"), (ax_res, res_img, [-1, 0, 1], "SIC change")):
    fig.colorbar(img, cax=ax.inset_axes([1.04, 0, 0.05, 1]), ticks=ticks).set_label(label, labelpad=4)

savefig(fig, "figures", "residual_example", width=TEXTWIDTH_IN, height=2.2)
