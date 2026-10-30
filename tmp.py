from utils.dataloader import CARRA2Forecast
from models.models import DatasetChoice
TRAINING_SLICE = slice("2000-01-01", "2019-10-31")

ds = CARRA2Forecast("siconc", "cpu", area=DatasetChoice.CARRA_CROPPED, time_slice=TRAINING_SLICE, batch_dim=False)
water = ds.finite_mask[0].bool()
s = ss = n = 0
for i in range(len(ds)):
    x, _ = ds[i]
    r = (x[0] - x[1])[water]
    s += r.sum(); ss += (r ** 2).sum(); n += r.numel()
print((ss / n - (s / n) ** 2).sqrt().item())
