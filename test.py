from models.unet import Unet
from models.ddpm import *
from utils import CARRA2, visualize_sample
import torch

#Get the data
train_ds = CARRA2("siconc", time_slice=slice("2012-01", "2013-12"), TEST=True)
sample = train_ds[6]  # [1, 1, 128, 128] - already has the batch dimension
visualize_sample(sample)

#define the model
model = Unet(image_size=128)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.1f}M")

#x = torch.randn(2, 1, 128, 128)
x = sample                          # [1, 1, 128, 128] -> batch size B = 1
t = torch.randint(0, 1000, (x.shape[0],))  # one timestep per image -> [B]

with torch.no_grad():
    out = model(x, t)

print("input :", tuple(x.shape))
print("output:", tuple(out.shape))
assert out.shape == x.shape
