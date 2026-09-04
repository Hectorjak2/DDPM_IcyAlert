import xarray as xr
import torch
import matplotlib.pyplot as plt

def visualize_sample(sample: torch.Tensor):
    sample = sample.squeeze()  # drop any leading batch/channel dims -> [H, W]
    finite_mask = torch.isfinite(sample)

    plt.figure(figsize=(8, 8))

    # custom colormap from blue (0) to white (1)
    cmap = plt.cm.Blues_r
    cmap.set_bad("gray")

    img = plt.imshow(
        finite_mask * sample,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="lower"
    )

    plt.colorbar(img, label="sea ice concentration", shrink=0.7)
    plt.title("Sea ice concentration (CARRA2-WEST)")
    plt.show()

