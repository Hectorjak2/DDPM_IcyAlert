import xarray as xr
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from typing import Literal

def visualize_sic_sample(sample: torch.Tensor, finite_mask: torch.Tensor = None):
    sample = sample.cpu().squeeze()  # drop any leading batch/channel dims -> [H, W]

    if finite_mask is None:
        finite_mask = torch.isfinite(sample)
    else:
        finite_mask = finite_mask.cpu().squeeze()

    sample[finite_mask == 0] = float("nan")  # set land pixels to NaN for visualization

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


def visualize_fashion(sample: torch.Tensor, title: str = "Fashion MNIST Sample"):
    """
    Simple visualization for Fashion MNIST images.

    Args:
        sample: Tensor of shape [H, W] or [C, H, W] or [B, C, H, W]
        title: Title for the plot
    """
    sample = sample.cpu().squeeze()  # drop any leading batch/channel dims -> [H, W]

    plt.figure(figsize=(4, 4))
    plt.imshow(sample, cmap='gray', vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_samples_grid(
    pickle_path: str,
    dataset_type: Literal["fashion", "carra"] = "fashion",
    grid_size: int = 4,
    figsize_per_sample: tuple = (2, 2),
    finite_mask: torch.Tensor = None
):
    """
    Load samples from a pickle file and plot them in a grid.

    Args:
        pickle_path: Path to the pickle file containing list of samples
        dataset_type: Type of dataset - "fashion" for Fashion MNIST or "carra" for CARRA2
        grid_size: Number of samples per row/column (default: 4x4 grid)
        figsize_per_sample: Figure size per individual sample (default: 2x2)
        finite_mask: Optional mask for carra dataset to mark land pixels

    Returns:
        None (displays plot)
    """
    # Load samples from pickle file
    with open(pickle_path, 'rb') as f:
        samples = pickle.load(f)

    # Ensure samples is a list
    if isinstance(samples, torch.Tensor):
        samples = samples.tolist() if samples.dim() == 1 else [samples[i] for i in range(samples.shape[0])]
    elif not isinstance(samples, list):
        samples = list(samples)

    # Limit to grid_size^2 samples
    num_samples = min(len(samples), grid_size ** 2)
    samples = samples[:num_samples]

    # Calculate figure size
    fig_w = grid_size * figsize_per_sample[0]
    fig_h = grid_size * figsize_per_sample[1]

    # Create grid of subplots
    fig, axes = plt.subplots(
        grid_size, grid_size,
        figsize=(fig_w, fig_h),
        squeeze=False
    )

    # Plot each sample
    for idx, sample in enumerate(samples):
        row = idx // grid_size
        col = idx % grid_size
        ax = axes[row, col]

        # Convert to tensor if needed
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)

        sample = sample.cpu().squeeze()

        # Plot based on dataset type
        if dataset_type.lower() == "fashion":
            im = ax.imshow(sample, cmap='gray', vmin=0.0, vmax=1.0)
            ax.set_title(f"Sample {idx + 1}", fontsize=8)
            ax.axis('off')

        elif dataset_type.lower() == "carra":
            if finite_mask is None:
                mask = torch.isfinite(sample)
            else:
                mask = finite_mask.cpu().squeeze() if isinstance(finite_mask, torch.Tensor) else torch.tensor(finite_mask)

            sample_masked = sample.clone()
            sample_masked[mask == 0] = float("nan")

            cmap = plt.cm.Blues_r
            cmap.set_bad("gray")

            im = ax.imshow(
                mask * sample_masked,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                aspect="auto",
                origin="lower"
            )
            ax.set_title(f"Sample {idx + 1}", fontsize=8)
            ax.axis('off')

    # Hide remaining subplots if fewer samples than grid
    for idx in range(num_samples, grid_size ** 2):
        row = idx // grid_size
        col = idx % grid_size
        axes[row, col].set_visible(False)

    plt.suptitle(f"{dataset_type.upper()} Generated Samples ({num_samples} samples)", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import os 
    os.chdir("..")
    finite_mask = torch.load("data/finite_WEST.pt")

    plot_samples_grid("results/Auto_exper222/samples.pkl", "carra", grid_size=5, finite_mask=finite_mask)