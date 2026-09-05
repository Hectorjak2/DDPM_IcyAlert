import torch
import matplotlib.pyplot as plt
import argparse

from utils import CARRA2, FashionMNIST, visualize_sample, download_carra2_monthly_data
from utils.device_utils import get_device, device_diagnostics
from models.ddpm import DDPM
from models.unet import Unet, UnetSmall

def run(carra=False, 
         fashion=False, 
         verbose=True,
         timesteps=100, 
         batch_size=16,
         epochs=100,
         lr=1e-3,
         download_carra2_data=False
         ):
    
    if not carra and not fashion:
        raise ValueError("Please specify a dataset to use: CARRA2 or FashionMNIST")
    
    device = get_device()
    if verbose:
        device_diagnostics(device)

    #Downloading the data
    if download_carra2_data:
        download_carra2_monthly_data("dataset", [f"20{i:02d}" for i in range(24)])

    #Defining the dataset and dataloader
    if carra: 
        train_ds = CARRA2("siconc", device, TEST=True, batch_dim=False)
    elif fashion:
        train_ds = FashionMNIST(train=True, device=device, batch_dim=False)

    dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    #Defining the DDPM model, and the UNET
    ddpm = DDPM(timesteps=timesteps, device=device, image_size=train_ds[0][0].shape[-1])
    model = Unet() if carra else UnetSmall() if fashion else None
    model.to(device)

    ddpm.train(model, dataloader, device, timesteps, epochs=epochs, lr=lr)

    #Sample and plot 
    #sample = ddpm.sample(model)
    #land_mask = torch.isfinite(train_ds[0][0].cpu())
    #visualize_sample(sample, finite_mask=land_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDPM model on CARRA2 or FashionMNIST dataset")

    # Dataset selection
    parser.add_argument("--carra", action="store_true", help="Use CARRA2 dataset")
    parser.add_argument("--fashion", action="store_true", help="Use FashionMNIST dataset")

    # Training hyperparameters
    parser.add_argument("--timesteps", type=int, default=100, help="Number of diffusion timesteps (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")

    # Data options
    parser.add_argument("--download-carra2", action="store_true", help="Download CARRA2 data before training")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Run the training
    run(
        carra=args.carra,
        fashion=args.fashion,
        verbose=not args.quiet,
        timesteps=args.timesteps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        download_carra2_data=args.download_carra2
    )