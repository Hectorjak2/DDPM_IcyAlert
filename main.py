import torch
import pickle

from tqdm import tqdm
from utils import CARRA2, FashionMNIST, download_carra2_monthly_data
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
         ) -> tuple[Unet | UnetSmall, DDPM, CARRA2 | FashionMNIST]:
    
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
        train_ds = CARRA2("siconc", device, batch_dim=False)
    elif fashion:
        train_ds = FashionMNIST(train=True, device=device, batch_dim=False)

    dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    #Defining the DDPM model, and the UNET
    ddpm = DDPM(timesteps=timesteps, device=device, image_size=train_ds[0][0].shape[-1])

    if carra:
        # Smaller UNet: ~10M params instead of 78.7M
        model = Unet(
            base_channels=64,           # Reduced from 128
            channel_mult=(1, 2, 2, 2),  # Reduced from (1, 2, 2, 2, 4)
            num_res_blocks=1,           # Reduced from 2
            attention_levels=()         # No attention
        )
    elif fashion:
        model = UnetSmall()

    print("Training the model... ")
    ddpm.train(model, dataloader, device, timesteps, epochs=epochs, lr=lr)

    return model, ddpm, train_ds

def download_samples(ddpm: DDPM, n_of_samples: int = 10): 
    print("Sampling from the trained model ...")
    samples = []
    for i in tqdm(range(n_of_samples)): 
        sample = ddpm.sample(model)
        samples.append(sample)

    pickle.dump(samples, open(f"results/{ddpm.output_name}/samples.pkl", "wb"))


if __name__ == "__main__":
    # Run the training
    print("The python script is running ...")
    
    model, ddpm, train_ds = run(
        carra=True,
        timesteps=1000,
        batch_size=2,
        epochs=2,
    )

    #Sampling from the trained model 
    download_samples(ddpm, n_of_samples=10)