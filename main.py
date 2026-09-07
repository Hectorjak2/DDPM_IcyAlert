import torch
import pickle

from tqdm import tqdm
from utils import CARRA2, FashionMNIST, download_carra2_monthly_data
from utils.device_utils import get_device, device_diagnostics
from models.ddpm import DDPM
from models.unet import Unet, UnetSmall
from config import UnetConfig, DDPMConfig, TrainConfig, SMOKE_TEST, HPC_RUN

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
    # The static land mask is fed to the network as an extra input channel, so the
    # model can distinguish land from open water (both otherwise look like a valid
    # concentration) and land can be masked back out of the generated samples.
    ddpm = DDPM(
        timesteps=timesteps,
        device=device,
        image_size=train_ds[0][0].shape[-1],
        land_mask=train_ds.finite_mask,
        schedule=DDPMConfig.schedule,
        shift_ref_resolution=DDPMConfig.shift_ref_resolution,
    )
    print(
        f"Noise schedule: {ddpm.schedule} "
        f"(image_size={ddpm.image_size}, ref={ddpm.shift_ref_resolution}) -- "
        f"sqrt(alpha_bar) at t=0: {ddpm.alphas_bar[0].sqrt():.4f}, "
        f"t=T: {ddpm.alphas_bar[-1].sqrt():.3e}"
    )

    if carra:
        # Build UNet from config (see config.py for the downsized CARRA2 config).
        # ~10M params instead of 78.7M; see docs/architecture.md for rationale.
        unet_cfg = UnetConfig()
        model = Unet(
            in_channels=unet_cfg.in_channels,
            out_channels=unet_cfg.out_channels,
            base_channels=unet_cfg.base_channels,
            channel_mult=unet_cfg.channel_mult,
            num_res_blocks=unet_cfg.num_res_blocks,
            attention_levels=unet_cfg.attention_levels,
            dropout=unet_cfg.dropout,
            groups=unet_cfg.groups,
        )
    elif fashion:
        model = UnetSmall()

    print("Training the model... ")
    ddpm.train(model, dataloader, device, timesteps, epochs=epochs, lr=lr)

    return model, ddpm, train_ds

def download_samples(ddpm: DDPM, model, n_of_samples: int = 10):
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    samples = []
    for i in tqdm(range(n_of_samples)):
        sample = ddpm.sample(model)
        samples.append(sample.cpu())

        #dump the samples after each iteration
        pickle.dump(samples, open(f"results/{ddpm.output_name}/samples.pkl", "wb"))


if __name__ == "__main__":
    # Run the training
    print("The python script is running ...")

    # Use a config preset; edit config.py to adjust default values.
    # SMOKE_TEST: quick local sanity check (batch_size=2, epochs=2)
    # HPC_RUN: full CARRA2 training (batch_size=16, epochs=100)
    train_cfg = SMOKE_TEST

    model, ddpm, train_ds = run(
        fashion=True,
        timesteps=train_cfg.timesteps if hasattr(train_cfg, 'timesteps') else DDPMConfig.timesteps,
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs,
        lr=train_cfg.lr,
    )

    # Sampling from the trained model (download_samples calls model.eval() itself)
    download_samples(ddpm, model, n_of_samples=1)