import torch
from utils import CARRA2, FashionMNIST, CARRA2Forecast, download_carra2_monthly_data
from utils.device_utils import get_device, device_diagnostics
from utils.helper_functions import download_samples, dump_config_snapshot, schedule_summary, download_conditional_samples

from models.models import DatasetChoice
from models.ddpm import DDPM, ConditionalDDPM
from models.unet import Unet, TestUnet, UnetSmall
from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE

device = get_device()

def run(dataset: DatasetChoice,
         timesteps: int,
         batch_size: int,
         epochs: int,
         lr: int,
         forecast_period: slice |None,
         verbose: bool =True,
         download_carra2_data=False,
         experiment_name="default",
         ) -> tuple[Unet | UnetSmall, DDPM, CARRA2 | FashionMNIST]:
    
    if not isinstance(dataset, DatasetChoice):
        raise ValueError("Please specify a valid dataset")
    
    if verbose:
        device_diagnostics(device)

    #Downloading the data
    if download_carra2_data:
        download_carra2_monthly_data("dataset", [f"20{i:02d}" for i in range(24)])

    #Defining the dataset and dataloader
    if dataset in (DatasetChoice.CARRA_WEST, DatasetChoice.CARRA_TEST): 
        if forecast_period: 
            train_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=TRAINING_SLICE, batch_dim=False)
        else: 
            train_ds = CARRA2("siconc", device, area=dataset, batch_dim=False)

    elif dataset == DatasetChoice.FASHION: 
        train_ds = FashionMNIST(train=True, device=device, batch_dim=False)

    # with GPU compute instead of blocking it (see docs/architecture.md). Datasets now
    # return CPU tensors for exactly this reason: MPS/CUDA tensors generally can't be
    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    #Defining the DDPM model, and the UNET
    # The static land mask is fed to the network as an extra input channel, so the
    # model can distinguish land from open water (both otherwise look like a valid
    # concentration) and land can be masked back out of the generated samples.
    if forecast_period: 
        ddpm = ConditionalDDPM(
                timesteps=timesteps,
                device=device,
                image_size=train_ds[0][0].shape[-1],
                land_mask=train_ds.finite_mask,
                schedule=DDPMConfig.schedule,
                shift_ref_resolution=DDPMConfig.shift_ref_resolution,
                forecast_period=forecast_period
            ) 
    else: 
        ddpm = DDPM(
            timesteps=timesteps,
            device=device,
            image_size=train_ds[0][0].shape[-1],
            land_mask=train_ds.finite_mask,
            schedule=DDPMConfig.schedule,
            shift_ref_resolution=DDPMConfig.shift_ref_resolution,
            forecast_period=forecast_period
        ) 
    print(schedule_summary(ddpm))

    if dataset == DatasetChoice.CARRA_WEST:
        # Build UNet from config (see config.py for the downsized CARRA2 config).
        # ~10M params instead of 78.7M; see docs/architecture.md for rationale.
        unet_cfg = UnetConfig()
        model = Unet(
            in_channels=unet_cfg.in_channels if not forecast_period else 4,
            out_channels=unet_cfg.out_channels,
            base_channels=unet_cfg.base_channels,
            channel_mult=unet_cfg.channel_mult,
            num_res_blocks=unet_cfg.num_res_blocks,
            attention_levels=unet_cfg.attention_levels,
            mid_attention=unet_cfg.mid_attention,
            dropout=unet_cfg.dropout,
            groups=unet_cfg.groups,
        )

    elif dataset == DatasetChoice.CARRA_TEST:
        model = TestUnet(in_channels=2 if not forecast_period else 4)

    elif dataset == DatasetChoice.FASHION:
        model = UnetSmall()

    print("Training the model... ")
    ddpm.train(model, dataloader, experiment_name, device, timesteps, epochs=epochs, lr=lr)
    
    #print("Loading model weights")
    #model.load_weights(path="results/conditional_test_1/final_TestUnet_CARRA2-siconc_batchsize4_t1500_epochs1_lr0.0003.pth" ,device="mps")

    return model, ddpm, train_ds


if __name__ == "__main__":
    # Run the training
    print("The python script is running ...")

    # Use a config preset; edit config.py to adjust default values.
    # SMOKE_TEST: quick local sanity check (batch_size=2, epochs=2)
    # HPC_RUN: full CARRA2 training (batch_size=16, epochs=100)

    model, ddpm, train_ds = run(
        dataset=TrainConfig.dataset,
        forecast_period = TrainConfig.forecasting_slice,
        timesteps=DDPMConfig.timesteps,
        batch_size=TrainConfig.batch_size,
        epochs=TrainConfig.epochs,
        lr=TrainConfig.lr,
        experiment_name=TrainConfig.experiment_name,
    )

    # Sampling from the trained model (download_samples calls model.eval() itself)
    # If forecasting: 
    if TrainConfig.forecasting_slice: 
        test_ds = CARRA2Forecast("siconc", device, area=TrainConfig.dataset, time_slice=TrainConfig.forecasting_slice, batch_dim=False)
        download_conditional_samples(ddpm, model, test_ds, TrainConfig, TrainConfig.number_of_samples)
    else: 
        download_samples(ddpm, model, TrainConfig, TrainConfig.number_of_samples) 

    config_path = dump_config_snapshot(TrainConfig, ddpm)
    print(f"Wrote config snapshot to {config_path}")

    print("SCRIPT DONE")