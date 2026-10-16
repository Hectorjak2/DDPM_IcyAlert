import torch
from utils import CARRA2, FashionMNIST, CARRA2Forecast, download_carra2_monthly_data
from utils.device_utils import get_device, device_diagnostics
from utils.helper_functions import download_samples, dump_config_snapshot, schedule_summary, download_conditional_samples, download_unet_samples

from models.models import DatasetChoice
from models.ddpm import DDPM, ConditionalDDPM
from models.unet import Unet, TestUnet, UnetSmall
from models.hec_unet import SimpleUnet, train, count_parameters
from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE

device = get_device()

def train_ddpm(dataset: DatasetChoice,
         timesteps: int,
         batch_size: int,
         epochs: int,
         lr: int,
         forecast_period: slice |None,
         verbose: bool =True,
         download_carra2_data=False,
         experiment_name="default",
         sampling: bool = True,
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

    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    # Build the DDPM model
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
        ) 

    print(schedule_summary(ddpm))

    # Build UNet from config (see config.py for the downsized CARRA2 config).
    unet_cfg = UnetConfig()

    model = SimpleUnet(
        in_channels=4,
        out_channels=unet_cfg.out_channels,
        channels=unet_cfg.channels,
        time_emb_dim=unet_cfg.time_emb_dim,
        dropout=unet_cfg.dropout, 
        groups=unet_cfg.groups,
        n_res_blocks=unet_cfg.n_res_blocks,
    )

    print("Training the model... ")
    ddpm.train(model, dataloader, experiment_name, device, timesteps, epochs=epochs, lr=lr)

    #print("Loading model weights")
    #model.load_weights(path="results/full_conditional_10epochs/final_Unet_CARRA2-siconc_batchsize4_t1000_epochs10_lr0.0003.pth", device=device)

    print("Training complete...")
    if sampling: 
        if TrainConfig.forecasting_slice: 
            test_ds = CARRA2Forecast("siconc", device, area=TrainConfig.dataset, time_slice=TrainConfig.forecasting_slice, batch_dim=False)
            download_conditional_samples(ddpm, model, test_ds, TrainConfig, TrainConfig.number_of_samples)
        else: 
            download_samples(ddpm, model, TrainConfig, TrainConfig.number_of_samples) 
    
    config_path = dump_config_snapshot(TrainConfig, ddpm)
    print(f"Wrote config snapshot to {config_path}")
    print("SCRIPT DONE")

    return model, ddpm, train_ds

def train_unet(dataset: DatasetChoice,
         batch_size: int,
         epochs: int,
         lr: int,
         forecast_period: slice |None,
         verbose: bool =True,
         download_carra2_data=False,
         experiment_name="default",
         sampling: bool = True): 
    
    if not isinstance(dataset, DatasetChoice):
            raise ValueError("Please specify a valid dataset")
        
    if verbose:
        device_diagnostics(device)

    #Downloading the data
    if download_carra2_data:
        download_carra2_monthly_data("dataset", [f"20{i:02d}" for i in range(24)])

    #Defining the dataset and dataloader
    if dataset in (DatasetChoice.CARRA_WEST, DatasetChoice.CARRA_TEST): 
        train_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=TRAINING_SLICE, batch_dim=False, standardize=True)

    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    unet_cfg = UnetConfig()
    model = SimpleUnet(
                in_channels=3,
                out_channels=unet_cfg.out_channels,
                channels=unet_cfg.channels,
                time_emb_dim=unet_cfg.time_emb_dim,
                dropout=unet_cfg.dropout, 
                groups=unet_cfg.groups,
                n_res_blocks=unet_cfg.n_res_blocks,
                land_mask=train_ds.finite_mask,
                device=device
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model = train(model, dataloader, optimizer, device, epochs=epochs, experiment_name=experiment_name)

    test_ds = CARRA2Forecast("siconc", device, area=TrainConfig.dataset, time_slice=TrainConfig.forecasting_slice, batch_dim=False, standardize=True)
    test_ds.mean, test_ds.std = train_ds.mean, train_ds.std  # the model was trained on these
    download_unet_samples(model, test_ds, TrainConfig.number_of_samples, TrainConfig)

    config_path = dump_config_snapshot(TrainConfig, ddpm)
    print(f"Wrote config snapshot to {config_path}")
    
    return model

if __name__ == "__main__":
    if TrainConfig.training_type == "unet": 
        model = train_unet(
            dataset=TrainConfig.dataset,
            forecast_period = TrainConfig.forecasting_slice,
            batch_size=TrainConfig.batch_size,
            epochs=TrainConfig.epochs,
            lr=TrainConfig.lr,
            experiment_name=TrainConfig.experiment_name,
            sampling=TrainConfig.sampling
        )

    elif TrainConfig.training_type == "ddpm":
        model, ddpm, train_ds = train_ddpm(
            dataset=TrainConfig.dataset,
            forecast_period = TrainConfig.forecasting_slice,
            timesteps=DDPMConfig.timesteps,
            batch_size=TrainConfig.batch_size,
            epochs=TrainConfig.epochs,
            lr=TrainConfig.lr,
            experiment_name=TrainConfig.experiment_name,
            sampling=TrainConfig.sampling
        )
