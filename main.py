import torch
from utils import CARRA2, FashionMNIST, CARRA2Forecast, download_carra2_monthly_data
from utils.device_utils import get_device, device_diagnostics
from utils.helper_functions import download_samples, dump_config_snapshot, schedule_summary, download_conditional_samples, download_unet_samples, analysis_sampling

from models.models import DatasetChoice
from models.ddpm import DDPM, ConditionalDDPM
from models.unet import SimpleUnet, train_network, count_parameters
from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE

device = get_device()

def run_ddpm(dataset: DatasetChoice,
         timesteps: int,
         batch_size: int,
         epochs: int,
         lr: int,
         forecast_period: slice |None,
         verbose: bool =True,
         download_carra2_data=False,
         experiment_name="default",
         sampling: bool = True,
         train: bool = True, 
         load_weights: str = None,
         ddpm_config: DDPMConfig = DDPMConfig,
         train_config: TrainConfig = TrainConfig,
         unet_config: UnetConfig = UnetConfig
         ) -> tuple[SimpleUnet, DDPM, CARRA2 | FashionMNIST]:
    
    if not isinstance(dataset, DatasetChoice):
        raise ValueError("Please specify a valid dataset")
    
    if verbose:
        device_diagnostics(device)

    #Downloading the data
    if download_carra2_data:
        download_carra2_monthly_data("dataset", [f"20{i:02d}" for i in range(24)])

    #Defining the dataset and dataloader
    if dataset in (DatasetChoice.CARRA_WEST, DatasetChoice.CARRA_TEST, DatasetChoice.CARRA_CROPPED): 
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
                schedule=ddpm_config.schedule,
                forecast_period=forecast_period
            ) 
    else: 
        ddpm = DDPM(
            timesteps=timesteps,
            device=device,
            image_size=train_ds[0][0].shape[-1],
            land_mask=train_ds.finite_mask,
            schedule=ddpm_config.schedule,
        ) 

    print(schedule_summary(ddpm))

    # Build UNet from config (see config.py for the downsized CARRA2 config).
    model = SimpleUnet(
        in_channels=4,
        out_channels=unet_config.out_channels,
        channels=unet_config.channels,
        time_emb_dim=unet_config.time_emb_dim,
        dropout=unet_config.dropout, 
        groups=unet_config.groups,
        n_res_blocks=unet_config.n_res_blocks,
        attention=unet_config.attention,
    )

    print(f"Writing config snapshot")
    dump_config_snapshot(train_config, ddpm, ddpm_config=ddpm_config)
    
    if load_weights: 
        print(f"Loading model weights: {load_weights}")
        model.load_weights(path=f"{train_config.output_dir}/{load_weights}", device=device)

    if train: 
        print("Training the model... ")
        ddpm.train(model, dataloader, experiment_name, device, timesteps, epochs=epochs, lr=lr, train_config=train_config)
        print("Training complete...")
        
    if sampling: 
        print("Sampling the model...")
        if ddpm_config.analysis_sampling: 
            print("Analysis sampling")
            test_ds = CARRA2Forecast("siconc", device, area=train_config.dataset, time_slice=train_config.forecasting_slice, batch_dim=False)
            analysis_sampling(ddpm, model, test_ds, train_config, train_config.number_of_samples)
        else: 
            if train_config.forecasting_slice: 
                test_ds = CARRA2Forecast("siconc", device, area=train_config.dataset, time_slice=train_config.forecasting_slice, batch_dim=False)
                download_conditional_samples(ddpm, model, test_ds, train_config, train_config.number_of_samples)
            else: 
                download_samples(ddpm, model, train_config, train_config.number_of_samples) 

    print("SCRIPT DONE")

    return model, ddpm, train_ds

def run_unet(dataset: DatasetChoice,
         batch_size: int,
         epochs: int,
         lr: int,
         forecast_period: slice |None,
         verbose: bool =True,
         download_carra2_data=False,
         experiment_name="default",
         sampling: bool = True,
         train: bool = True, 
         load_weights: str = None,
         train_config: TrainConfig = TrainConfig,
         unet_config: UnetConfig = UnetConfig,
         initialization_seed: int | None = None):
    
    if not isinstance(dataset, DatasetChoice):
            raise ValueError("Please specify a valid dataset")
        
    if verbose:
        device_diagnostics(device)

    #Downloading the data
    if download_carra2_data:
        download_carra2_monthly_data("dataset", [f"20{i:02d}" for i in range(24)])

    #Defining the dataset and dataloader
    if dataset in (DatasetChoice.CARRA_WEST, DatasetChoice.CARRA_TEST, DatasetChoice.CARRA_CROPPED): 
        train_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=TRAINING_SLICE, batch_dim=False, standardize=True)

    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    # Seeds the global RNG, so the seed fixes the initial weights and also the batch order and
    # dropout masks during training (a deep ensemble member, Lakshminarayanan et al. 2017).
    if initialization_seed is not None:
        torch.manual_seed(initialization_seed)

    model = SimpleUnet(
                in_channels=3,
                out_channels=unet_config.out_channels,
                channels=unet_config.channels,
                time_emb_dim=unet_config.time_emb_dim,
                dropout=unet_config.dropout, 
                groups=unet_config.groups,
                n_res_blocks=unet_config.n_res_blocks,
                land_mask=train_ds.finite_mask,
                device=device,
                attention=unet_config.attention,
            )
    print(f"Writing config snapshot")
    dump_config_snapshot(train_config, unet_config=unet_config, ddpm=model, unet=True)
    
    if load_weights: 
        print(f"Loading model weights: {load_weights}")
        model.load_weights(path=f"{train_config.output_dir}/{load_weights}", device=device)

    if train: 
        print("Training the model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = train_network(model, dataloader, optimizer, device, epochs=epochs, experiment_name=experiment_name, output_dir=train_config.output_dir)

    if sampling: 
        print("Sampling the model...")
        test_ds = CARRA2Forecast("siconc", device, area=train_config.dataset, time_slice=train_config.forecasting_slice, batch_dim=False, standardize=True)
        test_ds.mean, test_ds.std = train_ds.mean, train_ds.std  # the model was trained on these
        download_unet_samples(model, test_ds, train_config.number_of_samples, train_config)

    print("SCRIPT DONE ")
    return model

if __name__ == "__main__":
    if TrainConfig.training_type == "unet": 
        model = run_unet(
            dataset=TrainConfig.dataset,
            forecast_period = TrainConfig.forecasting_slice,
            batch_size=TrainConfig.batch_size,
            epochs=TrainConfig.epochs,
            lr=TrainConfig.lr,
            experiment_name=TrainConfig.experiment_name,
            sampling=TrainConfig.sampling,
            train_config=TrainConfig(),
            unet_config=UnetConfig()
        )

    elif TrainConfig.training_type == "ddpm":
        model, ddpm, train_ds = run_ddpm(
            dataset=TrainConfig.dataset,
            forecast_period = TrainConfig.forecasting_slice,
            timesteps=DDPMConfig.timesteps,
            batch_size=TrainConfig.batch_size,
            epochs=TrainConfig.epochs,
            lr=TrainConfig.lr,
            experiment_name=TrainConfig.experiment_name,
            sampling=TrainConfig.sampling, 
            ddpm_config=DDPMConfig(),
            train_config=TrainConfig(),
            unet_config=UnetConfig()
        )
