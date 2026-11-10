import torch
from utils import CARRA2, FashionMNIST, CARRA2Forecast, download_carra2_monthly_data
from utils.device_utils import get_device, device_diagnostics
from utils.helper_functions import dump_config_snapshot, schedule_summary
from utils.sampling import download_samples, analysis_sampling, forecast, forecast_dataset, DDPMForecaster, UnetForecaster
from pathlib import Path

from models.models import DatasetChoice
from models.ddpm import DDPM, ConditionalDDPM
from models.unet import SimpleUnet, train_network, count_parameters
from config import UnetConfig, DDPMConfig, TrainConfig

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
            train_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=train_config.training_slice, batch_dim=False)
        else: 
            train_ds = CARRA2("siconc", device, area=dataset, batch_dim=False)

    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    # Held-out validation block (TrainConfig.validation_slice), scored with the training loss after
    # every epoch. Its own residual_scale is None (not cached), which is fine: the DDPM uses the
    # training set's.
    val_dataloader = None
    if forecast_period and train_config.validation_slice is not None:
        # field_stats only skips recomputing a mean/std the DDPM ([-1, 1] scaling) does not use.
        val_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=train_config.validation_slice, batch_dim=False,
                                field_stats=(train_ds.mean, train_ds.std))
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        print(f"Train: {len(train_ds)} windows, last target {train_ds.get_dates(len(train_ds) - 1)[0]} | "
              f"Val: {len(val_ds)} windows, first k-1 {val_ds.get_dates(0)[-1]}")

    # Build the DDPM model
    if forecast_period: 
        ddpm = ConditionalDDPM(
                timesteps=timesteps,
                device=device,
                image_size=train_ds[0][0].shape[-1],
                land_mask=train_ds.finite_mask,
                schedule=ddpm_config.schedule,
                cosine_shift_res=ddpm_config.cosine_shift_res,
                cosine_shift_res_t0=ddpm_config.cosine_shift_res_t0,
                forecast_period=forecast_period,
                batch_size=train_config.batch_size,
                unfolded=train_config.unfolded_loss,
                dose=train_config.dose_loss,
                residual_std=train_ds.residual_scale,
                clamp_x0=ddpm_config.clamp_x0,
                residual_context=ddpm_config.residual_context,
                sampler=ddpm_config.sampler,
                ddim_steps=ddpm_config.ddim_steps,
                ddim_eta=ddpm_config.ddim_eta,
                input_perturbation=train_config.input_perturbation,
            )
    else: 
        ddpm = DDPM(
            timesteps=timesteps,
            device=device,
            image_size=train_ds[0][0].shape[-1],
            land_mask=train_ds.finite_mask,
            schedule=ddpm_config.schedule,
            cosine_shift_res=ddpm_config.cosine_shift_res,
            cosine_shift_res_t0=ddpm_config.cosine_shift_res_t0,
        )

    print(schedule_summary(ddpm))

    # Build UNet from config (see config.py for the downsized CARRA2 config).
    model = SimpleUnet(
        # The conditional DDPM's input width depends on DDPMConfig.residual_context.
        in_channels=ddpm.in_channels if forecast_period else unet_config.in_channels,
        out_channels=unet_config.out_channels,
        channels=unet_config.channels,
        time_emb_dim=unet_config.time_emb_dim,
        dropout=unet_config.dropout, 
        groups=unet_config.groups,
        n_res_blocks=unet_config.n_res_blocks,
        attention=unet_config.attention,
    )

    print(f"Writing config snapshot")
    dump_config_snapshot(train_config, ddpm, ddpm_config=ddpm_config, residual_std=getattr(train_ds, "residual_scale", None))
    
    if load_weights: 
        print(f"Loading model weights: {load_weights}")
        model.load_weights(path=f"{train_config.output_dir}/{load_weights}", device=device)

    if train: 
        print("Training the model... ")
        ddpm.train(model, dataloader, experiment_name, device, timesteps, epochs=epochs, lr=lr, train_config=train_config, val_dataloader=val_dataloader)
        print("Training complete...")
        
    if sampling: 
        print("Sampling the model...")
        if train_config.analysis_sampling: 
            print("Analysis sampling")
            test_ds = CARRA2Forecast("siconc", device, area=train_config.dataset, time_slice=train_config.forecasting_slice, batch_dim=False)
            analysis_sampling(ddpm, model, test_ds, train_config, train_config.number_of_samples)
        else: 
            if train_config.forecasting_slice: 
                # Disable dropout for inference. Left in train mode the residual blocks inject
                # noise at every one of the T reverse steps, which wrecks the sample.
                model.eval()
                forecast(DDPMForecaster(ddpm, model), forecast_dataset(train_config), [0], train_config.number_of_samples,
                         train_config.sample_seed, Path(train_config.samples_dir) / train_config.experiment_name / "samples.pkl")
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
        train_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=train_config.training_slice, batch_dim=False, ddpm=False)

    dataloader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    # Held-out validation block, as in run_ddpm. Z-scored with the training mean and std, never
    # its own: the model only knows the training statistics, as UnetForecaster does at test time.
    val_dataloader = None
    if train_config.validation_slice is not None:
        val_ds = CARRA2Forecast("siconc", device, area=dataset, time_slice=train_config.validation_slice, batch_dim=False,
                                ddpm=False, field_stats=(train_ds.mean, train_ds.std))
        val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        print(f"Train: {len(train_ds)} windows, last target {train_ds.get_dates(len(train_ds) - 1)[0]} | "
              f"Val: {len(val_ds)} windows, first k-1 {val_ds.get_dates(0)[-1]}")

    # Seeds the global RNG, so the seed fixes the initial weights and also the batch order and
    # dropout masks during training (a deep ensemble member, Lakshminarayanan et al. 2017).
    if initialization_seed is not None:
        torch.manual_seed(initialization_seed)

    model = SimpleUnet(
                in_channels=unet_config.in_channels-1,
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
    dump_config_snapshot(train_config, unet_config=unet_config, ddpm=model, unet=True, residual_std=train_ds.residual_scale)
    
    if load_weights: 
        print(f"Loading model weights: {load_weights}")
        model.load_weights(path=f"{load_weights}", device=device)

    if train: 
        print("Training the model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model = train_network(model, dataloader, optimizer, train_ds.residual_scale, device, epochs=epochs, experiment_name=experiment_name, output_dir=train_config.output_dir, val_dataloader=val_dataloader)

    if sampling: 
        print("Sampling the model...")
        model.eval()
        # The forecaster standardizes with the training statistics the model was trained on.
        forecast(UnetForecaster(model, train_ds.mean, train_ds.std, train_ds.residual_scale), forecast_dataset(train_config), [0], train_config.number_of_samples,
                 train_config.sample_seed, Path(train_config.samples_dir) / train_config.experiment_name / "samples.pkl")

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
            train=TrainConfig.training,
            load_weights=TrainConfig.weights,
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
            train=TrainConfig.training,
            load_weights=TrainConfig.weights,
            ddpm_config=DDPMConfig(),
            train_config=TrainConfig(),
            unet_config=UnetConfig()
        )
