from config import UnetConfig, DDPMConfig, TrainConfig
from main import run_ddpm, run_unet

unet_conf = UnetConfig()
train_conf = TrainConfig()
ddpm_conf = DDPMConfig()

train_conf.training = False 

train_conf.weights = "ddpm-titanfull-shif64/final_weights.pth"
ddpm_conf.cosine_shift_res = 64
train_conf.experiment_name = "ddpm-corrected-land-64"

model, ddpm, train_ds = run_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            timesteps=ddpm_conf.timesteps,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling, 
            train=train_conf.training,
            load_weights=train_conf.weights,
            ddpm_config=ddpm_conf,
            train_config=train_conf,
            unet_config=unet_conf
        )


train_conf.weights = "ddpm-titanfull-shif64-ip/final_weights.pth"
ddpm_conf.cosine_shift_res = 64
train_conf.experiment_name = "ddpm-corrected-land-64-ip"

model, ddpm, train_ds = run_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            timesteps=ddpm_conf.timesteps,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling, 
            train=train_conf.training,
            load_weights=train_conf.weights,
            ddpm_config=ddpm_conf,
            train_config=train_conf,
            unet_config=unet_conf
        )


train_conf.weights = "ddpm-titanfull-shif128-ip/final_weights.pth"
ddpm_conf.cosine_shift_res = 128
train_conf.experiment_name = "ddpm-corrected-land-128-ip"

model, ddpm, train_ds = run_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            timesteps=ddpm_conf.timesteps,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling, 
            train=train_conf.training,
            load_weights=train_conf.weights,
            ddpm_config=ddpm_conf,
            train_config=train_conf,
            unet_config=unet_conf
        )


train_conf.weights = "ddpm-titanfull-shif256-ip/final_weights.pth"
ddpm_conf.cosine_shift_res = 256
train_conf.experiment_name = "ddpm-corrected-land-256-ip"

model, ddpm, train_ds = run_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            timesteps=ddpm_conf.timesteps,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling, 
            train=train_conf.training,
            load_weights=train_conf.weights,
            ddpm_config=ddpm_conf,
            train_config=train_conf,
            unet_config=unet_conf
        )