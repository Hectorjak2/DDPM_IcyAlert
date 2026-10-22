from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from main import train_ddpm, train_unet
import os 

SAMPLINGROOT = os.environ.get("TEMP_RESULT_STORAGE", "results/samples")

unet_conf = UnetConfig()
train_conf = TrainConfig()
ddpm_conf = DDPMConfig()

ddpm_conf.cosine_shift_res = 256

train_conf.experiment_name = "ddpm-shifted_cos_256"
train_conf.output_dir = f"{SAMPLINGROOT}/{train_conf.experiment_name}"

model, ddpm, train_ds = train_ddpm(
            dataset=TrainConfig.dataset,
            forecast_period = TrainConfig.forecasting_slice,
            timesteps=DDPMConfig.timesteps,
            batch_size=TrainConfig.batch_size,
            epochs=TrainConfig.epochs,
            lr=TrainConfig.lr,
            experiment_name=TrainConfig.experiment_name,
            sampling=TrainConfig.sampling, 
            ddpm_config=ddpm_conf,
            train_config=train_conf,
            unet_config=unet_conf
        )
