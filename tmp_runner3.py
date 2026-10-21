from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from main import train_ddpm, train_unet
import os 

SAMPLINGROOT = os.environ.get("TEMP_RESULT_STORAGE", "results")

unet_conf = UnetConfig()
unet_conf.attention = True

train_conf = TrainConfig()
train_conf.experiment_name = "testing_ddpmatt_samples"
train_conf.output_dir = f"{SAMPLINGROOT}/{train_conf.experiment_name}"
train_conf.number_of_samples = 10

ddpm_conf = DDPMConfig()
ddpm_conf.analysis_sampling = True
ddpm_conf.schedule = "shifted_cosine"

model = train_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            timesteps=DDPMConfig.timesteps,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling,
            train_config=train_conf,
            unet_config=unet_conf,
            ddpm_config=ddpm_conf,
            load_weights="ddpm-current_attention/final_weights.pth",
            train=False
        )