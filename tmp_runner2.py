from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from main import train_ddpm, train_unet
import os 

SAMPLINGROOT = os.environ.get("TEMP_RESULT_STORAGE", "results/samples")

unet_conf = UnetConfig()
unet_conf.attention = False

train_conf = TrainConfig()
train_conf.experiment_name = "sampling_ddpm"
train_conf.output_dir = SAMPLINGROOT

model = train_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling,
            train_config=train_conf,
            unet_config=unet_conf,
            ddpm_config=DDPMConfig,
            load_weights="ddpm-current/final_weights.pth",
            train=False
        )