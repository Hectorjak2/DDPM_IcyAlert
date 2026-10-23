from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from main import train_ddpm, train_unet
import os 

SAMPLINGROOT = os.environ.get("TEMP_RESULT_STORAGE", "results")

unet_conf = UnetConfig()

train_conf = TrainConfig()
train_conf.experiment_name = "unet-residual"
train_conf.output_dir = f"{SAMPLINGROOT}/{train_conf.experiment_name}"

ddpm_conf = DDPMConfig()
ddpm_conf.analysis_sampling = True
ddpm_conf.schedule = "cosine"

model = train_unet(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling,
            train_config=train_conf,
            unet_config=unet_conf,
            train=True
        )