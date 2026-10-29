from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from main import run_ddpm, run_unet
import os 

SAMPLINGROOT = os.environ.get("TEMP_RESULT_STORAGE", "results")

unet_conf = UnetConfig()

train_conf = TrainConfig()
train_conf.experiment_name = "unet-rollout-m0v2"
train_conf.output_dir = f"{SAMPLINGROOT}"
train_conf.lead_months = 6


model = run_unet(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling,
            train_config=train_conf,
            unet_config=unet_conf,
            train=False,
            load_weights="ensembles/member_0/final_weights.pth",
        )