from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from main import run_ddpm, run_unet

ddpm_conf = DDPMConfig()
ddpm_conf.schedule = "shifted_cosine"


train_conf = TrainConfig()
train_conf.experiment_name = "ddpm-residual-shifted"

model = run_ddpm(
            dataset=train_conf.dataset,
            forecast_period = train_conf.forecasting_slice,
            timesteps=DDPMConfig.timesteps,
            batch_size=train_conf.batch_size,
            epochs=train_conf.epochs,
            lr=train_conf.lr,
            experiment_name=train_conf.experiment_name,
            sampling=train_conf.sampling,
            train_config=train_conf,
            unet_config=UnetConfig(),
            ddpm_config=ddpm_conf,
        )