from config import UnetConfig, DDPMConfig, TrainConfig
from main import run_ddpm, run_unet

unet_conf = UnetConfig()
train_conf = TrainConfig()
ddpm_conf = DDPMConfig()

ddpm_conf.cosine_shift_res = 64

train_conf.unfolded_loss = True
train_conf.experiment_name = "ddpm-titanfull-unfold-ip"

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