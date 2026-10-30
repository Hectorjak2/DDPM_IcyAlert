"""
All sampling from trained models.

A Forecaster maps a context (k, k-1) to a forecast of k+1, both in SIC [0, 1] with land NaN.
Each one converts to and from its own model's range internally, so forecast() never needs to
know which model it drives, and a prediction can be fed straight back in as the next context
(rollout). A single forecast is an ensemble of one member, and a one-month forecast is a
rollout of length one, so forecast() covers single runs, ensembles and multi-month leads.

forecast() writes (samples, contexts, targets, times, member_seeds), all in SIC [0, 1], land NaN:
    samples[i]  [n_members, lead_months, H, W]
    targets[i]  [lead_months, H, W]    = (k+1, ..., k+lead_months)
    contexts[i] [2, H, W]              = (k, k-1)
    times[i]    (k+1, ..., k+lead_months, k, k-1) dates
With lead_months=1 this is the ensemble_samples.pkl layout plotting/ensemble_evaluation.py reads.
"""
import pickle
import random
from pathlib import Path

import torch
from tqdm import tqdm

from config import UnetConfig, DDPMConfig, TrainConfig, TRAINING_SLICE
from models.ddpm import DDPM, ConditionalDDPM
from models.unet import SimpleUnet
from utils.dataloader import CARRA2Forecast, to_model_range, to_data_range
from utils.device_utils import get_device

device = get_device()


class Forecaster:
    """Forecast of k+1 from (k, k-1). Subclasses set encode/decode to their model's range."""

    def encode(self, sic: torch.Tensor) -> torch.Tensor:
        """SIC [0, 1] -> the model's range. Land stays NaN."""
        raise NotImplementedError

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """The model's range -> SIC [0, 1]."""
        raise NotImplementedError

    def predict(self, context: torch.Tensor, month: int, member: int) -> torch.Tensor:
        """context [2, H, W] = (k, k-1) in SIC, month (0-11) of k+1 -> [1, H, W] SIC, land NaN."""
        raise NotImplementedError


class UnetForecaster(Forecaster):
    """
    The residual UNet baseline, in standardized units with the training mean and std.

    models is one SimpleUnet, used for every member (a single forecast, or MC dropout when left
    in train()), or a dict member seed -> SimpleUnet (a deep ensemble).
    """

    def __init__(self, models: SimpleUnet | dict[int, SimpleUnet], mean: float, std: float):
        self.models = models
        self.mean, self.std = float(mean), float(std)

    def encode(self, sic):
        return (sic - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean

    def predict(self, context, month, member):
        model = self.models[member] if isinstance(self.models, dict) else self.models
        # Same input as training: NaN -> 0 (= the training mean), plus the land mask channel.
        x = torch.cat([self.encode(context).nan_to_num(0.0).unsqueeze(0),
                       model.mask_channel.expand(1, -1, -1, -1)], dim=1)
        t = torch.zeros(1, device=context.device)
        month = torch.tensor([month], device=context.device)
        # The network predicts the residual k+1 - k; add day k back, as in train_network.
        pred = x[:, 0:1] + model(x, t, month)  # [1, 1, H, W], standardized
        # Back to SIC, clamped like the DDPM output, land set to NaN.
        return self.decode(pred).clamp(0.0, 1.0).masked_fill(~model.water_mask, float("nan"))[0]


class DDPMForecaster(Forecaster):
    """The conditional DDPM in model range [-1, 1]. One call = one reverse process; the spread
    between members comes from the initial noise, which forecast() seeds per member."""

    def __init__(self, ddpm: ConditionalDDPM, model):
        self.ddpm = ddpm
        self.model = model

    def encode(self, sic):
        return to_model_range(sic) #x * 2.0 - 1.0

    def decode(self, x):
        return to_data_range(x) #(x + 1.0) / 2.0

    def predict(self, context, month, member):
        # ddpm.sample already returns SIC [0, 1] with land NaN.
        return self.ddpm.sample(context=self.encode(context), month=month, model=self.model)[0]


def build_unet(in_channels: int, weights: str, unet_cfg: UnetConfig, land_mask: torch.Tensor = None) -> SimpleUnet:
    """SimpleUnet as built in main.py, with the weights in `weights` loaded. in_channels is 3 for the
    UNet baseline (k, k-1, land mask) and 4 for the DDPM (x_t, k, k-1, land mask)."""
    model = SimpleUnet(
        in_channels=in_channels,
        out_channels=unet_cfg.out_channels,
        channels=unet_cfg.channels,
        time_emb_dim=unet_cfg.time_emb_dim,
        dropout=unet_cfg.dropout,
        groups=unet_cfg.groups,
        n_res_blocks=unet_cfg.n_res_blocks,
        land_mask=land_mask,
        device=device,
        attention=unet_cfg.attention,
    )
    model.load_weights(path=weights, device=device)
    return model


def forecast_dataset(train_cfg: TrainConfig) -> CARRA2Forecast:
    """The test set every forecaster is scored on. Its scaling does not matter, forecast() converts
    it to SIC; lead_months decides which anchors have ground truth far enough ahead."""
    return CARRA2Forecast("siconc", device, area=train_cfg.dataset, time_slice=train_cfg.forecasting_slice,
                          batch_dim=False, lead_months=train_cfg.lead_months)


def unet_training_set(train_cfg: TrainConfig) -> CARRA2Forecast:
    """The standardized training set, for the mean, std and land mask the residual UNet was trained with."""
    return CARRA2Forecast("siconc", device, area=train_cfg.dataset, time_slice=TRAINING_SLICE, batch_dim=False, standardize=True)


def dropout_forecaster(weights: str, train_cfg: TrainConfig, unet_cfg: UnetConfig) -> UnetForecaster:
    """MC-dropout ensemble of the residual UNet baseline in `weights`."""
    assert unet_cfg.dropout > 0, "MC dropout needs UnetConfig.dropout > 0"
    train_ds = unet_training_set(train_cfg)
    model = build_unet(3, weights, unet_cfg, land_mask=train_ds.finite_mask)
    # train() only switches dropout on: SimpleUnet normalises with GroupNorm, which has no running statistics.
    model.train()
    return UnetForecaster(model, train_ds.mean, train_ds.std)


def deep_forecaster(members_dir: str, member_seeds: list[int], train_cfg: TrainConfig, unet_cfg: UnetConfig) -> UnetForecaster:
    """Deep ensemble of the residual UNets that train_ensemble wrote to members_dir/member_{seed}/final_weights.pth."""
    weights = {seed: Path(members_dir) / f"member_{seed}" / "final_weights.pth" for seed in member_seeds}
    missing = [seed for seed, path in weights.items() if not path.exists()]
    assert not missing, f"members not trained yet (run train_ensemble): {missing}"

    train_ds = unet_training_set(train_cfg)
    models = {seed: build_unet(3, str(path), unet_cfg, land_mask=train_ds.finite_mask) for seed, path in weights.items()}
    # Dropout off: here the spread comes from the members' different initial weights and training runs.
    for model in models.values():
        model.eval()
    return UnetForecaster(models, train_ds.mean, train_ds.std)


def ddpm_forecaster(weights: str, test_ds: CARRA2Forecast, train_cfg: TrainConfig, unet_cfg: UnetConfig, ddpm_cfg: DDPMConfig) -> DDPMForecaster:
    """The conditional DDPM in `weights`. ddpm_cfg must match the run that produced the weights
    (schedule, cosine_shift_res, timesteps)."""
    ddpm = ConditionalDDPM(
        timesteps=ddpm_cfg.timesteps,
        device=device,
        image_size=test_ds[0][0].shape[-1],
        land_mask=test_ds.finite_mask,
        schedule=ddpm_cfg.schedule,
        forecast_period=train_cfg.forecasting_slice,
    )
    model = build_unet(4, weights, unet_cfg)
    # Dropout off: here the spread comes from the reverse process, not the network.
    model.eval()
    return DDPMForecaster(ddpm, model)


def draw_dates(test_ds, n_dates: int, sample_seed: int) -> list[int]:
    """Indices of the test dates to forecast. Own RNG so the dates depend only on the seed,
    not on earlier use of the global one; the same seed gives the same dates for every model."""
    date_rng = random.Random(sample_seed)
    return [date_rng.randrange(len(test_ds)) for _ in range(n_dates)]


def rollout(forecaster: Forecaster, context: torch.Tensor, month: int, member: int, lead_months: int) -> torch.Tensor:
    """Autoregressive forecast of k+1, ..., k+lead_months from context (k, k-1), month of k+1.
    Each prediction becomes day k of the next step. Returns [lead_months, H, W] SIC, land NaN."""
    window = context
    preds = []
    for lead in range(lead_months):
        pred = forecaster.predict(window, (month + lead) % 12, member)
        preds.append(pred)
        window = torch.cat([pred, window[:1]], dim=0)
    return torch.cat(preds, dim=0)


@torch.no_grad()
def forecast(forecaster: Forecaster, test_ds: CARRA2Forecast, member_seeds: list[int], n_dates: int, sample_seed: int, out_path: Path):
    """
    Roll out test_ds.lead_months months from n_dates test dates, once per seed in member_seeds,
    and pickle the result to out_path after every date (see the module docstring for the layout).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    contexts = []
    targets = []
    times = []
    for idx in tqdm(draw_dates(test_ds, n_dates, sample_seed)):
        context, target = test_ds.get_rollout(idx)  # (k, k-1), (k+1, ..., k+lead_months)
        context, target = test_ds.to_sic(context), test_ds.to_sic(target)
        month = test_ds.get_month(idx)               # month of k+1 (0-11)
        times.append(test_ds.get_dates(idx))

        members = []
        for seed in member_seeds:
            # One seed per trajectory, not per step: member m's whole rollout is reproducible on
            # its own and distinct from the others, and its first lead equals a lead_months=1 run.
            torch.manual_seed(seed)
            members.append(rollout(forecaster, context, month, seed, test_ds.lead_months).cpu())

        samples.append(torch.stack(members))  # [n_members, lead_months, H, W]
        targets.append(target.cpu())
        contexts.append(context.cpu())

        # dump the samples after each date, an ensemble date is expensive
        pickle.dump((samples, contexts, targets, times, member_seeds), open(out_path, "wb"))

    print(f"Wrote {len(samples)} dates x {len(member_seeds)} members x {test_ds.lead_months} leads to {out_path}")


def download_samples(ddpm: DDPM, model, train_cfg: TrainConfig, n_of_samples: int = 10):
    """Unconditional samples (no context, so no Forecaster)."""
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    Path(f"{train_cfg.samples_dir}/{train_cfg.experiment_name}").mkdir(parents=True, exist_ok=True)
    samples = []
    for i in tqdm(range(n_of_samples)):
        sample = ddpm.sample(model)
        samples.append(sample.cpu())

        #dump the samples after each iteration
        pickle.dump(samples, open(f"{train_cfg.samples_dir}/{train_cfg.experiment_name}/{train_cfg.experiment_name}_samples.pkl", "wb"))


def analysis_sampling(ddpm: ConditionalDDPM, model, test_ds, train_cfg: TrainConfig, n_of_samples: int = 10):
    """Diagnostics of the conditional DDPM, written to loss_stats.pkl (read by plotting/sample_analysis.py):
    the running loss during sampling with the true, a wrong and a zero context, and the loss when
    sampling from intermediate t. test_ds is in model range [-1, 1] (not standardized)."""
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    outdir = f"{train_cfg.samples_dir}/{train_cfg.experiment_name}"

    Path(outdir).mkdir(parents=True, exist_ok=True)

    T = ddpm.timesteps
    total_varying_loss = []
    total_sampling_loss = []
    total_no_context_loss = []
    total_zero_context_loss = []

    # Own RNG so the drawn dates depend only on the seed, not on earlier use of the global one.
    # Not draw_dates: the wrong-context draws are interleaved with the date draws.
    date_rng = random.Random(train_cfg.sample_seed)
    for i in tqdm(range(n_of_samples)):
        idx = date_rng.randrange(len(test_ds))
        sample, month = test_ds[idx]   # [3, H, W] = (k+1, k, k-1); month of k+1 (0-11)
        context = sample[1:]           # (k, k-1) — what you condition on
        target  = sample[:1]           # k+1 — ground truth, for comparison only

        #Retrieving a wrong context to test how reliant forecast is on the context
        wrong_idx = date_rng.randrange(len(test_ds))
        while abs(wrong_idx - idx) < 30:
            wrong_idx = date_rng.randrange(len(test_ds))

        s, m = test_ds[wrong_idx]
        wrong_context = s[1:]
        zero_context = torch.zeros_like(wrong_context)

        target = to_data_range(target) #[0,1]: for the running sampling loss and for storage

        #regular sampling
        sample, loss_during_sampling = ddpm.sample(context=context, month=int(month), model=model, target=target) #sample already back to [0,1] range in function

        #Sampling with wrong context. The residual is still added to the true day k, so the
        #loss only reflects how much the network relies on its conditioning, not the wrong baseline.
        sample, no_context_loss= ddpm.sample(context=wrong_context, month=int(month), model=model, target=target, base=context[:1])

        sample, zero_context_loss = ddpm.sample(context=zero_context, month=int(month), model=model, target=target, base=context[:1])

        varying_t_loss = []
        #Testing different parts of t-range ability to produce SIC fields
        for t in torch.arange(100,ddpm.timesteps+100, 100):
            ddpm.timesteps = t

            sample = ddpm.sample(context=context, month=int(month), model=model) #sample already back to [0,1] range in function

            se = (sample - target.nan_to_num(nan=0.0).unsqueeze(1)) ** 2
            loss = se[ddpm.water_mask.expand_as(se)].mean()
            varying_t_loss.append(loss.item())

        ddpm.timesteps = T

        total_varying_loss.append(varying_t_loss)
        total_sampling_loss.append(loss_during_sampling)
        total_no_context_loss.append(no_context_loss)
        total_zero_context_loss.append(zero_context_loss)

        #dump the samples after each iteration
        pickle.dump((total_varying_loss, total_sampling_loss, total_no_context_loss, total_zero_context_loss), open(f"{outdir}/loss_stats.pkl", "wb"))
        print(f"wrote to: {outdir}/loss_stats.pkl")
