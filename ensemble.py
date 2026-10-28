"""
Ensemble forecasts from a trained model.

Three ways of getting a spread of forecasts for each test date:

- dropout: the residual UNet baseline, run n_members times with dropout left on (MC dropout,
  Gal & Ghahramani 2016). Each pass drops a different set of activations.
- ddpm: the conditional DDPM, sampled n_members times, each from a different initial noise.
- deep: n_members residual UNets trained by train_ensemble, each from its own seed (deep ensemble,
  Lakshminarayanan et al. 2017). Each member is one deterministic forecast, dropout off.

All write {output_dir}/{experiment_name}/ensemble_samples.pkl as (samples, contexts, targets, times, member_seeds),
the layout of download_unet_samples except that each entry of samples is [n_members, 1, H, W].
Dates are drawn with TrainConfig.sample_seed, so they match the single-forecast runs.
"""
import argparse
import pickle
import random
from pathlib import Path

import torch
from tqdm import tqdm

from utils import CARRA2Forecast
from utils.dataloader import to_data_range
from utils.device_utils import get_device
from models.ddpm import ConditionalDDPM
from models.unet import SimpleUnet
from config import UnetConfig, DDPMConfig, TrainConfig, EnsembleConfig, TRAINING_SLICE
from main import run_ddpm, run_unet
import os

device = get_device()
ENSEMBLE_SEEDS = [i for i in range(EnsembleConfig.n_members)]

def claim_seed(ensemble_path: Path, seed: int, force: bool = False) -> bool:
    """
    Mark member_{seed} as taken by this job by creating member_{seed}/claimed. Creating it with
    O_EXCL is atomic, so two jobs starting at once can't both get the same seed. force=True
    takes the seed even if it is already claimed (to rerun a crashed or killed member).
    """
    member_dir = ensemble_path / f"member_{seed}"
    member_dir.mkdir(parents=True, exist_ok=True)
    claim = member_dir / "claimed"
    if force:
        claim.unlink(missing_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"job {os.environ.get('LSB_JOBID', 'local')}\n".encode())
    os.close(fd)
    return True

def train_ensemble(seed: int | None = None):
    """
    Train one member of the deep ensemble into results/ensembles/member_{seed}.

    seed=None picks the first seed in ENSEMBLE_SEEDS that is neither trained nor claimed by a
    running job; the pick happens when the job starts, not when it is submitted. Passing a seed
    trains that one, even if it is claimed, so a crashed member can be rerun with --seed.
    """
    ensemble_path = Path("results/ensembles")
    ensemble_path.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        claim_seed(ensemble_path, seed, force=True)
        selected_seed = seed
    else:
        # A member counts as trained once its final weights exist.
        trained = {s for s in ENSEMBLE_SEEDS if (ensemble_path / f"member_{s}" / "final_weights.pth").exists()}
        remaining = [s for s in ENSEMBLE_SEEDS if s not in trained]
        print(f"{len(remaining)} untrained seeds: {remaining}")
        selected_seed = next((s for s in remaining if claim_seed(ensemble_path, s)), None)
        if selected_seed is None:
            print("Every untrained seed is claimed by another job; nothing to do. "
                  "If one crashed, rerun it with --seed <seed>.")
            return
    print(f"Selected {selected_seed}")

    train_conf = TrainConfig()
    train_conf.experiment_name = f"member_{selected_seed}"
    train_conf.output_dir = str(ensemble_path)

    model = run_unet(
        dataset=train_conf.dataset,
        batch_size=train_conf.batch_size,
        epochs=train_conf.epochs,
        lr=train_conf.lr,
        forecast_period=train_conf.forecasting_slice,
        experiment_name=train_conf.experiment_name,
        sampling=False,
        train_config=train_conf,
        unet_config=UnetConfig(),
        initialization_seed=selected_seed
    )
    
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

@torch.no_grad()
def run_ensemble(predict, to_sic, test_ds, member_seeds: list[int], out_dir: Path, train_cfg: TrainConfig):
    """
    Forecast train_cfg.number_of_samples test dates, once per seed in member_seeds.

    predict(context, month, seed) -> [1, 1, H, W] forecast in [0, 1], land NaN; called once per member.
    to_sic(x) maps the dataset's range back to SIC in [0, 1], for storing contexts and targets.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    contexts = []
    targets = []
    times = []
    # Own RNG so the drawn dates depend only on the seed, not on earlier use of the global one.
    date_rng = random.Random(train_cfg.sample_seed)
    for i in tqdm(range(train_cfg.number_of_samples)):
        idx = date_rng.randrange(len(test_ds))
        sample, month = test_ds[idx]   # [3, H, W] = (k+1, k, k-1); month of k+1 (0-11)
        context = sample[1:]           # (k, k-1) — what you condition on
        target  = sample[:1]           # k+1 — ground truth, for comparison only
        times.append(test_ds.get_dates(idx))

        members = []
        for seed in member_seeds:
            # Seeding per member makes member m reproducible on its own and distinct from the others.
            torch.manual_seed(seed)
            members.append(predict(context, month, seed).cpu())

        samples.append(torch.cat(members, dim=0))  # [n_members, 1, H, W]
        targets.append(to_sic(target).cpu())
        contexts.append(to_sic(context).cpu())

        # dump the samples after each date, an ensemble date is expensive
        pickle.dump((samples, contexts, targets, times, member_seeds), open(out_dir / "ensemble_samples.pkl", "wb"))

    print(f"Wrote {len(samples)} dates x {len(member_seeds)} members to {out_dir / 'ensemble_samples.pkl'}")

def unet_datasets(train_cfg: TrainConfig):
    """Test set standardized with the training statistics the residual UNet was trained with,
    the training set (for its land mask), and to_sic, which undoes the standardization."""
    train_ds = CARRA2Forecast("siconc", device, area=train_cfg.dataset, time_slice=TRAINING_SLICE, batch_dim=False, standardize=True)
    test_ds = CARRA2Forecast("siconc", device, area=train_cfg.dataset, time_slice=train_cfg.forecasting_slice, batch_dim=False, standardize=True)
    test_ds.mean, test_ds.std = train_ds.mean, train_ds.std
    mean, std = float(train_ds.mean), float(train_ds.std)

    def to_sic(x):
        return x * std + mean

    return train_ds, test_ds, to_sic

def unet_forecast(model: SimpleUnet, context: torch.Tensor, month: torch.Tensor, to_sic) -> torch.Tensor:
    """One residual UNet forecast of k+1, as in download_unet_samples: NaN -> 0 (= the training mean),
    plus the land mask channel, and the network predicts the residual k+1 - k."""
    x = torch.cat([context.nan_to_num(0.0).unsqueeze(0),
                   model.mask_channel.expand(1, -1, -1, -1)], dim=1)
    t = torch.zeros(1, device=context.device)
    pred = x[:, 0:1] + model(x, t, month.unsqueeze(0))
    return to_sic(pred).clamp(0.0, 1.0).masked_fill(~model.water_mask, float("nan"))

def dropout_ensemble(ens_cfg: EnsembleConfig, train_cfg: TrainConfig, unet_cfg: UnetConfig):
    """MC-dropout ensemble of the residual UNet baseline in ens_cfg.weights."""
    assert unet_cfg.dropout > 0, "MC dropout needs UnetConfig.dropout > 0"

    train_ds, test_ds, to_sic = unet_datasets(train_cfg)

    model = build_unet(3, ens_cfg.weights, unet_cfg, land_mask=train_ds.finite_mask)
    # train() only switches dropout on: SimpleUnet normalises with GroupNorm, which has no running statistics.
    model.train()

    def predict(context, month, seed):
        return unet_forecast(model, context, month, to_sic)

    run_ensemble(predict, to_sic, test_ds, list(range(ens_cfg.n_members)), Path(train_cfg.output_dir) / ens_cfg.experiment_name, train_cfg)

def ddpm_ensemble(ens_cfg: EnsembleConfig, train_cfg: TrainConfig, unet_cfg: UnetConfig, ddpm_cfg: DDPMConfig):
    """Ensemble of the conditional DDPM in ens_cfg.weights, one member per initial noise.
    ddpm_cfg must match the run that produced the weights (schedule, cosine_shift_res, timesteps)."""
    test_ds = CARRA2Forecast("siconc", device, area=train_cfg.dataset, time_slice=train_cfg.forecasting_slice, batch_dim=False)

    ddpm = ConditionalDDPM(
        timesteps=ddpm_cfg.timesteps,
        device=device,
        image_size=test_ds[0][0].shape[-1],
        land_mask=test_ds.finite_mask,
        schedule=ddpm_cfg.schedule,
        forecast_period=train_cfg.forecasting_slice,
    )
    model = build_unet(4, ens_cfg.weights, unet_cfg)
    # Dropout off: here the spread comes from the reverse process, not the network.
    model.eval()

    def predict(context, month, seed):
        return ddpm.sample(context=context, month=int(month), model=model)

    run_ensemble(predict, to_data_range, test_ds, list(range(ens_cfg.n_members)), Path(train_cfg.output_dir) / ens_cfg.experiment_name, train_cfg)

def deep_ensemble(ens_cfg: EnsembleConfig, train_cfg: TrainConfig, unet_cfg: UnetConfig):
    """Deep ensemble of the residual UNets that train_ensemble wrote to
    {ens_cfg.members_dir}/member_{seed}/final_weights.pth, one member per seed in ENSEMBLE_SEEDS."""
    member_seeds = ENSEMBLE_SEEDS[:ens_cfg.n_members]
    weights = {seed: Path(ens_cfg.members_dir) / f"member_{seed}" / "final_weights.pth" for seed in member_seeds}
    missing = [seed for seed, path in weights.items() if not path.exists()]
    assert not missing, f"members not trained yet (run train_ensemble): {missing}"

    train_ds, test_ds, to_sic = unet_datasets(train_cfg)

    models = {seed: build_unet(3, str(path), unet_cfg, land_mask=train_ds.finite_mask) for seed, path in weights.items()}
    # Dropout off: here the spread comes from the members' different initial weights and training runs.
    for model in models.values():
        model.eval()

    def predict(context, month, seed):
        return unet_forecast(models[seed], context, month, to_sic)

    run_ensemble(predict, to_sic, test_ds, member_seeds, Path(train_cfg.output_dir) / ens_cfg.experiment_name, train_cfg)

def sample_ensemble(ens_cfg: EnsembleConfig, train_cfg: TrainConfig, unet_cfg: UnetConfig, ddpm_cfg: DDPMConfig):
    assert train_cfg.forecasting_slice is not None, "Ensembles are of forecasts; set TrainConfig.forecasting_slice"

    if ens_cfg.method == "dropout":
        dropout_ensemble(ens_cfg, train_cfg, unet_cfg)
    elif ens_cfg.method == "ddpm":
        ddpm_ensemble(ens_cfg, train_cfg, unet_cfg, ddpm_cfg)
    elif ens_cfg.method == "deep":
        deep_ensemble(ens_cfg, train_cfg, unet_cfg)
    else:
        raise ValueError(f"unknown ensemble method {ens_cfg.method!r}; expected 'dropout', 'ddpm' or 'deep'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ENSEMBLE_SEED is for HPC, where `bsub < run.sh` can't pass arguments: `ENSEMBLE_SEED=3 bsub < run.sh`.
    parser.add_argument("--seed", type=int, default=os.environ.get("ENSEMBLE_SEED"),
                        help="train this ensemble member; default is the first one not trained or running")
    args = parser.parse_args()

    train_ensemble(args.seed)
    #sample_ensemble(EnsembleConfig(), TrainConfig(), UnetConfig(), DDPMConfig())
