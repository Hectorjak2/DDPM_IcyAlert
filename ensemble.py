"""
Ensemble forecasts from a trained model.

Three ways of getting a spread of forecasts for each test date:

- dropout: the residual UNet baseline, run n_members times with dropout left on (MC dropout,
  Gal & Ghahramani 2016). Each pass drops a different set of activations.
- ddpm: the conditional DDPM, sampled n_members times, each from a different initial noise.
- deep: n_members residual UNets trained by train_ensemble, each from its own seed (deep ensemble,
  Lakshminarayanan et al. 2017). Each member is one deterministic forecast, dropout off.

All write {output_dir}/{experiment_name}/ensemble_samples.pkl through utils.sampling.forecast, as
(samples, contexts, targets, times, member_seeds) with each entry of samples [n_members, lead_months, H, W].
Dates are drawn with TrainConfig.sample_seed, so they match the single-forecast runs with the same lead_months.
"""
import argparse
from pathlib import Path

from utils.sampling import forecast, forecast_dataset, dropout_forecaster, ddpm_forecaster, deep_forecaster
from config import UnetConfig, DDPMConfig, TrainConfig, EnsembleConfig
from main import run_unet
import os

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
    
def sample_ensemble(ens_cfg: EnsembleConfig, train_cfg: TrainConfig, unet_cfg: UnetConfig, ddpm_cfg: DDPMConfig):
    """Ensemble forecast with ens_cfg.method, rolled out TrainConfig.lead_months months (see utils/sampling.py)."""
    assert train_cfg.forecasting_slice is not None, "Ensembles are of forecasts; set TrainConfig.forecasting_slice"
    test_ds = forecast_dataset(train_cfg)
    member_seeds = list(range(ens_cfg.n_members))

    if ens_cfg.method == "dropout":
        forecaster = dropout_forecaster(ens_cfg.weights, train_cfg, unet_cfg)
    elif ens_cfg.method == "ddpm":
        forecaster = ddpm_forecaster(ens_cfg.weights, test_ds, train_cfg, unet_cfg, ddpm_cfg)
    elif ens_cfg.method == "deep":
        member_seeds = ENSEMBLE_SEEDS[:ens_cfg.n_members]
        forecaster = deep_forecaster(ens_cfg.members_dir, member_seeds, train_cfg, unet_cfg)
    else:
        raise ValueError(f"unknown ensemble method {ens_cfg.method!r}; expected 'dropout', 'ddpm' or 'deep'")

    forecast(forecaster, test_ds, member_seeds, train_cfg.number_of_samples, train_cfg.sample_seed,
             Path(train_cfg.output_dir) / ens_cfg.experiment_name / "ensemble_samples.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ENSEMBLE_SEED is for HPC, where `bsub < run.sh` can't pass arguments: `ENSEMBLE_SEED=3 bsub < run.sh`.
    parser.add_argument("--seed", type=int, default=os.environ.get("ENSEMBLE_SEED"),
                        help="train this ensemble member; default is the first one not trained or running")
    args = parser.parse_args()

    train_ensemble(args.seed)
    #sample_ensemble(EnsembleConfig(), TrainConfig(), UnetConfig(), DDPMConfig())
