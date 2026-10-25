import pickle
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from models.ddpm import DDPM, ConditionalDDPM
from config import UnetConfig, DDPMConfig, TrainConfig 
from utils.dataloader import to_data_range

import torch
import random

def schedule_summary(ddpm: DDPM) -> str:
    """One-line description of the realised noise schedule (printed and snapshotted)."""
    return (
        f"Noise schedule: {ddpm.schedule} "
        f"(image_size={ddpm.image_size} -- "
        f"sqrt(alpha_bar) at t=0: {ddpm.alphas_bar[0].sqrt():.4f}, "
        f"t=T: {ddpm.alphas_bar[-1].sqrt():.3e}"
    )

def download_samples(ddpm: DDPM, model, train_cfg: TrainConfig, n_of_samples: int = 10):
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    Path(f"results/{train_cfg.experiment_name}").mkdir(parents=True, exist_ok=True)
    samples = []
    for i in tqdm(range(n_of_samples)):
        sample = ddpm.sample(model)
        samples.append(sample.cpu())

        #dump the samples after each iteration
        pickle.dump(samples, open(f"results/{train_cfg.experiment_name}/{train_cfg.experiment_name}_samples.pkl", "wb"))

def download_conditional_samples(ddpm: ConditionalDDPM, model, test_ds, train_cfg: TrainConfig, n_of_samples: int = 10):
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    outdir = train_cfg.output_dir

    Path(outdir).mkdir(parents=True, exist_ok=True)

    samples = []
    contexts = []
    targets = []
    times = []
    sampling_losses = []
    # Own RNG so the drawn dates depend only on the seed, not on earlier use of the global one.
    date_rng = random.Random(train_cfg.sample_seed)
    for i in tqdm(range(n_of_samples)):
        idx = date_rng.randrange(len(test_ds))
        sample, month = test_ds[idx]   # [3, H, W] = (k+1, k, k-1); month of k+1 (0-11)
        context = sample[1:]           # (k, k-1) — what you condition on
        target  = sample[:1]           # k+1 — ground truth, for comparison only
        times.append(test_ds.get_dates(idx))

        target = to_data_range(target) #[0,1]: for the running sampling loss and for storage

        sample, loss_during_sampling = ddpm.sample(context=context, month=int(month), model=model, target=target) #sample already back to [0,1] range in function
        
        context = to_data_range(context)  # back to [0,1] range for comparison with sample

        samples.append(sample.cpu())
        targets.append(target.cpu())
        contexts.append(context.cpu())
        sampling_losses.append(loss_during_sampling)

        #dump the samples after each iteration
        pickle.dump((samples, contexts, targets, times, sampling_losses), open(f"{outdir}/samples.pkl", "wb"))

def analysis_sampling(ddpm: ConditionalDDPM, model, test_ds, train_cfg: TrainConfig, n_of_samples: int = 10):
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    outdir = train_cfg.output_dir

    Path(outdir).mkdir(parents=True, exist_ok=True)

    T = ddpm.timesteps
    total_varying_loss = []
    total_sampling_loss = []
    total_no_context_loss = []

    # Own RNG so the drawn dates depend only on the seed, not on earlier use of the global one.
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

        target = to_data_range(target) #[0,1]: for the running sampling loss and for storage

        #regular sampling
        sample, loss_during_sampling = ddpm.sample(context=context, month=int(month), model=model, target=target) #sample already back to [0,1] range in function

        #Sampling with wrong context. The residual is still added to the true day k, so the
        #loss only reflects how much the network relies on its conditioning, not the wrong baseline.
        sample, no_context_loss= ddpm.sample(context=wrong_context, month=int(month), model=model, target=target, base=context[:1])

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

        #dump the samples after each iteration
        pickle.dump((total_varying_loss, total_sampling_loss, total_no_context_loss), open(f"{outdir}/loss_stats.pkl", "wb"))
        print(f"wrote to: {outdir}/loss_stats.pkl")

@torch.no_grad()
def download_unet_samples(model, test_ds, n_of_samples: int, train_cfg: TrainConfig):
    """Deterministic UNet baseline. Saves (samples, contexts, targets, times) in [0, 1] with
    land as NaN -- the same layout as download_conditional_samples, minus sampling_losses."""
    print("Sampling from the trained model ...")
    outdir = train_cfg.output_dir
    # test_ds must carry the *training* statistics (see train_unet in main.py).
    mean, std = float(test_ds.mean), float(test_ds.std)
    
    def de_standardize(x):
        return x * std + mean

    model.eval()
    Path(outdir).mkdir(parents=True, exist_ok=True)

    samples = []
    contexts = []
    targets = []
    times = []
    # Own RNG so the drawn dates depend only on the seed, not on earlier use of the global one.
    date_rng = random.Random(train_cfg.sample_seed)
    for i in tqdm(range(n_of_samples)):
        idx = date_rng.randrange(len(test_ds))
        sample, month = test_ds[idx]   # [3, H, W] = (k+1, k, k-1), standardized; month of k+1 (0-11)
        context = sample[1:]           # (k, k-1) — what you condition on, NaN on land
        target  = sample[:1]           # k+1 — ground truth, for comparison only, NaN on land
        times.append(test_ds.get_dates(idx))

        # Same input as training: NaN -> 0 (= the training mean), plus the land mask channel.
        x = torch.cat([context.nan_to_num(0.0).unsqueeze(0),
                       model.mask_channel.expand(1, -1, -1, -1)], dim=1)
        t = torch.zeros(1, device=context.device)
        # The network predicts the residual k+1 - k; add day k back, as in train_network.
        pred = x[:, 0:1] + model(x, t, month.unsqueeze(0))  # [1, 1, H, W], standardized

        # Back to SIC, clamped like the DDPM output, land set to NaN.
        pred = de_standardize(pred).clamp(0.0, 1.0).masked_fill(~model.water_mask, float("nan"))

        samples.append(pred.cpu())
        targets.append(de_standardize(target).cpu())
        contexts.append(de_standardize(context).cpu())

        if i%10 == 0 or i == (n_of_samples-1): 
            pickle.dump((samples, contexts, targets, times), open(f"{outdir}/samples.pkl", "wb"))


def dump_config_snapshot(train_cfg: TrainConfig ,ddpm: DDPM, unet_config:UnetConfig = UnetConfig(), ddpm_config:DDPMConfig = DDPMConfig(), unet:bool = False) -> Path:
    """Write the config values used by this run into the run's results dir.

    config.py keeps changing between runs, so snapshotting the values next to the
    samples means an old result can still be traced back to what produced it. The
    realised noise schedule is included because it depends on the image size, which
    only exists at runtime.
    """
    out_dir = Path("results") / train_cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.txt"

    lines = [f"Config snapshot -- {datetime.now().isoformat(timespec='seconds')}", ""]
    # Instances, not classes: default_factory fields only exist on instances.
    for cfg in (unet_config, ddpm_config, train_cfg):
        lines.append(type(cfg).__name__)
        for f in fields(cfg):
            lines.append(f"    {f.name} = {getattr(cfg, f.name)!r}")
        lines.append("")

    if not unet: 
        lines.append(schedule_summary(ddpm))
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path