import pickle
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from models.ddpm import DDPM, ConditionalDDPM
from config import UnetConfig, DDPMConfig, TrainConfig 
from utils.dataloader import to_data_range

import random

def schedule_summary(ddpm: DDPM) -> str:
    """One-line description of the realised noise schedule (printed and snapshotted)."""
    return (
        f"Noise schedule: {ddpm.schedule} "
        f"(image_size={ddpm.image_size}, ref={ddpm.shift_ref_resolution}) -- "
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
        pickle.dump(samples, open(f"results/{train_cfg.experiment_name}/samples.pkl", "wb"))

def download_conditional_samples(ddpm: ConditionalDDPM, model, test_ds, train_cfg: TrainConfig, n_of_samples: int = 10):
    print("Sampling from the trained model ...")
    # Disable dropout for inference. Left in train mode the residual blocks inject
    # noise at every one of the T reverse steps, which wrecks the sample.
    model.eval()
    Path(f"results/{train_cfg.experiment_name}").mkdir(parents=True, exist_ok=True)

    samples = []
    contexts = []
    targets = []
    times = []
    sampling_losses = []
    for i in tqdm(range(n_of_samples)):
        idx = random.randrange(len(test_ds))
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
        pickle.dump((samples, contexts, targets, times, sampling_losses), open(f"results/{train_cfg.experiment_name}/samples.pkl", "wb"))


def dump_config_snapshot(train_cfg: TrainConfig, ddpm: DDPM) -> Path:
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
    for cfg in (UnetConfig(), DDPMConfig(), train_cfg):
        lines.append(type(cfg).__name__)
        for f in fields(cfg):
            lines.append(f"    {f.name} = {getattr(cfg, f.name)!r}")
        lines.append("")

    lines.append(schedule_summary(ddpm))
    lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path
