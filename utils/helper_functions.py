"""Run bookkeeping: config snapshots and the schedule summary. Sampling lives in utils/sampling.py."""
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from models.ddpm import DDPM
from config import UnetConfig, DDPMConfig, TrainConfig, EnsembleConfig

def schedule_summary(ddpm: DDPM) -> str:
    """One-line description of the realised noise schedule (printed and snapshotted)."""
    return (
        f"Noise schedule: {ddpm.schedule} "
        f"(image_size={ddpm.image_size} -- "
        f"sqrt(alpha_bar) at t=0: {ddpm.alphas_bar[0].sqrt():.4f}, "
        f"t=T: {ddpm.alphas_bar[-1].sqrt():.3e}"
    )

def snapshot_text(cfgs: tuple, ddpm: DDPM | None = None) -> str:
    """The values of the config instances in cfgs, plus the realised noise schedule if ddpm is given."""
    lines = [f"Config snapshot -- {datetime.now().isoformat(timespec='seconds')}", ""]
    # Instances, not classes: default_factory fields only exist on instances.
    for cfg in cfgs:
        lines.append(type(cfg).__name__)
        for f in fields(cfg):
            lines.append(f"    {f.name} = {getattr(cfg, f.name)!r}")
        lines.append("")

    if ddpm is not None:
        lines.append(schedule_summary(ddpm))
        lines.append("")
    return "\n".join(lines)

def dump_config_snapshot(train_cfg: TrainConfig ,ddpm: DDPM, unet_config:UnetConfig = UnetConfig(), ddpm_config:DDPMConfig = DDPMConfig(), unet:bool = False) -> list[Path]:
    """Write the config values used by this run into the run's weights dir and samples dir.

    config.py keeps changing between runs, so snapshotting the values next to the
    weights and the samples means an old result can still be traced back to what
    produced it. Both get a copy because samples_dir is on /work3 on HPC, apart from
    the weights. The realised noise schedule is included because it depends on the
    image size, which only exists at runtime.
    """

    # A set, so the file is written once when both dirs are the same (locally both are results/).
    out_dirs = {Path(d) / train_cfg.experiment_name for d in (train_cfg.output_dir, train_cfg.samples_dir)}
    text = snapshot_text((unet_config, ddpm_config, train_cfg), None if unet else ddpm)

    out_paths = []
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "config.txt"
        out_path.write_text(text)
        out_paths.append(out_path)
    return out_paths

def dump_ensemble_snapshot(ens_cfg: EnsembleConfig, train_cfg: TrainConfig, unet_cfg: UnetConfig, ddpm_cfg: DDPMConfig, ddpm: DDPM | None = None) -> Path:
    """Write the config values of an ensemble sampling run next to its ensemble_samples.pkl.

    Only the samples dir: ensemble sampling trains nothing, so it has no weights dir of its own.
    TrainConfig.experiment_name is not this run's name (ens_cfg.experiment_name is), but its
    dates, slice and lead_months are what the samples were drawn with. Pass ddpm for method="ddpm"
    to include the realised schedule.
    """
    out_dir = Path(train_cfg.samples_dir) / ens_cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "config.txt"
    out_path.write_text(snapshot_text((ens_cfg, unet_cfg, ddpm_cfg, train_cfg), ddpm))
    return out_path
