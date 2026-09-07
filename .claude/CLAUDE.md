# DDPM_IcyAlert — AI Context Guide

## ⛔ CRITICAL: Never run git commands

Claude should **NEVER** use bash to run `git` commands (git add, git commit, git push, git rebase, etc.).
Only the user commits code. If changes need committing, notify the user and wait for explicit approval.

---

## Working with Claude

Claude should read this file and docs/architecture.md first on every session.
When diagnosing issues, consult architecture.md for design rationale before speculating.

## Overview

This is a diffusion model (DDPM) for **sea-ice concentration (SIC) forecasting** on the CARRA2 reanalysis dataset. The model learns to denoise noisy SIC fields and generate plausible sea-ice concentration maps over the North Atlantic/European shelf region. [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) is used as a toy/sanity-check dataset for quick local iteration; it is not a research target.

## Environments

Two distinct execution contexts — **always keep this distinction in mind when reading the code or deploying**:

### Local (Mac, `mps`/cpu)
- **Purpose:** Dev, debugging, smoke-testing.
- **Data:** FashionMNIST (auto-downloaded) or CARRA2 TEST region (128×128 crop, fast).
- **Configs:** Small (`batch_size=2, epochs=2`) to iterate quickly.
- **Device:** `mps` (Metal Performance Shaders, Apple GPU) or CPU fallback.

### DTU HPC (LSF/BSUB, `gpua100` queue)
- **Purpose:** Real CARRA2 training over the full WEST region.
- **Data:** Full CARRA2-WEST (1216×1216, real data).
- **Configs:** Whatever is hardcoded in [main.py](main.py)'s `__main__` block at submission time — `run.sh` just runs `python main.py` with no CLI args, so you must edit `main.py` and recommit before `bsub < run.sh`.
- **Device:** CUDA 12.4.1 (loaded by [run.sh](run.sh)).
- **Resources:** 1×GPU (80GB), 64GB system RAM, 12h walltime, 4 CPU cores.

**Critical:** Don't accidentally deploy a local smoke-test config to HPC. Check [main.py](main.py) line 72–77 before `bsub`.

## Architecture

Module map and design decisions (NaN-masking, UNet downsizing, region split, device handling, known inconsistencies) are documented in [docs/architecture.md](docs/architecture.md). Read this when you need it.

## Configuration

Hyperparameters (formerly scattered) are centralized in [config.py](config.py) as dataclasses: `UnetConfig`, `DDPMConfig`, `TrainConfig`. This file is the source of truth for what actually runs.

## Verification

**No automated test suite exists.** Verification is currently manual:

1. **Local smoke-test:** Run the FashionMNIST or CARRA2 TEST path locally, inspect the loss curve printed to stdout, then check the pickled samples:
   ```bash
   python main.py  # runs local smoke-test (whatever __main__ says)
   # Inspect output:
   ls results/
   python -c "import pickle; samples = pickle.load(open('results/<run_name>/samples.pkl', 'rb')); print(f'Generated {len(samples)} samples')"
   ```

2. **HPC output:** After `bsub` completes, check `gpu_<job_id>.out` and `gpu_<job_id>.err` for training curves and errors, then inspect the checkpoint:
   ```bash
   ls results/
   ```

## Conventions

- **Commit messages are not a reliable source of "why"** — they are sparse/non-descriptive. Design rationale lives in [docs/architecture.md](docs/architecture.md) (stable facts) or your personal Claude memory (in-flight decisions, rejected alternatives). When you make a design decision, update [docs/architecture.md](docs/architecture.md) so the next session knows why.
- **Session-by-session reasoning lives in [docs/diaries/](docs/diaries/)** — theory worked through, alternatives rejected, decisions deliberately deferred. Written by the `/diary` skill ([.claude/skills/diary/SKILL.md](.claude/skills/diary/SKILL.md)), which **only the user invokes** — never write a diary entry on your own initiative. Read the most recent entries when picking up in-flight work; [docs/architecture.md](docs/architecture.md) remains the home for stable facts.
- **Keep hyperparameters in [config.py](config.py)** — it's the single source of truth for what runs, and makes changes diffable.
- **When architecture changes, update [docs/architecture.md](docs/architecture.md)** — the module map, design decisions, and reasoning. Don't let that knowledge evaporate.
