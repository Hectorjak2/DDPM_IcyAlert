# DDPM_IcyAlert

This project has **two operating modes** (toggle with `shift+tab` or `/mode`);
the current mode lives in `.pi/experiment-state.json`.

- **Normal mode (default, green):** Full access — `read`/`edit`/`write`/`bash`
  all work on any file. The only guardrail is a confirmation prompt before
  risky bash commands (`rm -rf`, `sudo`, force-push, etc.). The hard rules
  below do NOT apply here; use normal judgement.
- **Experiment mode (yellow):** The restricted thesis workflow described
  below (only `config.py` edits + `./utils/hpc_run.sh`).

The rest of this file describes **experiment mode**.

---

An "experiment" = editing `config.py`, then launching a training+sampling run.

## Hard rules (also enforced deterministically by the experiment-mode extension)

1. **Never run git.** The only allowed bash command is:
   ```
   ./utils/hpc_run.sh <experiment_name>
   ```
   (The launch script handles git/HPC submission internally — that is not you
   running git.) Inspect files with the `read`, `grep`, `find`, `ls` tools, not bash.
2. **Only edit `config.py`.** The `write` tool is disabled; `edit` works only on
   `config.py`. Every hyperparameter is config-driven (`main.py` consumes
   `UnetConfig`, `DDPMConfig`, `TrainConfig`), so this is sufficient.
3. **Name invariant — critical.** Every time you edit `config.py`, set
   `TrainConfig.experiment_name`, and use that **exact same name** as the argument
   to `./utils/hpc_run.sh`. This is how results get filed where you can find them.
   A launch is blocked if the name and the config disagree.
4. **Out-of-bounds ideas:** if an experiment needs a change beyond `config.py`,
   do NOT attempt it. Log it with the `record_finding` tool and continue. Raise
   all findings to the user after the planned experiments are done.

## Waiting for results from the experiment: 
Launch is long-running: ./utils/hpc_run.sh submits to the HPC queue and blocks until 'SCRIPT DONE' (up to ~12h, polling every 5 min). Do NOT treat this as a hang or set a short bash timeout. 


## You may
- Read `config.py`, `main.py`, and files under `models/` and `utils/` to understand
  what a hyperparameter does.

## Known experiment constraints                                                                              
   - Memory: at the current image resolution, batch_size > 2 OOMs. **batch_size = 2 is the max.**               
   - Timesteps: safe to increase. Theory says more timesteps only makes *sampling* slower,                      
     not training — so it's a cheap knob to explore. 

## Useful commands
- `/experiments` — run log (running / completed / failed)
- `/findings` — out-of-bounds findings to raise
- `/clear-findings` — clear them after raising
- `/budget <n>` — start an iteration of `n` runs; `/budget` alone shows `used/total`
- `/clear-budget` — remove the cap (unlimited launches)

## Experiment budget (deterministic iteration cap)
An "iteration" = a fixed number of runs you commit to before reviewing.
- Set it with the `set_experiment_budget` tool (or `/budget <n>`). Saying
  "start experimenting, 5 total" ⇒ budget of 5.
- **Every launch counts — success OR failure.** A crashed run still consumes one.
- Within the budget you may launch experiments back-to-back.
- When `used == total`, the next `./utils/hpc_run.sh` launch is **hard-blocked**;
  stop and hand back for review.
- To continue, set a new budget (resets the count) or start a new session.

## Workflow per experiment
1. Read what needs changing.
2. Edit `config.py`, choosing a unique `experiment_name`.
3. Launch: `./utils/hpc_run.sh <same experiment_name>`.
4. The dashboard tracks status; the launch blocks until the run reports `SCRIPT DONE`.
