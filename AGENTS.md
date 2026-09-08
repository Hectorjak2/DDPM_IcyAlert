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

1. **Never run git.** The allowed bash commands are:
   ```
   ./utils/hpc_run.sh <experiment_name>                      # launch a run
   ./.venv/bin/python utils/evaluate_samples.py <name> [...]  # evaluate a run
   ./.venv/bin/python analysis/<script>.py [...]             # ad-hoc analysis
   ```
   (The launch script handles git/HPC submission internally — that is not you
   running git.) No shell chaining/metacharacters are permitted. Inspect files
   with the `read`, `grep`, `find`, `ls` tools, not bash.
2. **Only edit `config.py`** (to change the experiment) **or scratch files under
   `analysis/`** (temporary evaluation scripts). `write` is allowed only under
   `analysis/`; `edit` only on `config.py` or `analysis/`. The source pipeline
   (`main.py`, `models/`, `utils/`) stays read-only. Every hyperparameter is
   config-driven (`main.py` consumes `UnetConfig`, `DDPMConfig`, `TrainConfig`),
   so editing `config.py` is sufficient to define a run.
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

## Evaluating samples

The task is **unconditional generation**: we are not matching a sample to a
specific target field, we are asking whether the *set* of generated fields looks
like the *set* of real CARRA2 fields. Evaluation is therefore **distributional**.

**Guiding principle — never trust a single scalar.** Two failure modes have
already fooled point metrics (see `docs/diaries/`): a training loss that
"converged" while samples were noise, and samples whose *mean* looked plausible
(~0.52 water) but which were just the prior mean with no spatial structure. Both
were only caught by looking at the **value distribution** and the **spatial
structure**, not any one number. Read at least two levels before judging a run.

**The tool:** `utils/evaluate_samples.py` compares a run's `samples.pkl` against
real fields drawn evenly across the time axis (seasonal cycle matters — an
all-January subset makes the spatial stats lie). It writes a JSON summary + 3 PNGs
to `results/<experiment_name>/evaluation/`. Run it with the project venv:
```
./.venv/bin/python utils/evaluate_samples.py <experiment_name> [--n-real 96]
```
(`--n-real >= 96` is needed for stable spatial statistics.)

**What it reports, and how to read it (all Wasserstein/L1: lower = better):**
- **Concentration histogram + Wasserstein** — the value distribution over water
  pixels. Real SIC is strongly **bimodal** (spike at 0 = open water, spike at 1 =
  full ice). A unimodal blob near the mean is the classic prior-mean collapse.
  Watch `frac < 0.05`: the model must reproduce open water, not just the average.
- **Radially-averaged power spectrum (log-log) + `spectrum_log_l1`** — whether
  structure exists across scales. This is the primary spatial-structure check; a
  flat or wrong-slope spectrum means the field is noise or over-smooth.
- **Per-field summaries** (mean conc, ice fraction, within-field spatial std),
  each with its own Wasserstein — the set-level "does the population look right"
  check. Near-zero `spatial_std` = flat fields.
- **Sanity checks** — values in [0,1], and land-mask agreement with the data.

**Mode note.** In **experiment mode** you CAN now run the evaluator and ad-hoc
analysis between runs: `./.venv/bin/python utils/evaluate_samples.py <name> [...]`
and `./.venv/bin/python analysis/<script>.py [...]` are allowed, and you may
`write`/`edit` scratch scripts under `analysis/`. These do NOT count toward the
experiment budget — only `./utils/hpc_run.sh` launches do. The source pipeline
stays read-only, so **wiring the evaluator into `main.py`** to run automatically
at the end of a run is still a pipeline change (out of bounds): log it with
`record_finding`, don't attempt it.

**First real 1216² run is expected to fail.** With the current undertraining
(~288 optimizer steps) and the never-before-tested full-resolution schedule, the
point of evaluating is to make the failure *legible* (which level breaks) rather
than to declare success.
