#!/bin/bash
# Submit a job on the HPC, wait for it to finish, copy the results back.
#
#   ./hpc_run.sh MyExperimentName
#
set -euo pipefail

NAME="${1:?usage: ./hpc_run.sh JOBNAME}"

KEY=~/.ssh/gbar
HOST=s234822@login.hpc.dtu.dk
REMOTE=/zhome/eb/6/205174/DDPM_IcyAlert
REPO=/Users/hectorheltjakobsen/Documents/Dokumenter/DTU/BACHELOR/DDPM_IcyAlert
LOCAL=$REPO/results

# Commit and push whatever is in the working tree, so the remote git pull picks
# it up. `git commit .` stages everything under the repo root by itself, so run
# it from there. If nothing changed, carry on rather than aborting on set -e.
cd "$REPO"
git add config.py
git commit . -m "$NAME" || echo "nothing to commit — pushing what's already here"
git push

# git pull + submit. -o/-e override the %J filenames in run.sh so the log is
# named after the job, not the job id.
# 'bash -l' runs a LOGIN shell on the remote side, so /etc/profile and your
# .bash_profile are sourced and bsub/bjobs end up on PATH. A plain
# `ssh host "cmd"` skips those and bsub is not found.
ssh -i "$KEY" "$HOST" bash -l <<EOF
set -e
sleep 2
cd $REMOTE
git pull
bsub -J $NAME -o gpu_$NAME.out -e gpu_$NAME.err < run.sh
EOF

echo "waiting for 'SCRIPT DONE' in gpu_$NAME.out (checking every 5 min)"
while true; do
  sleep 300
  LAST=$(ssh -i "$KEY" "$HOST" "tail -n 1 $REMOTE/gpu_$NAME.out 2>/dev/null" || true)
  echo "[$(date +%H:%M:%S)] ${LAST:-<no output yet>}"
  case "$LAST" in *"SCRIPT DONE"*) break ;; esac
done

scp -r -i "$KEY" "$HOST:$REMOTE/results/$NAME" "$LOCAL"

echo "Experiment completed"
