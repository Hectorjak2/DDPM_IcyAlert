#!/bin/bash
# Submit a job on the HPC, wait for it to finish, copy the results back,
# then suspend and kill the job.
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

cd "$REPO"
git add config.py
git commit . -m "$NAME" || echo "nothing to commit — pushing what's already here"
git push

# Capture the remote output so we can pull the job id out of the bsub line.
SUBMIT_OUT=$(ssh -i "$KEY" "$HOST" bash -l <<EOF
set -e
sleep 2
cd $REMOTE
git pull
bsub -J $NAME -o gpu_$NAME.out -e gpu_$NAME.err < run.sh
EOF
)
echo "$SUBMIT_OUT"

# bsub prints: Job <29363658> is submitted to queue <gpua100>.
JOBID=$(printf '%s\n' "$SUBMIT_OUT" | sed -n 's/^Job <\([0-9][0-9]*\)>.*/\1/p' | tail -n 1)
echo "job id: ${JOBID:-<not found>}"

echo "waiting for 'SCRIPT DONE' in gpu_$NAME.out (checking every 5 min)"
while true; do
  sleep 300
  LAST=$(ssh -i "$KEY" "$HOST" "tail -n 1 $REMOTE/gpu_$NAME.out 2>/dev/null" || true)
  echo "[$(date +%H:%M:%S)] ${LAST:-<no output yet>}"
  case "$LAST" in *"SCRIPT DONE"*) break ;; esac
done

scp -r -i "$KEY" "$HOST:$REMOTE/results/$NAME" "$LOCAL"

# Fallback: if we never got an id from bsub, dig it out of bstat. LSF truncates
# long names to the last 10 chars and prefixes them with '*', so match on suffix.
if [ -z "$JOBID" ]; then
  echo "no job id from bsub — falling back to bstat"
  JOBID=$(ssh -i "$KEY" "$HOST" "bash -lc bstat" \
    | awk -v n="$NAME" '
        $1 ~ /^[0-9]+$/ {
          jn = $4; sub(/^\*/, "", jn)
          if (jn == n || substr(n, length(n) - length(jn) + 1) == jn) print $1
        }' | tail -n 1)
fi

if [ -n "$JOBID" ]; then
  echo "stopping and killing job $JOBID"
  ssh -i "$KEY" "$HOST" bash -l <<EOF
bstop $JOBID || true
bkill $JOBID || true
EOF
else
  echo "could not determine job id — no bstop/bkill issued" >&2
fi

echo "Experiment completed"