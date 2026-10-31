#!/bin/bash	

#SBATCH --cpus-per-task=1
#SBATCH --time=24:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu
#SBATCH --nodelist=comp-gpu04
#SBATCH --mail-user=s234822@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

## Usage: ./titan.s <name>  (self-submits with job name and output file set to <name>)
if [[ -z $SLURM_JOB_ID ]]; then
  NAME=${1:-hcjtest}
  exec sbatch --job-name="$NAME" --output="$NAME-%J.out" "$0" "$@"
fi

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

export CARRA2_DATA_ROOT=/scratch/s234822/CARRA2_DAILY

source ~/.bashrc
source .venv/bin/activate
python -m main

echo "Done: $(date +%F-%R:%S)"