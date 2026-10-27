#!/bin/bash	

#SBATCH --job-name=hcjtest
#SBATCH --output=hcjtest-%J.out
#SBATCH --cpus-per-task=1
#SBATCH --time=5:00
#SBATCH --mem=16gb
#SBATCH --gres=gpu
#SBATCH --nodelist=comp-gpu13
#SBATCH --mail-user=s234822@dtu.dk
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=titans
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

SCRATCH=/scratch/$USER
if [[ ! -d $SCRATCH ]]; then
  mkdir $SCRATCH
fi

source ~/.bashrc
source .venv/bin/activate
python -m main

echo "Done: $(date +%F-%R:%S)"