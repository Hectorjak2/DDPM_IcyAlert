#!/bin/sh

### General options
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=64GB]"
#BSUB -R "select[gpu80gb]"

#BSUB -u hectorheltj@gmail.com
#BSUB -B
#BSUB -N

#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

nvidia-smi

module load cuda/12.4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/zhome/eb/6/205174/DDPM_IcyAlert/.venv/bin/python -u \
    /zhome/eb/6/205174/DDPM_IcyAlert/main.py