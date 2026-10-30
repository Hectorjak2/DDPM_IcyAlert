#!/bin/sh

### General options ,gpua40, gpuv100, gpul40s, gpua100
#BSUB -q gpul40s
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -R "rusage[mem=16GB]"
###BSUB -R "select[gpu80gb]"

#BSUB -u hectorheltj@gmail.com
#BSUB -B
#BSUB -N

#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err

nvidia-smi

module load cuda/12.4.1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CARRA2_DATA_ROOT=/work3/s234822/data/data/CARRA2_DAILY
export TEMP_RESULT_STORAGE=/work3/s234822/data/data/samples

/zhome/eb/6/205174/DDPM_IcyAlert/.venv/bin/python -u \
    /zhome/eb/6/205174/DDPM_IcyAlert/main.py