#!/bin/bash
# SLURM script for multi-node distributed training
# Set environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Activate conda environment if needed
# conda activate rs-diffusion

# Run training
python main.py \
    --config config/DDPM_UNet/train_config.yaml \
    --mode train

