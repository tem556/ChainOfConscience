#!/bin/bash
#SBATCH --gres=gpu:A100_80GB:4
#SBATCH --partition=gpu-all
#SBATCH --job-name=llama-coc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_%j.log

# Load the correct CUDA module
module load cuda12.2/toolkit/12.2.1

# Source conda setup (load conda functions into current shell)
source ~/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate coc

# Set environment variables
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run the training script with Accelerate
accelerate launch \
    --config_file src/open-r1/recipes/accelerate_configs/zero2.yaml \
    --num_processes=3 \
    src/open-r1/src/open_r1/grpo.py \
    --config src/open-r1/recipes/qwen/Qwen2.5-3B-Instruct/grpo/confg_full.yaml
