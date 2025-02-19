#!/bin/bash
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH --partition=gpu-all
#SBATCH --job-name=llama-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=50:00
#SBATCH --output=slurm_%j.log

# Load the correct CUDA module
module load cuda12.2/toolkit/12.2.1

# Source conda setup (load conda functions into current shell)
source ~/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate coc

# Set environment variables
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Run the evaluation
python src/evaluate/struct_bench.py
