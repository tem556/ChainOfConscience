# Chain of Conscience: Dynamic Safety Framework for LLMs

An adaptive safety system that intelligently allocates test-time compute for enhanced LLM alignment through dynamic safety reasoning.

---

## Overview

This project implements a **safety framework** that:
- Decouples safety reasoning from instruction following  
- Dynamically allocates additional compute for potentially harmful prompts  
- Uses **GRPO (Guided Reward Policy Optimization)** for training  
- Minimizes computational overhead for safe interactions  
- Enables scalable deployment of safety mechanisms  

---

## Installation

### Requirements
- Python 3.11  
- CUDA 12.1  
- Git LFS  

### Setup Environment
1. **Create a Python virtual environment** using `uv` (or any preferred method).
2. **Install vLLM**.
3. **Install project dependencies**.
4. **Login** to Hugging Face and Weights & Biases.
5. **Install Git LFS** if not present.

```bash
# Example commands (adjust as needed)

# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install vLLM (example)
pip install vllm

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Log in to Hugging Face and W&B
huggingface-cli login
wandb login

# 5. Install Git LFS
sudo apt-get install git-lfs
git lfs install
```

## Usage
### Training
The project uses GRPO for training the safety framework. We support both DDP and DeepSpeed (ZeRO-2/ZeRO-3) training methods.

```bash
# Basic training command (customize based on your environment)
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=3 \
    src/safety_grpo.py \
    --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml
```
### Using SLURM
For cluster environments, use the provided SLURM script:

```bash
sbatch --output=/path/to/logs/%x-%j.out \
       --error=/path/to/logs/%x-%j.err \
       slurm/train.slurm
```
### Configuration
The project uses YAML configuration files located in the `qwen` directory. Key configurations include:

Model configurations (in config_full.yaml): Defines model architecture, training hyperparameters, and resource requirements.


### Hardware Requirements
- Recommended: 8× H100 GPUs (80GB)
- Minimum: 4× A100 GPUs
- For different hardware configurations, adjust batch size and gradient accumulation steps in the config files accordingly.

### Project Structure
```
├─ qwen/
│   ├─ configs/
│   │   ├─ config_full.yaml
│   │   └─ ...
│   ├─ ...
├─ scripts/
│   ├─ slurm_job_script.sh
│   └─ ...
├─ train.py
├─ requirements.txt
└─ README.md
```

