# Chain of Conscience: Dynamic Safety Framework for LLMs

An adaptive safety system that intelligently allocates test-time compute for enhanced LLM alignment through dynamic safety reasoning.

> **Note**: This project builds upon the [open-r1](https://github.com/huggingface/open-r1) project. The GRPO implementation and training recipes are derived from open-r1 and modified to suit our safety-focused requirements.

## Overview

This project introduces a *dynamic safety framework* for Large Language Models (LLMs) that:

- **Decouples Safety Reasoning from Instruction Following**: Separates the processes of understanding user instructions and evaluating their safety implications to enhance response reliability.​

- **Dynamically Allocates Additional Compute for Potentially Harmful Prompts:** Utilizes a budget allocator to assess the risk profile of each prompt and assigns appropriate computational resources for safety evaluation.​

- **Minimizes Computational Overhead for Safe Interactions:** Ensures that additional safety computations are performed only when necessary, maintaining efficiency.​

- **Enables Scalable Deployment of Safety Mechanisms:** Facilitates the integration of advanced safety features into existing LLM deployments without significant infrastructure changes.​

Current Status: The main LLM, equipped with integrated safety reasoning capabilities, has been developed and is operational. Development of the budget allocator component, responsible for dynamic compute allocation based on prompt risk assessment, is ongoing.


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
The project uses GRPO (adapted from open-r1) for training the safety framework. We support both DDP and DeepSpeed (ZeRO-2/ZeRO-3) training methods.

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
- Minimum (Current Setup): 4× A100 GPUs 
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

## Acknowledgments

This project significantly builds upon the work of the [open-r1](https://github.com/huggingface/open-r1) project. We are grateful to the open-r1 team for their excellent foundation, particularly:

- GRPO (Group Relative Policy Optimization) implementation
- Training recipes and configurations
- Infrastructure setup scripts

While I have modified these components to suit our safety-focused requirements, the core architecture draws from their work.

