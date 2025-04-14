#!/bin/bash
#SBATCH --job-name=FINETUNE_SCRIPT_TEST
#SBATCH --gres=gpu:H100:1 --cpus-per-gpu=8
#SBATCH -t8:00:00
#SBATCH -oReport-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gogo979r@gmail.com

# Initialize Conda (Mamba uses the same initialization)
source ~/.bashrc  # Load your shell config
conda activate backdooring 

echo "Using Python at: $(which python)"

python3 finetune_unsloth.py