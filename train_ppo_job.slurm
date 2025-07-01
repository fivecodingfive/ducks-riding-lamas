#!/bin/bash
#SBATCH --job-name=ppo_single
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Load required modules (adjust to match LRZ environment)
module load cuda/12.1

# Activate virtual environment
source ~/venvs/rl/bin/activate

# Set up W&B offline mode for cluster runs
export WANDB_MODE="offline"
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p $WANDB_DIR

# Run the PPO training with default hyperparameters
srun python -u main_ppo.py \
    --variant 0 \
    --mode training \
    --episodes 1000 \
    --policy_lr 3e-4 \
    --value_lr 1e-3 \
    --clip 0.2 \
    --entropy 0.05 \
    --lam 0.95 \
    --seed 123

echo "Job completed at $(date)"
