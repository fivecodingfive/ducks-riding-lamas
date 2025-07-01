#!/bin/bash
#SBATCH --job-name=ppo_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-89            # 90 combinations (3*2*3*3*2*2 = 90)
#SBATCH --output=logs/sweep_%A_%a.out

# Load required modules (adjust to match LRZ environment)
module load cuda/12.1

# Activate virtual environment
source ~/venvs/rl/bin/activate

# Set up W&B offline mode for cluster runs
export WANDB_MODE="offline"
export WANDB_DIR="$SCRATCH/wandb"
mkdir -p $WANDB_DIR

# Run the PPO training with sweep configuration
# The hyperparameters are set inside the script based on SLURM_ARRAY_TASK_ID
srun python -u main_ppo.py \
    --variant 0 \
    --mode training \
    --episodes 5 \
    --seed $((123 + $SLURM_ARRAY_TASK_ID))

echo "Sweep task $SLURM_ARRAY_TASK_ID completed at $(date)"
