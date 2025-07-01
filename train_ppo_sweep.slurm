#!/bin/bash
#SBATCH -J ppo_sweep
#SBATCH -D .                                # Working directory
#SBATCH -o logs/sweep_%A_%a.out            # Stdout
#SBATCH -e logs/sweep_%A_%a.err            # Stderr
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-44%10                     # 45 tasks (3*2*3*3*2 = 45), max 10 running at once

# Load environment
module load slurm_setup
module load python/3.10.10-extended
source ../tfvenv/bin/activate

# Make sure our log directory exists
mkdir -p logs

# W&B offline & plotting
export MPLBACKEND=Agg
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb
mkdir -p "$WANDB_DIR"

# Use all allocated CPUs for OMP/BLAS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ">>> Starting PPO sweep task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

srun python -u main_ppo.py \
    --mode training \
    --episodes 5 \
    --variant 0 \
    --sweep_id $SLURM_ARRAY_TASK_ID

echo ">>> Sweep task $SLURM_ARRAY_TASK_ID done at $(date)"
