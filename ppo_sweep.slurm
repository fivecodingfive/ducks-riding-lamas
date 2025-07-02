#!/bin/bash
#SBATCH -J ppo_sweep
#SBATCH -D .
#SBATCH -o logs/sweep_%A_%a.out
#SBATCH -e logs/sweep_%A_%a.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-44%10            # 45 configs, 10 in parallel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jonas05.coding@gmail.com

set -euo pipefail                  # fail fast

# ─── modules & venv ───────────────────────────────────────────────────────────
module load slurm_setup
module load python/3.10.10-extended
source ../tfvenv/bin/activate

# ─── logging / plotting ───────────────────────────────────────────────────────
mkdir -p logs
export MPLBACKEND=Agg

# ─── Weights & Biases (OFFLINE) ──────────────────────────────────────────────
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb            # fast local storage
export WANDB_PROJECT=ride-those-llamas
export WANDB_ENTITY=ducks-riding-llamas
mkdir -p "$WANDB_DIR"

# ─── BLAS threading ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ">>> Starting PPO sweep task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

srun python -u main_ppo.py \
     --mode training \
     --episodes 5 \
     --variant 0 \
     --sweep_id $SLURM_ARRAY_TASK_ID

echo ">>> Sweep task $SLURM_ARRAY_TASK_ID done at $(date)"
echo "    Offline W&B run stored at $WANDB_DIR"
