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
#SBATCH --array=0-143  

set -euxo pipefail              # fail fast

# ─── modules & venv ───────────────────────────────────────────────────────────
module load slurm_setup
module load python/3.10.12-extended
source ~/venvs/ppo/bin/activate

# ─── logging / plotting ───────────────────────────────────────────────────────
# plain matplotlib still fine
mkdir -p logs

# ─── TensorBoard log dir (one subfolder per array task) ──────────────────────
export TB_ROOT=$SCRATCH/tb
export TB_LOGDIR=$TB_ROOT/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$TB_LOGDIR"

# ─── BLAS threading ──────────────────────────────────────────────────────────
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo ">>> Starting PPO sweep task $SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"

srun python -u main_ppo.py \
     --mode training \
     --episodes 400 \
     --variant 1 \
     --sweep_id $SLURM_ARRAY_TASK_ID \

echo ">>> Sweep task $SLURM_ARRAY_TASK_ID done at $(date)"
echo "    Offline W&B run stored at $WANDB_DIR"
