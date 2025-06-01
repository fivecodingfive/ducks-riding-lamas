#!/bin/bash
#SBATCH -J dqn_sweep
#SBATCH -D ./                                # Working directory
#SBATCH -o ./sweep_%A_%a.out                 # Stdout: job array ID and task ID
#SBATCH -e ./sweep_%A_%a.err                 # Stderr
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0 #-11                          # Run 12 sweep variants (adjust as needed)

# Load environment
module load slurm_setup
module load python/3.10.10-extended
source ../tfvenv/bin/activate

# W&B and plotting config
export MPLBACKEND=Agg
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

# Pass SLURM array ID into the script
echo ">>> Starting sweep run with SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

python -u main.py \
    --mode training \
    --variant 0 \
    --episodes 10 #400 \
    --sweep_id $SLURM_ARRAY_TASK_ID
