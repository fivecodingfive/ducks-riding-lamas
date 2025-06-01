#!/bin/bash
#SBATCH -J dqntrain
#SBATCH -D ./
#SBATCH -o ./dqntrain.%j.out
#SBATCH -e ./dqntrain.%j.err
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

module load slurm_setup
module load python/3.10.10-extended
source ../tfvenv/bin/activate

export MPLBACKEND=Agg
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH/wandb

python -u main.py \
    --mode training \
    --variant 0 \
    --episodes 5