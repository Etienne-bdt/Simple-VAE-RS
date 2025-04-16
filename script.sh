#!/bin/sh

#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=VAE
#SBATCH -o ./slurm_logs/slurm.%j.out # STDOUT
#SBATCH -e ./slurm_logs/slurm.%j.err # STDERR
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2

module load python/3.8

source activate vae-rs

export SCRATCH="/scratch/disc/e.bardet/"

srun python train.py

