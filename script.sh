#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=1-00:00:00
#SBATCH --job-name=VAE
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --partition=gpu02
#SBATCH --gres=gpu:1

module load python/3.8

source activate vae-rs

export SCRATCH="/scratch/disc/e.bardet/"

python train.py

