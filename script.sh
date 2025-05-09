#!/bin/sh

#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=Conditionnal_VAE
#SBATCH -o ./slurm_logs/slurm.%j.out # STDOUT
#SBATCH -e ./slurm_logs/slurm.%j.err # STDERR
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu02
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

module load python/3.8

source activate vae-rs

export SCRATCH="/scratch/disc/e.bardet/"

python train.py --patch_size 64 --batch_size 2 --pre_epochs 0 --model_ckpt "pre_best_model_3846651.pth"

