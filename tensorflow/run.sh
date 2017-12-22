#!/bin/env bash

#SBATCH --cpus-per-task=8
#SBATCH --job-name=aa_tf
#SBATCH --partition=shared-gpu
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:titan

srun python tf.py bbc.mat -e 4000
