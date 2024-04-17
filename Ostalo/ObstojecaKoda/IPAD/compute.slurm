#!/usr/bin/env bash

#SBATCH --job-name=ge_compute

## 128 is the maximum number of CPUs we have available
#SBATCH --cpus-per-task=128

## MaxRSS was 20GB (on 128 cores, but cores have very little impact), so 32GB should be plenty
#SBATCH --mem=256GB

## We need just one GPU for the Keras model predictions
##SBATCH --gpus=1

## With recognition experiments, takes roughly 1 day per model (15 train-test configs), but 7 days is the max we can request
#SBATCH --time=7-00

srun \
	--container-image "$HOME/Containers/GE.sqfs" \
	--container-mounts "/shared/datasets/fri/mvitek:/data" \
	--container-workdir "/root/GE" \
	compute.py "${1:-/data/GE/Models}" "${2:-/data/GE/Datasets}"
