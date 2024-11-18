#!/bin/bash
#SBATCH --job-name=IPAD
#SBATCH --time=0-02:00:00

#SBATCH -p frida
#SBATCH -w ana
#SBATCH -c 64
#SBATCH --gpus=A100_80GB
#SBATCH --output=x0_out.txt

# srun bash z_combining_script.sh

srun bash z_segnet.sh

#srun bash z_unet.sh
