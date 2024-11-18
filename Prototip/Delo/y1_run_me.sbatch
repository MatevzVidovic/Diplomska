#!/bin/bash
#SBATCH --job-name=IPAD
#SBATCH --time=0-02:00:00

#SBATCH -p frida
#SBATCH -w aga1
#SBATCH -c 64
#SBATCH --gpus=A100
#SBATCH --output=x1_out.txt

# srun bash z_combining_script.sh

srun bash z_unet.sh

# srun bash z_segnet.sh
