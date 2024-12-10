#!/bin/bash
#SBATCH --job-name=uspI
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100
#SBATCH --output=x_unet_sclera_pruning_IPAD.txt







# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5
# ptd=$6
# ntibp=$7


srun bash z_pipeline_base/z1_pruning_strong_main.sh main.py unet_sclera_pruning_IPAD 90 48 80 ./sclera_data 5
