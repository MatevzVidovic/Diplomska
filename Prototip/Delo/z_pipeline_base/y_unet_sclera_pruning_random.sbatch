#!/bin/bash
#SBATCH --job-name=uspr
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100
#SBATCH --output=x_unet_sclera_pruning_random.txt



# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# ptd=$5
# ntibp=$6





srun bash z_pipeline_base/z1_pruning_random.sh main.py unet_sclera_pruning_random 90 48 ./sclera_data 5