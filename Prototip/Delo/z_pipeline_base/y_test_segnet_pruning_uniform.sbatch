#!/bin/bash
#SBATCH --job-name=tspu
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100_80GB
#SBATCH --output=x_test_segnet_pruning_uniform.txt






# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4

srun bash z_pipeline_base/z1_test_uniform.sh segnet_main.py test_segnet_pruning_uniform 48 48