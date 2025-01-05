#!/bin/bash
#SBATCH --job-name=spu
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100
#SBATCH --output=x_segnet_pruning_uniform.txt


# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# ptd=$5
# ntibp=$6





srun bash z_pipeline_base/z1_pruning_uniform.sh segnet_main.py segnet_pruning_uniform 210 64 ./vein_sclera_data 20