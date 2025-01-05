#!/bin/bash
#SBATCH --job-name=ts
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100
#SBATCH --output=x_test_segnet.txt


# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# ptd=$5

srun bash z_pipeline_base/z1_test_basic.sh segnet_main.py test_segnet 56 48 ./vein_sclera_data