#!/bin/bash


#SBATCH --job-name=te
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100

#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_test_everything.txt






main=unet_original_pruning_main.py
#bs=1
model_name=64_2_6

##main=unet_main.py
bs=1
#model_name=64_2_4





# main_name=$1
# bs=$2
# nodw=$3
# iw=$4
# ih=$5
# model_name=$6
# pnkao=$7
# param_num=7

srun bash z_pipeline_base/z1_test_pruning.sh ${main} 1 7 2048 1024 ${model_name} 400



# main_name=$1
# bs=$2
# nodw=$3
# iw=$4
# ih=$5
# model_name=$6

# srun bash z_pipeline_base/z1_test_main.sh ${main} 1 7 2048 1024 ${model_name}



# main_name=$1
# bs=$2
# nodw=$3
# iw=$4
# ih=$5
# model_name=$6
# pnkao=$7
# param_num=7

srun bash z_pipeline_base/z1_test_basic.sh ${main} 1 7 2048 1024 ${model_name} 400


# main_name=$1
# bs=$2
# nodw=$3
# iw=$4
# ih=$5
# model_name=$6
# param_num=6

# srun bash z_pipeline_base/z1_test_uniform.sh ${main} 1 7 2048 1024 ${model_name}