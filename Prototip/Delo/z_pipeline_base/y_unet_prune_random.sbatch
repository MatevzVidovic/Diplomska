#!/bin/bash


#SBATCH --job-name=upr
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_unet_prune_random.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.



# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# lr=$5
# ptd=$6
# iw=$7
# ih=$8
# model_name=$9
# tesl=${10}
# ntibp=${11}


# bs 4 gre, bs 6 ne gre 
# bs 5 gre

srun bash z_pipeline_base/z1_pruning_random.sh unet_original_pruning_main.py unet_prune_random 2 7 1e-4 ./vein_sclera_data 2048 1024 64_2_6 400 20


