#!/bin/bash


#SBATCH --job-name=sppM
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 18
#SBATCH --gpus=A100

#SBATCH -w axa
#SBATCH --mem-per-cpu=8G 



# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.

# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5
# lr=$6
# ptd=$7
# iw=$8
# ih=$9
# ds=${10}
# model_name=${11}
# tesl=${12}



# Pri 3000 2000, unet_3000_2000, starting kernels 5, expansion 1.3,
# Za 15 dela. Za 20 nedela. Zdej se pa pejt. Bojan ma že 1 tedn vse nodee.


srun --output=x_segnet_partial_preaug_MCDL${1}.txt  bash z_pipeline_base/z1_main_MCDL.sh segnet_main.py segnet_partial_preaug_MCDL${1} 17 18 80 1e-2 ./partial_preaug_vein_sclera_data 3000 2000 partially_preaugmented SegNet_3000_2000 50


