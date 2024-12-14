#!/bin/bash


#SBATCH --job-name=uppM
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

# Pri 3000 2000, unet_3000_2000, starting kernels 5, expansion 1.3,   je maksimalen bs 17

# Pri 3000 2000, unet_3000_2000_input_skip_conn, starting kernels 5, expansion 1.3,   je maksimalen bs 13





srun --output=x_unet_partial_preaug_MCDL${1}.txt  bash z_pipeline_base/z1_main_MCDL.sh unet_main.py unet_partial_preaug_MCDL${1} 15 18 80 1e-2 ./partial_preaug_vein_sclera_data 3000 2000 partially_preaugmented UNet_3000_2000 400


