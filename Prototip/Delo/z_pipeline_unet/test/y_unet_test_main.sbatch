#!/bin/bash


#SBATCH --job-name=tm
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=1
#SBATCH --constraint=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_test_main.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


source z0_bash_saver.sh
source z0_sh_help.sh


create_empty_folder test_main_vein
python3 unet_original_main.py --tras 3 --mti 2 --ptd ./vein_sclera_data --sd test_main_vein --yaml z_pipeline_unet/unet_original_0.yaml           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))



# create_empty_folder test_main_sclera
# python3 unet_original_main.py --tras 3 --mti 2 --ptd ./sclera_data --sd test_main_sclera --yaml z_pipeline_unet/unet_original_0.yaml           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
