#!/bin/bash


#SBATCH --job-name=ut
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G 


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.





# bs 4 gre, bs 6 ne gre 
# bs 5 gre


srun --output=x_unet_train${1}.txt python3 unet_original_main.py --mti ${2} --ptd ./vein_sclera_data --sd unet_train${1} --yaml z_pipeline_unet/unet_original_0.yaml


