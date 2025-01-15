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



srun python3 unet_original_main.py --ifn random -p --ptd ./vein_sclera_data --sd unet_prune_random --yaml z_pipeline_unet/unet_original_0.yaml


