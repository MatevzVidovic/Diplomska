#!/bin/bash


#SBATCH --job-name=upI
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_unet_prune_IPAD.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.




# bs 4 gre, bs 6 ne gre 
# bs 5 gre


srun python3 unet_original_main.py --ifn IPAD_eq -p --ptd ./Data/vein_and_sclera_data --sd unet_prune_IPAD --yaml z_pipeline_unet/unet_original_0.yaml

