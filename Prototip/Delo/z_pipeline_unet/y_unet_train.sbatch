#!/bin/bash


#SBATCH --job-name=ut
#SBATCH --time=0-5:00:00

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

id=${1}
mti=${2}

param_num=2

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

srun --output=x_unet_${id}_train.txt python3 unet_original_main.py --mti ${mti} --ptd ./Data/vein_and_sclera_data --sd unet_${id}_train --yaml z_pipeline_unet/unet_original_${id}.yaml


