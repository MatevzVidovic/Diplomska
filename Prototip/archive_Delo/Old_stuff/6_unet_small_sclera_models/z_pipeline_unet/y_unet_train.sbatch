#!/bin/bash


#SBATCH --job-name=ut
#SBATCH --time=0-5:00:00

#SBATCH -p frida
#SBATCH -c 32
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

counter=0
out_name=x_unet_${id}_train_${counter}.txt
# -e for file exists, -L for symlink exists
while [[ -e "$out_name" || -L "$out_name" ]]; do
    # Increment the counter and create a new filename
    ((counter++))
    out_name=x_unet_${id}_train_${counter}.txt
done

source z_pipeline_unet/z0_main_name.sh

srun --output=${out_name} python3 ${main_name} --mti ${mti} --sd unet_${id}_train --yaml z_pipeline_unet/unet_${id}.yaml


