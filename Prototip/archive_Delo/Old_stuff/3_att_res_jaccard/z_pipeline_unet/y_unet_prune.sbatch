#!/bin/bash


#SBATCH --job-name=up
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


id=${1}
pr_type=${2} # can be uniform, random, IPAD_eq


param_num=2

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

source z_pipeline_unet/z0_main_name.sh

srun --output=x_unet_prune_${pr_type}.txt python3 ${main_name} --ifn ${pr_type} -p --sd unet_${id}_prune_${pr_type} --yaml z_pipeline_unet/unet_${id}.yaml
