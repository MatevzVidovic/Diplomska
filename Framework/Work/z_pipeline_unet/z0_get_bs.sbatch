#!/bin/bash

#SBATCH --job-name=ut
#SBATCH --time=0-5:00:00

#SBATCH -p frida
#SBATCH -c 32
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=4G
#SBATCH --output=x_unet_batch_size.txt




id=$1

param_num=1
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

source z0_sh_help.sh

source z_pipeline_unet/z0_main_name.sh

sd_name="unet_${1}_batch_size_train"
create_empty_folder $sd_name
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_unet/unet_${id}.yaml    < "$batch_size_train"

sd_name="unet_${1}_batch_size_eval"
create_empty_folder $sd_name
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_unet/unet_${id}.yaml    < "$batch_size_eval"