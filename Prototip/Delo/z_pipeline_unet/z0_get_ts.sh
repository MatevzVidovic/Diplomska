#!/bin/bash




# srun -c 2 --gpus=A100:1 bash z_pipeline_unet/z0_get_graphs.sh name_of_sd > x_get_graphs.txt

id=$1
sd_name=$2

param_num=2
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi
echo "sd_name: $sd_name"


source z0_sh_help.sh

source z_pipeline_unet/z0_main_name.sh


python3 ${main_name} --ips 0 --ptd ./Data/vein_and_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$test_showcase"