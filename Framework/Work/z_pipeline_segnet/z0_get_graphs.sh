#!/bin/bash




# srun -c 2 --gpus=A100:1 bash z_pipeline_segnet/z0_get_graphs.sh name_of_sd > x_get_graphs.txt
id=$1
sd_name=$2

param_num=2
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

echo "sd_name: $sd_name"


source z0_sh_help.sh

source z_pipeline_segnet/z0_main_name.sh

python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_segnet/segnet_${id}.yaml    < "$graph_and_stop"
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_segnet/segnet_${id}.yaml    < "$results_and_stop"
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_segnet/segnet_${id}.yaml    < "$resource_graph_and_stop"