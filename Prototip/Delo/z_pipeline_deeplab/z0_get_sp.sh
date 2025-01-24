#!/bin/bash




# srun -c 10 --gpus=A100:1 bash z_pipeline_deeplab/z0_get_sp.sh sclera2 deeplab_sclera2_train > x_sp_sclera2_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_deeplab/z0_get_sp.sh small_sclera_recall_2 deeplab_small_sclera_recall_train > x_sp_small_sclera_recall.txt



id=$1
sd_name=$2

param_num=2
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi
echo "sd_name: $sd_name"


source z0_sh_help.sh

source z_pipeline_deeplab/z0_main_name.sh


python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_deeplab/deeplab_${id}.yaml    < "$save_preds"