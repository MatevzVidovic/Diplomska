#!/bin/bash




# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att_res unet_att_res_train > x_ts_att_res.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att_res_j unet_att_res_j_train > x_ts_att_res_j.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att unet_att_train > x_ts_att.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att_j unet_att_j_train > x_ts_att_j.txt



# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh sclera unet_sclera_train > x_ts_sclera_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh sclera2 unet_sclera2_train > x_ts_sclera2_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh sclera_4_1.5_4 unet_sclera_4_1.5_4_train > x_ts_sclera_4_1.5_4_train.txt



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


python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_unet/unet_${id}.yaml    < "$test_showcase"