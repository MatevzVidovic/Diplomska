#!/bin/bash




# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh att_res segnet_att_res_train > x_ts_att_res.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh att_res_j segnet_att_res_j_train > x_ts_att_res_j.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh att segnet_att_train > x_ts_att.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh att_j segnet_att_j_train > x_ts_att_j.txt



# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh sclera segnet_sclera_train > x_ts_sclera_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh sclera2 segnet_sclera2_train > x_ts_sclera2_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh sclera_4_1.5_4 segnet_sclera_4_1.5_4_train > x_ts_sclera_4_1.5_4_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_segnet/z0_get_ts.sh small_sclera_recall_2 segnet_small_sclera_recall_train > x_ts_small_sclera_recall.txt




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


python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_segnet/segnet_${id}.yaml    < "$test_showcase"