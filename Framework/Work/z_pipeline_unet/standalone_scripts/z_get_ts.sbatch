#!/bin/bash


#SBATCH --job-name=utu
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.










# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att_res unet_att_res_train > x_ts_att_res.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att_res_j unet_att_res_j_train > x_ts_att_res_j.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att unet_att_train > x_ts_att.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh att_j unet_att_j_train > x_ts_att_j.txt



# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh sclera unet_sclera_train > x_ts_sclera_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh sclera2 unet_sclera2_train > x_ts_sclera2_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh sclera_4_1.5_4 unet_sclera_4_1.5_4_train > x_ts_sclera_4_1.5_4_train.txt

# srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh small_sclera_recall_2 unet_small_sclera_recall_train > x_ts_small_sclera_recall.txt









yaml_id=${1}
sd_name=${2}
yo_ids=${3}
# how to pass yo_ids:  bash y_unet_test_quick.sbatch test_quick "small weak sthsth"

param_num=3
num_optional=1
pipeline_name=z_pipeline_unet


source z_sh_help.sh
source ${pipeline_name}/z_constants.sh
check_param_num $param_num $num_optional "$@"
yo_paths=$(get_yo_paths ${pipeline_name} "$yo_ids")



python3 $main_name --ips 0 --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths    < "$test_showcase"