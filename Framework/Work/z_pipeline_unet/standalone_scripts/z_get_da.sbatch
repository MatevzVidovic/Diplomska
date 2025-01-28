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





# srun -c 2 --gpus=A100:1 bash z_pipeline_unet/z0_get_graphs.sh name_of_sd > x_get_graphs.txt









yaml_id=${1}
sd_name=${2}
yo_ids=${3}
# how to pass yo_ids:  bash y_unet_test_quick.sbatch test_quick "small weak sthsth"

param_num=3
num_optional=1
pipeline_name=z_pipeline_unet


source ${pipeline_name}/z_constants.sh
source z_sh_help.sh
check_param_num $param_num $num_optional "$@"
yo_paths=$(get_yo_paths ${pipeline_name} "$yo_ids")


python3 $main_name --ips 0 --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths    < "$data_aug"