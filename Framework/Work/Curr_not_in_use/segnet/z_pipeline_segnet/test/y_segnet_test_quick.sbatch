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


source z0_sh_help.sh

create_empty_folder test_quick


id=${1}

param_num=1
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi


source z_pipeline_segnet/z0_main_name.sh

# training phase
srun --output=x_segnet_test_quick_${id}.txt python3 ${main_name} --tras 2 --mti 2 --sd test_quick_${id} --yaml z_pipeline_segnet/segnet_${id}.yaml --yo z_pipeline_segnet/overriding_yamls/small.yaml


# pruning_phase
#for i in {1..1}; do
 #   python3 ${main_name} --ntibp 1 --ptp 0.0000001 --map 1 --tras 2 --tp --ifn uniform -p --sd test_quick --yaml z_pipeline_segnet/segnet_${id}.yaml
#done
