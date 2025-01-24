#!/bin/bash


#SBATCH --job-name=utfpu
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_deeplab_test_full_pruning_uniform.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


source z0_sh_help.sh


# You should give this script an already trained model. (copy the trained model folder and rename it to test_full_pruning_uniform).
# The point is to see what the result of the pruning will be. This also allows you to make and test out the porter script.

sd_name="test_full_pruning_uniform"

id=${1}

param_num=1
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

source z_pipeline_deeplab/z0_main_name.sh

# pruning_phase
python3 ${main_name} --ntibp 1 --tras 2 --ifn uniform -p --sd $sd_name --yaml z_pipeline_deeplab/deeplab_${id}.yaml

python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_deeplab/deeplab_${id}.yaml    < "$graph_and_stop"
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_deeplab/deeplab_${id}.yaml    < "$results_and_stop"
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_deeplab/deeplab_${id}.yaml    < "$resource_graph_and_stop"
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_deeplab/deeplab_${id}.yaml    < "$save_and_stop"


