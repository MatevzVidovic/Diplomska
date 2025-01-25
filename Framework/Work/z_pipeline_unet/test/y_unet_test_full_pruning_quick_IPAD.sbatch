#!/bin/bash


#SBATCH --job-name=utfpqI
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_unet_test_full_pruning_quick_IPAD.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


source z0_sh_help.sh


# You should give this script an already trained model. (copy the trained model folder and rename it to test_full_pruning_IPAD).
# The point is to see what the result of the pruning will be. This also allows you to make and test out the porter script.

sd_name="test_full_pruning_quick_IPAD"

id=${1}

param_num=1
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

source z_pipeline_unet/z0_main_name.sh

# pruning_phase
python3 ${main_name} --ntibp 1 --tras 2 --ifn IPAD_eq -p --sd $sd_name --yaml z_pipeline_unet/unet_${id}.yaml --yo z_pipeline_unet/overriding_yamls/small.yaml

bash z_pipeline_unet/z0_get_graphs.sh $sd_name
python3 ${main_name} --ips 0 --sd $sd_name --yaml z_pipeline_unet/unet_${id}.yaml  --yo z_pipeline_unet/overriding_yamls/small.yaml  < "$save_and_stop"


