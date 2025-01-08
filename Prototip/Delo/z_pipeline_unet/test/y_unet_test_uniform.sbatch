#!/bin/bash


#SBATCH --job-name=utu
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_unet_test_uniform.txt.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


source z0_sh_help.sh

create_empty_folder test_uniform

id=${1}

param_num=1
if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi




# training phase
python3 unet_original_main.py --ips 0 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$save_and_stop"
python3 unet_original_main.py --tras 2 --mti 2 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml
python3 unet_original_main.py --ips 0 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$save_and_stop"


# pruning_phase
for i in {1..30}; do
    python3 unet_original_main.py --ntibp 1 --ptp 0.0000001 --map 1 --tras 2 --tp --ifn uniform -p --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml
    python3 unet_original_main.py --ips 0 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$graph_and_stop"
done

python3 unet_original_main.py --ips 0 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$results_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$resource_graph_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./Data/vein_and_sclera_data --sd test_uniform --yaml z_pipeline_unet/unet_original_${id}.yaml    < "$save_and_stop"


