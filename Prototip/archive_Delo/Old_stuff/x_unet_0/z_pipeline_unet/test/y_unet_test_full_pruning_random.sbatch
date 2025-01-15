#!/bin/bash


#SBATCH --job-name=utfpr
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_unet_test_full_pruning_random.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


source z0_sh_help.sh


# You should give this script an already trained model. (copy the trained model folder and rename it to test_full_pruning_random).
# The point is to see what the result of the pruning will be. This also allows you to make and test out the porter script.

sd_name="test_full_pruning_random"


# pruning_phase
python3 unet_original_main.py --ntibp 1 --tras 2 --ifn random -p --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml

python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$graph_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$results_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$resource_graph_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$save_and_stop"


