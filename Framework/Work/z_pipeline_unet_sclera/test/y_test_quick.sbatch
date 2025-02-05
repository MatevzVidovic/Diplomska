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





yaml_id=${1}
yo_ids=${2}
# how to pass yo_ids:  bash y_unet_test_quick.sbatch test_quick "small weak sthsth"



param_num=2
num_optional=1

sbatch_id=test_quick
protect_out_files='false'
retain_savedir='false'
#pass 'false' for false, anything else for true

pipeline_name=z_pipeline_unet
source z_boilerplate.sh 
# get out_name, sd_name, yo_paths



# training phase
python3 $main_name --tras 2 --mti 2 --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths   >> $out_name 2>&1


python3 $main_name --ntibp 1 --ptp 0.0000001 --map 2 --tras 2 --tp --ifn uniform -p --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths   >> $out_name 2>&1


# pruning_phase
#for i in {1..1}; do
#done
