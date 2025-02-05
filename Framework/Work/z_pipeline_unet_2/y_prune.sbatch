#!/bin/bash


#SBATCH --job-name=up
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
ifn=${2}
yo_ids=${3}
# how to pass yo_ids:  bash y_unet_test_quick.sbatch test_quick "small weak sthsth"


param_num=3
num_optional=1

sbatch_id=prune
sbatch_id=${sbatch_id}_${ifn}
protect_out_files='true'
retain_savedir='true'
#pass 'false' for false, anything else for true

pipeline_name=z_pipeline_unet_2
source z_boilerplate.sh 
# get out_name, sd_name, yo_paths




python3 $main_name --ifn ${ifn} -p --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths   >> $out_name 2>&1
