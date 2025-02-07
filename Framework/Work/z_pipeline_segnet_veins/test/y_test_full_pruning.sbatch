#!/bin/bash


#SBATCH --job-name=utfpI
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G
#SBATCH --output=x_unet_test_full_pruning_IPAD.txt


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.




# how to pass yo_ids:  bash y_unet_test_quick.sbatch test_quick "small weak sthsth"


yaml_id=${1}
ifn=${2}
yo_ids=${3}
# how to pass yo_ids:  bash y_unet_test_quick.sbatch test_quick "small weak sthsth"

param_num=3
num_optional=2


sbatch_id=test_full_pruning


if [[ $ifn == "" ]]; then
    ifn=$main_ifn
fi
sbatch_id=${sbatch_id}_${ifn}

protect_out_files='true'
retain_savedir='false'
#pass 'false' for false, anything else for true

pipeline_name=z_pipeline_segnet_veins
source z_boilerplate.sh 
# get out_name, sd_name, yo_paths

echo "ifn: $ifn"
echo "yo_ids: $yo_ids"
echo "yo_paths: $yo_paths"

# pruning_phase
python3 $main_name --ntibp 1 --tras 2 --ifn ${ifn} -p --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths   >> $out_name 2>&1

bash ${pipeline_name}/standalone_scripts/z_get_graphs.sbatch $yaml_id $sd_name "$yo_ids"    >> $out_name 2>&1
python3 $main_name --ips 0 --sd $sd_name --yaml ${pipeline_name}/${yaml_id}.yaml --yo $yo_paths   >> $out_name 2>&1    < "$save_and_stop"


