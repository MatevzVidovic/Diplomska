#!/bin/bash


# When you source without passing stuff, $@ is passed to the sourced file.
# source z_sh_help.sh    # the $@ is the same in z_sh_help.sh
# source z_sh_help.sh "sth"    # in sh_help, $@ is just "sth"
# source z_sh_help.sh "$@"    # this does the same as the first line "$@" is a special thing that retains the args exactly the same
# source z_sh_help.sh "$@" "sth"    # all @ and also sth are the $@ of z_sh_help.sh



source z_sh_help.sh
source $pipeline_name/z_constants.sh

check_param_num $param_num $num_optional "$@"
yo_paths=$(get_yo_paths ${pipeline_name} "$yo_ids")
yo_str=$(get_yo_str "$yo_ids")
echo "yo_paths: $yo_paths"

sd_name=${sbatch_id}_${yaml_id}${yo_str}
if [[ $retain_savedir == "false" ]]; then
    echo "Maybe overwriting ${sd_name}." >&2
    create_empty_folder $sd_name
fi

base_name=x_${model}_${sd_name}
out_name=$(get_out_name $base_name $protect_out_files)
create_empty_file $out_name
