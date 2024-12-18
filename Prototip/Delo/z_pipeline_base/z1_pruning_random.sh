#!/bin/bash


# The script is meant for automatic storing of outputs of programs.

# This script creates bash_results/curr/ and bash_results/older
# It moves what is in curr into older.
# Then it writes into curr the copy of the current .sh file,
# and the time at the start of it's execution.

# It then enables you to save the output of your commands by simply pasting this line:
# # [command]    2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
# This will then write to a file in bash_results/curr, while also printing to stdout.

# For this, please do not use:
# command_num=0
# obn=${output_basic_name}
# rfn=${results_folder_name}
# cbi=${curr_bash_ix}
# cn=${command_num}

source z_pipeline_base/z0_bash_saver.sh


# Creates temp files save_and_stop, results_and_stop, and graph_and_stop
source z_pipeline_base/z0_temp_inputs.sh


# additional use of z_bash_saver.sh
program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"







main_name=$1
folder_name=$2
bs=$3
nodw=$4
lr=$5
ptd=$6
iw=$7
ih=$8
model_name=$9
tesl=${10}
mti=${11}
ntibp=${12}
param_num=12

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

echo $@


# additional use of z_bash_saver.sh
program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"




# Pruning:

python3 ${main_name} --bs ${bs} --nodw ${nodw} --sd ${folder_name} --ptd ${ptd} --mti ${mti} --lr ${lr} --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --ntibp ${ntibp} --pruning_phase --pbop --map 70 --rn flops_num --ptp 0.01 --ifn 0            2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))



