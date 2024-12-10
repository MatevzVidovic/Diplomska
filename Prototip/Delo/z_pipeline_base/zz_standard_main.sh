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





main_name=$1
folder_name=$2
bs=$3
nodw=$4
pnkao=$5
ptd=$6

param_num=6

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct."
    exit 1
fi



# additional use of z_bash_saver.sh
program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"












# Main training:

# te so itak hitri, ker majhen train set
python3 ${main_name} --ips 999999 --bs ${bs} --nodw ${nodw} --ntibp 1 --sd ${folder_name} --ptd ${ptd} --mti 50          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# torej (2 tr + k epoch passov) * 15 = 45 epochov
# 4 tr * 15 = 60 trainingov
# vsak pruning pa ima vsakih 100 kernelov en epoch pass
# Rezimo torej še 7 epoch passov
python3 ${main_name} --ips 999999 --bs ${bs} --nodw ${nodw} --ntibp 4 --sd ${folder_name} --ptd ${ptd} --pruning_phase --pbop --map 15 --pnkao ${pnkao} --rn flops_num --ptp 0.05          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))




