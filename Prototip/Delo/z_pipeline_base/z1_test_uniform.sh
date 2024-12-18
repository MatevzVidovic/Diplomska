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


# Creates temp files save_and_stop, results_and_stop, and graph_and_stop, resource_graph_and_stop
source z_pipeline_base/z0_temp_inputs.sh







# Function to delete a folder if it exists and then create it empty
create_empty_folder() {
    local folder_name="$1"  # Access the first parameter

    if [ -d "$folder_name" ]; then
        rm -r "$folder_name"
    fi

    mkdir "$folder_name"
}





# main_name=$1
# bs=$2
# nodw=$3
# iw=$4
# ih=$5
# model_name=$6
# param_num=6

main_name=$1
bs=$2
nodw=$3
iw=$4
ih=$5
model_name=$6
param_num=6

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

echo $@


# additional use of z_bash_saver.sh
program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"







# Test pruning:

# Set min weights for layers to 0.9999.
# Then prune 10 layers each round of pruning, and then save the graph.
# In the end, each layer should have only one kernel or one input slice pruned.



tesl=$((bs + bs / 2))
tras=$((bs * 2))

create_empty_folder test_uniform



# training phase
python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$save_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

python3 ${main_name} --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --mti 2 --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras}            2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$save_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


# pruning_phase
python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$graph_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

for i in {1..30}; do
    python3 ${main_name} --tp --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1            2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
    python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$save_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
    python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$graph_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
done

python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$resources_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 ${main_name} --ips 0 --bs ${bs} --nodw ${nodw} --sd test_uniform --ptd ./vein_sclera_data --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras} --ntibp 2 --pruning_phase --pbop --map 1 --pnkao 1 --rn flops_num --ptp 0.00001  --ifn 1  < "$results_and_stop"              2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


