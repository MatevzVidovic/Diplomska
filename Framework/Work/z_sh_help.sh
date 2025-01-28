#!/bin/bash


# Creates temp files that can be used as input. 
# E.g.:
# python3 ${main_file} < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

save_and_stop=$(mktemp)
printf "s\nstop\n" > "$save_and_stop"

results_and_stop=$(mktemp)
printf "r\nstop\n" > "$results_and_stop"

graph_and_stop=$(mktemp)
printf "g\n\nstop\n" > "$graph_and_stop"    # needs another \n because of input("wait")

resource_graph_and_stop=$(mktemp)
printf "resource_graph\nstop\n" > "$resource_graph_and_stop"

test_showcase=$(mktemp)
printf "ts\nall\nstop\nstop\n" > "$test_showcase"

data_aug=$(mktemp)
printf "da\n\n\n\n\n1\n\n\n\n\n2\n\n\n\n\nstop\nstop\n" > "$data_aug"

save_preds=$(mktemp)
printf "sp\nstop\nstop\n" > "$save_preds"


batch_size_train=$(mktemp)
printf "bst\nstop\nstop\n" > "$batch_size_train"

batch_size_eval=$(mktemp)
printf "bse\nstop\nstop\n" > "$batch_size_eval"



# Function to delete a folder if it exists and then create it empty
create_empty_folder() {
    local folder_name="$1"  # Access the first parameter

    if [ -d "$folder_name" ]; then
        rm -r "$folder_name"
    fi

    mkdir "$folder_name"
}

create_empty_file() {
    local file_name="$1"  # Access the first parameter

    if [ -f "$file_name" ]; then
        rm "$file_name"
    fi

    touch "$file_name"
}




# Check if number of arguments is within acceptable range
check_param_num() {

    local param_num=$1
    local num_optional=$2
    shift 2  # Remove first two parameters
    local params=("$@")  # Store remaining args as array
    local real_num_params=${#params[@]}  # Get array length
    local min_param_num=$((param_num - num_optional))

    if [[ $real_num_params -lt $min_param_num || $real_num_params -gt $param_num ]]; then
        echo "Error: Invalid number of parameters. Expected between $min_param_num and $param_num parameters, given $real_num_params. Given params: " >&2
        for param in "${params[@]}"; do
            echo "$param" >&2
        done
        exit 1
    fi
}

get_yo_paths() {
    local pipeline_name="$1"
    local yo_ids="$2"

    # echo $yo_ids
    
    yo_paths=""
    # splits it by spaces
    for yo_id in $yo_ids; do
        yo_paths="$yo_paths ${pipeline_name}/overriding_yamls/${yo_id}.yaml"
    done
    echo $yo_paths
}

get_yo_str() {
    local yo_ids="$1"

    echo "yo_ids" >&2
    echo $yo_ids >&2
    
    yo_str=""
    # splits it by spaces
    for yo_id in $yo_ids; do
        yo_str="${yo_str}-${yo_id}"
    done
    echo ${yo_str}
}

get_out_name() {
    local base_name="$1"
    local protect_out_files="$2"

    if [[ $protect_out_files == "false" ]]; then
        echo "Maybe overwriting ${base_name}.txt." >&2
        echo "$base_name.txt"
        return
    fi

    counter=0
    out_name=${base_name}_${counter}.txt
    # -e for file exists, -L for symlink exists
    while [[ -e "$out_name" || -L "$out_name" ]]; do
        # Increment the counter and create a new filename
        ((counter++))
        out_name=${base_name}_${counter}.txt
    done

    echo "$out_name"
}