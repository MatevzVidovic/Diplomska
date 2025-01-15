#!/bin/bash


# Creates temp files that can be used as input. 
# E.g.:
# python3 ${main_file} < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

save_and_stop=$(mktemp)
printf "s\nstop\n" > "$save_and_stop"

results_and_stop=$(mktemp)
printf "r\nstop\n" > "$results_and_stop"

graph_and_stop=$(mktemp)
printf "g\nstop\n" > "$graph_and_stop"

resource_graph_and_stop=$(mktemp)
printf "resource_graph\nstop\n" > "$resource_graph_and_stop"

test_showcase=$(mktemp)
printf "ts\nall\nstop\nstop\n" > "$test_showcase"

data_aug=$(mktemp)
printf "da\n\n\n\n\n1\n\n\n\n\n2\n\n\n\n\nstop\nstop\n" > "$data_aug"





# Function to delete a folder if it exists and then create it empty
create_empty_folder() {
    local folder_name="$1"  # Access the first parameter

    if [ -d "$folder_name" ]; then
        rm -r "$folder_name"
    fi

    mkdir "$folder_name"
}