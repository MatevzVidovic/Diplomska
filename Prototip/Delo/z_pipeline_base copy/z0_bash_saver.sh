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

# source z_bash_saver.sh





output_basic_name="main"

results_folder_name="bash_results"

# Step 1: Create the "bash_results" directory if it doesn't exist
mkdir -p ${results_folder_name}

# Create /older and /curr directories inside bash_results if they don't exist
mkdir -p "${results_folder_name}/older"
mkdir -p "${results_folder_name}/curr"






# Step 2: Read and update the current bash index
prev_bash_ix_file="${results_folder_name}/prev_bash_ix.txt"
if [[ -f $prev_bash_ix_file ]]; then
    curr_bash_ix=$(<"$prev_bash_ix_file")
    curr_bash_ix=$((curr_bash_ix + 1))
else
    curr_bash_ix=0
fi
echo "$curr_bash_ix" > "$prev_bash_ix_file"





# Move contents from /curr to /older
# (the 2>/dev/null is to suppress the error message if the folder is empty)
mkdir -p "${results_folder_name}/older/moved_in_${curr_bash_ix}"
mv "${results_folder_name}/curr/"* "${results_folder_name}/older/" 2>/dev/null







# Step 3: Save the script content into a file
script_content_file="${results_folder_name}/curr/bash_code_${curr_bash_ix}.sh"
cat "$0" > "$script_content_file"

# Save the starting time of the script
start_time=$(date)
time_file="${results_folder_name}/curr/time_${curr_bash_ix}.txt"
echo "$start_time" > "$time_file"

# Step 4: Execute commands and save outputs
command_num=0
obn=${output_basic_name}
rfn=${results_folder_name}
cbi=${curr_bash_ix}
cn=${command_num}




# Writing to a file and a terminal:
# [command]    2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
# Explanation:
# 2>&1 - redirect stderr to stdout 
# | - pipe the output to the next command
# tee - read from stdin and write to stdout and files (stdin is here the output of the previous command)


# This didn't work so far:
# {
# We want another file, which will collect all errors from all the commands.
# So we can see all the errors that happened fast. To locate them after, we just ctrl+f them in the actual output files.
# We do this like so:

# Saving std error to a file while we are also doing the above saving of stdout and stderr:
# 2>> "${rfn}/curr/stderr_log_{cbi}.txt" 2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
# The first part redirects stderr to this file (>> appends, > overwrites)
# So somehow 2 goes both to the file and to 1. Don't ask me how exactly this works.
# }


