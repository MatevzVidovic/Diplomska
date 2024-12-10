

import argparse

import ast

import os
import os.path as osp

import json_handler as jh



import sys





# python3 model_eval_graphs.py --msp ./scp_1/SegNet_random_main_60/saved_main --smp ./scp_1/SegNet_random_main_60/saved_model_wrapper
# [(61, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/SegNet_random_main_60 -m "[(61, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn SegNet --fid random_main --main segnet_main.py --gti 140

# python3 model_eval_graphs.py --msp ./scp_1/SegNet_uniform_main_60/saved_main --smp ./scp_1/SegNet_uniform_main_60/saved_model_wrapper
# [(61, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/SegNet_uniform_main_60 -m "[(61, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn SegNet --fid uniform_main --main segnet_main.py --gti 140

# python3 model_eval_graphs.py --msp ./scp_1/UNet_random_main_60/saved_main --smp ./scp_1/UNet_random_main_60/saved_model_wrapper
# [(63, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/UNet_random_main_60 -m "[(63, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn UNet --fid random_main --main main.py --gti 140

# python3 model_eval_graphs.py --msp ./scp_1/UNet_uniform_main_60/saved_main --smp ./scp_1/UNet_uniform_main_60/saved_model_wrapper
# [(63, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/UNet_uniform_main_60 -m "[(63, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn UNet --fid uniform_main --main main.py --gti 140








# Print all arguments, including the script name
print("All arguments:", sys.argv)


def models_handler(train_iter_arg):

    print("train_iter_arg:", train_iter_arg)
    m_list = ast.literal_eval(train_iter_arg)

    return m_list


argparser = argparse.ArgumentParser(description='Porter Stemmer')
argparser.add_argument('--sp', help='Starting path e.g. scp_copy_0/SegNet_main', required=True, type=str)
argparser.add_argument('-m', help='Models. [(train_iter, uniq_id),...] e.g. "[(1, \"ending_save\"), (80, \"\")]" ', required=True, type=models_handler)
argparser.add_argument('--mn', help='model name. e.g. SegNet', required=True, type=str)
argparser.add_argument('--fid', help='Folder ID. e.g. "_uniform"', required=True, type=str)
argparser.add_argument('--main', help='main.py. e.g. segnet_main.py', required=True, type=str)
argparser.add_argument('--gti', help='Goal train iters. e.g. 140', required=True, type=int)


args = argparser.parse_args()
starting_path = args.sp
models = args.m
model_name = args.mn
folder_id = args.fid
main_py = args.main
goal_train_iters = args.gti

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The explanation in words is at the bottom of the document.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Use ctrl K 0  to collapse all strings in VS Code. Makes it easier to read.

folder_name = f"{model_name}_{folder_id}"
base_folder = f"{folder_name}_PrunedModels"
os.makedirs(base_folder, exist_ok=True)

pipeline_folder = f"z_pipeline_{folder_name}"
to_pipeline_folder = osp.join(base_folder, pipeline_folder)
os.makedirs(to_pipeline_folder, exist_ok=True)

# Create the base bash scripts
z0_bash_saver = """#!/bin/bash

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



# If 1 argument is given, it is the name of the pipeline

# Apparently, while the name of the script is $0, it is not counted in the number of arguments.

if [ $# -eq 1 ]; then
    pipeline_name=$1
    mkdir -p ${pipeline_name}
    results_folder_name="${pipeline_name}/${results_folder_name}"
fi






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
"""

to_z0_bash_saver = osp.join(to_pipeline_folder, "z0_bash_saver.sh")
with open(to_z0_bash_saver, "w") as f:
    f.write(z0_bash_saver)


z0_temp_inputs = """#!/bin/bash

# Creates temp files save_and_stop, results_and_stop, and graph_and_stop

save_and_stop=$(mktemp)
printf "s\nstop\n" > "$save_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


results_and_stop=$(mktemp)
printf "r\nstop\n" > "$results_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$results_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

graph_and_stop=$(mktemp)
printf "g\nstop\n" > "$graph_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$graph_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

resource_graph_and_stop=$(mktemp)
printf "resource_graph\nstop\n" > "$resource_graph_and_stop"
"""

to_z0_temp_inputs = osp.join(to_pipeline_folder, "z0_temp_inputs.sh")
with open(to_z0_temp_inputs, "w") as f:
    f.write(z0_temp_inputs)


threads = 48

model_params= """#!/bin/bash
batch_size=100
nodw={threads}
"""
to_model_params = osp.join(to_pipeline_folder, f"{folder_name}_params.sh")
with open(to_model_params, "w") as f:
    f.write(model_params)




for model in models:




    # - make folder {folder_name}_{train_iter}_{uniq_id}, and subfolders saved_main and saved_model_wrapper

    # (to be clear: {str_id} is {train_iter}_{uniq_id})


    train_iter, uniq_id = model
    str_id = f"{train_iter}_{uniq_id}"


    to_model_dir = osp.join(base_folder, f"{folder_name}_{str_id}")
    os.makedirs(to_model_dir, exist_ok=True)
    to_saved_main = osp.join(to_model_dir, "saved_main")
    os.makedirs(to_saved_main, exist_ok=True)
    to_saved_model_wrapper = osp.join(to_model_dir, "saved_model_wrapper")
    os.makedirs(to_saved_model_wrapper, exist_ok=True)



    # - copy scp_copy_0/SegNet_main/safety_copies/actual_safety_copies/SegNet_{train_iter}_{uniq_id}.pth
    #  to saved_model_wrapper

    model_pth = f"{model_name}_{str_id}.pth"
    to_model_pth = osp.join(starting_path, "safety_copies", "actual_safety_copies", model_pth)
    os.system(f"cp {to_model_pth} {to_saved_model_wrapper}")
    
    # - copy scp_copy_0/SegNet_main/saved_model_wrapper/pruner_{train_iter}_{uniq_id}.pkl to saved_model_wrapper

    pruner_pkl = f"pruner_{str_id}.pkl"
    to_pruner_pkl = osp.join(starting_path, "saved_model_wrapper", pruner_pkl)
    os.system(f"cp {to_pruner_pkl} {to_saved_model_wrapper}")

    # - make a new previous_model_details.json: {"previous_model_filename":"SegNet_{str_id}.pth","previous_pruner_filename":"pruner_{str_id}.pkl"}

    previous_model_details = {"previous_model_filename": model_pth, "previous_pruner_filename": pruner_pkl}
    to_previous_model_details = osp.join(to_saved_model_wrapper, "previous_model_details.json")
    jh.dump(to_previous_model_details, previous_model_details)






        
    # For {folder_name}_{str_id}/saved_main/:

    # From scp_copy_0/SegNet_main/saved_main/:
    # - copy training_logs_{str_id}.pkl and pruning_logs_{str_id}.pkl to saved_main
    # - copy initial_train_iters.json

    training_logs_pkl = f"training_logs_{str_id}.pkl"
    to_training_logs_pkl = osp.join(starting_path, "saved_main", training_logs_pkl)
    os.system(f"cp {to_training_logs_pkl} {to_saved_main}")

    pruning_logs_pkl = f"pruning_logs_{str_id}.pkl"
    to_pruning_logs_pkl = osp.join(starting_path, "saved_main", pruning_logs_pkl)
    os.system(f"cp {to_pruning_logs_pkl} {to_saved_main}")

    initial_train_iters = "initial_train_iters.json"
    to_initial_train_iters = osp.join(starting_path, "saved_main", initial_train_iters)
    os.system(f"cp {to_initial_train_iters} {to_saved_main}")



    # From scp_copy_0/SegNet_main/saved_main/copies/:
    # - copy prev_training_logs_name_{str_id}.json to saved_main and rename it to prev_training_logs_name.json
    # - copy previous_pruning_logs_name_{str_id}.json to saved_main and rename it to prev_pruning_logs_name.json

    prev_training_logs_name = f"prev_training_logs_name_{str_id}.json"
    to_prev_training_logs_name = osp.join(starting_path, "saved_main", "copies", prev_training_logs_name)
    os.system(f"cp {to_prev_training_logs_name} {to_saved_main}")
    # below renames the file
    os.system(f"mv {osp.join(to_saved_main, prev_training_logs_name)} {osp.join(to_saved_main, 'prev_training_logs_name.json')}")

    previous_pruning_logs_name = f"previous_pruning_logs_name_{str_id}.json"
    to_previous_pruning_logs_name = osp.join(starting_path, "saved_main", "copies", previous_pruning_logs_name)
    os.system(f"cp {to_previous_pruning_logs_name} {to_saved_main}")
    # below renames the file
    os.system(f"mv {osp.join(to_saved_main, previous_pruning_logs_name)} {osp.join(to_saved_main, 'prev_pruning_logs_name.json')}")







    # Now the model copying is done.
    # But we would also like to automatically create the bash pipeline.

    # To use the pipeline, move /z_pipeline_{folder_name}/ and all the model folders
    #   out of /{folder_name}_PrunedModels/ into the working directory.
    # Basically move it one directory up. At the start it is in /Delo/{folder_name}_PrunedModels/
    # then it should be in /Delo/.

    # Then you run:
    # sbatch z_pipeline_{folder_name}/y_{folder_name}_{str_id}_run_me.sbatch
    
        

    # For each model:

    # - create a bash script for each model:
    # (make it an fstring, so you can substitute mti into it)

    model_pipeline_sh_name = f"{folder_name}_{str_id}.sh"
    to_model_pipeline_sh = osp.join(to_pipeline_folder, model_pipeline_sh_name)

    # To escape { and } in fstrings, use double brackets: {{ and }}

    model_pipeline_sh = f"""#!/bin/bash

source {pipeline_folder}/z0_bash_saver.sh {pipeline_folder}

# Creates temp files save_and_stop, results_and_stop, and graph_and_stop
source {pipeline_folder}/z0_temp_inputs.sh

# Sets vars batch_size, nodw
source {pipeline_folder}/{folder_name}_params.sh

mti={goal_train_iters - train_iter}

python3 {main_py} --ips 999999 --bs ${{batch_size}} --nodw $nodw --sd {folder_name}_{str_id} --ptd ./vein_sclera_data --mti ${{mti}}          2>&1 | tee "${{rfn}}/curr/${{obn}}_${{cbi}}_${{cn}}.txt"; cn=$((cn + 1))
"""

    with open(to_model_pipeline_sh, "w") as f:
        f.write(model_pipeline_sh)
    
        
    
    # - create an sbatch script for each model:

    model_pipeline_sbatch_name = f"y_{folder_name}_{str_id}_run_me.sbatch"
    to_model_pipeline_sbatch = osp.join(to_pipeline_folder, model_pipeline_sbatch_name)

    model_pipeline_sbatch = f"""#!/bin/bash
#SBATCH --job-name=IPAD
#SBATCH --time=0-02:00:00

#SBATCH -p frida
#SBATCH -c {threads}
#SBATCH --gpus=A100
#SBATCH --output={folder_name}_{str_id}_out.txt

srun bash {model_pipeline_sh_name}
"""
    
    with open(to_model_pipeline_sbatch, "w") as f:
        f.write(model_pipeline_sbatch)


    """
    
    What do we have to do?

    - make folder {folder_name}_PrunedModels. Everything goes into it. It will be what we scp to the server.

    
    For {folder_name}_{str_id}/saved_model_wrapper/:

    - make folder {folder_name}_{train_iter}_{uniq_id}, and subfolders saved_main and saved_model_wrapper

    (to be clear: {str_id} is {train_iter}_{uniq_id})

    - copy scp_copy_0/SegNet_main/safety_copies/actual_safety_copies/SegNet_{train_iter}_{uniq_id}.pth
     to saved_model_wrapper
    
    - copy scp_copy_0/SegNet_main/saved_model_wrapper/pruner_{train_iter}_{uniq_id}.pkl to saved_model_wrapper

    - make a new previous_model_details.json: {"previous_model_filename":"SegNet_{str_id}.pth","previous_pruner_filename":"pruner_{str_id}.pkl"}

    
    For {folder_name}_{str_id}/saved_main/:

    From scp_copy_0/SegNet_main/saved_main/:
    - copy training_logs_{str_id}.pkl and pruning_logs_{str_id}.pkl to saved_main
    - copy initial_train_iters.json

    From scp_copy_0/SegNet_main/saved_main/copies/:
    - copy prev_training_logs_name_{str_id}.json to saved_main and rename it to prev_training_logs_name.json
    - copy previous_pruning_logs_name_{str_id}.json to saved_main and rename it to prev_pruning_logs_name.json


    




    Now the model copying is done.
    But we would also like to automatically create the bash pipeline.

    We will create a folder /z_pipeline_{folder_name}/
     with all the necessary bash and sbatch scripts for the pipeline.

    To use the pipeline, move /z_pipeline_{folder_name}/ and all the model folders
      out of /{folder_name}_PrunedModels/ into the working directory.
    Basically move it one directory up. At the start it is in /Delo/{folder_name}_PrunedModels/
    then it should be in /Delo/.

    Then you run:
    sbatch z_pipeline_{folder_name}/y_{folder_name}_{str_id}_run_me.sbatch
    


    So how does the preparation of scripts work:


    - Create a folder /z_pipeline_{folder_name}/.
    
    - create 3 base bash scripts:
        - z0_bash_saver.sh     # same as so far 
        - z0_temp_inputs.sh     # same as so far

        - {folder_name}_params.sh   
        # just batch_size=100, nodw={threads}.  This is created here so it can later be changed easily for the whole pipeline.
    
        

    For each model:

    - create a bash script for each model:
    (make it an fstring, so you can substitute mti into it)

    filename: {folder_name}_{str_id}.sh:
    
    #!/bin/bash

    source z0_bash_saver.sh

    # Creates temp files save_and_stop, results_and_stop, and graph_and_stop
    source z0_temp_inputs.sh

    # Sets vars batch_size, nodw
    source {folder_name}_params.sh

    mti={goal_train_iters - train_iter}

    python3 {main} --ips 999999 --bs $\{batch_size\} --nodw $nodw --sd {folder_name}_{str_id} --ptd ./vein_sclera_data --mti $\{mti\}          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

    
    
    - create an sbatch script for each model:

    filename: y_{folder_name}_{str_id}_run_me.sbatch

    #!/bin/bash
    #SBATCH --job-name=IPAD
    #SBATCH --time=0-02:00:00

    #SBATCH -p frida
    #SBATCH -c {threads}
    #SBATCH --gpus=A100
    #SBATCH --output=x0_out.txt

    srun bash z_pipeline_{folder_name}/z_combining_script.sh

    """
