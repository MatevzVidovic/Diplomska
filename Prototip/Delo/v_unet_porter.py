

import argparse

import ast

import os
import os.path as osp

import helper_json_handler as jh



import sys





# python3 model_eval_graphs.py --msp ./scp_1/SegNet_random_main_60/saved_main --smp ./scp_1/SegNet_random_main_60/saved_model_wrapper
# [(61, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/SegNet_random_main_60 -m "[(61, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn SegNet --fid random_main --main segnet_main.py --gti 140

# python3 model_eval_graphs.py --msp ./scp_1/SegNet_uniform_main_60/saved_main --smp ./scp_1/SegNet_uniform_main_60/saved_model_wrapper
# [(61, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/SegNet_uniform_main_60 -m "[(61, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn SegNet --fid uniform_main --main segnet_main.py --gti 140

# python3 model_eval_graphs.py --msp ./scp_1/UNet_random_main_60/saved_main --smp ./scp_1/UNet_random_main_60/saved_model_wrapper
# [(63, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/UNet_random_main_60 -m "[(63, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn UNet --fid random_main --main unet_main.py --gti 140

# python3 model_eval_graphs.py --msp ./scp_1/UNet_uniform_main_60/saved_main --smp ./scp_1/UNet_uniform_main_60/saved_model_wrapper
# [(63, ""), (80, "after_pruning"), (100, "after_pruning") (116, "after_pruning")]
# python3 porter.py --sp scp_1/UNet_uniform_main_60 -m "[(63, \"\"), (80, \"after_pruning\"), (100, \"after_pruning\"), (116, \"after_pruning\")]" --mn UNet --fid uniform_main --main unet_main.py --gti 140



# srun --pty -p dev -c 7 --gpus=A100 python3 v_unet_porter.py --sp unet_prune_IPAD --afn unet_IPAD --gti 1200 -m "[(250, \"\"), (490, \"after_pruning\"), (740, \"after_pruning\"), (940, \"after_pruning\")]"

# srun --pty -p dev -c 7 --gpus=A100 python3 v_unet_porter.py --sp unet_prune_uniform --afn unet_uniform --gti 1200 -m "[(250, \"\"), (490, \"after_pruning\"), (740, \"after_pruning\"), (940, \"after_pruning\")]"

# srun --pty -p dev -c 7 --gpus=A100 python3 v_unet_porter.py --sp unet_prune_random --afn unet_random --gti 1200 -m "[(250, \"\"), (490, \"after_pruning\"), (740, \"after_pruning\"), (940, \"after_pruning\")]"




# Print all arguments, including the script name
print("All arguments:", sys.argv)


def models_handler(train_iter_arg):

    print("train_iter_arg:", train_iter_arg)
    m_list = ast.literal_eval(train_iter_arg)

    return m_list


argparser = argparse.ArgumentParser(description='Porter Stemmer')
argparser.add_argument('--sp', help='Starting path e.g. unet_prune_ipad', required=True, type=str)
argparser.add_argument('--afn', help='Abstract folder name. e.g. "unet_uniform"', required=True, type=str)
argparser.add_argument('-m', help='Models. [(train_iter, uniq_id),...] e.g. "[(1, \"ending_save\"), (80, \"\")]" ', required=True, type=models_handler)
argparser.add_argument('--gti', help='Goal train iters. e.g. 140', required=True, type=int)


args = argparser.parse_args()
starting_path = args.sp
models = args.m
abstract_folder_name = args.afn
goal_train_iters = args.gti

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The explanation in words is at the bottom of the document.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Use ctrl K 0  to collapse all strings in VS Code. Makes it easier to read.





# What do we have to do?

# base_folder = f"{abstract_folder_name}_PrunedModels"    e.g. UNet_IPAD_PrunedModels

# - make base folder

model_name = "UNet"
base_folder = f"ana_pruned_models_{abstract_folder_name}"    # e.g. UNet_IPAD_PrunedModels
os.makedirs(base_folder, exist_ok=True)









for model in models:



    # For each model:

    # str_id = f"{train_iter}_{uniq_id}"
    # curr_folder_name = f"{abstract_folder_name}_{str_id}"
    
    # - make folder:  base_folder/curr_folder_name,   
    #     e.g.     UNet_IPAD_PrunedModels/UNet_IPAD_80_after_pruning
    # - and subfolders saved_main and saved_model_wrapper

    train_iter, uniq_id = model
    str_id = f"{train_iter}_{uniq_id}"
    
    curr_folder_name = f"{abstract_folder_name}_{str_id}"
    to_curr_folder = osp.join(base_folder, curr_folder_name)
    os.makedirs(to_curr_folder, exist_ok=True)


    to_saved_main = osp.join(to_curr_folder, "saved_main")
    os.makedirs(to_saved_main, exist_ok=True)
    to_saved_model_wrapper = osp.join(to_curr_folder, "saved_model_wrapper")
    os.makedirs(to_saved_model_wrapper, exist_ok=True)




    # For curr_folder_name/saved_model_wrapper/:

    # From /SegNet_main/safety_copies/actual_safety_copies/:
    # - copy SegNet_{str_id}.pth

    model_pth = f"{model_name}_{str_id}.pth"
    to_model_pth = osp.join(starting_path, "safety_copies", "actual_safety_copies", model_pth)
    os.system(f"cp {to_model_pth} {to_saved_model_wrapper}")
    
    
    # From /SegNet_main/saved_model_wrapper/:    
    # - copy pruner_{str_id}.pkl

    pruner_pkl = f"pruner_{str_id}.pkl"
    to_pruner_pkl = osp.join(starting_path, "saved_model_wrapper", pruner_pkl)

    if osp.exists(to_pruner_pkl):
        
        os.system(f"cp {to_pruner_pkl} {to_saved_model_wrapper}")
        previous_model_details = {"previous_model_filename": model_pth, "previous_pruner_filename": pruner_pkl}


    else:
        print(f"Could not find {to_pruner_pkl}. Skipping copying of {pruner_pkl}. This is expected if the model was not pruned.")
        previous_model_details = {"previous_model_filename": model_pth, "previous_pruner_filename": None}
    
    # - make a new previous_model_details.json: {"previous_model_filename":"SegNet_{str_id}.pth","previous_pruner_filename":"pruner_{str_id}.pkl"}

    to_previous_model_details = osp.join(to_saved_model_wrapper, "previous_model_details.json")
    jh.dump(to_previous_model_details, previous_model_details)






    # For curr_folder_name/saved_main/:

    # From /SegNet_main/saved_main/:
    # - copy training_logs_{str_id}.pkl and pruning_logs_{str_id}.pkl
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




    # From /SegNet_main/saved_main/copies/:
    # - copy prev_training_logs_name_{str_id}.json and rename it to prev_training_logs_name.json
    # - copy previous_pruning_logs_name_{str_id}.json and rename it to prev_pruning_logs_name.json


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
    # But we would also like to automatically create the sbatch files to run the programes with the correct parameters.

    # We will create a run_folder {base_folder}/z_run_{abstract_folder_name}/
    #  with all the necessary sbatch scripts.

    run_folder = f"z_run_{abstract_folder_name}"
    to_run_folder = osp.join(base_folder, run_folder)
    os.makedirs(to_run_folder, exist_ok=True)


    # Then you will be running:
    # sbatch {base_folder}/z_run_{folder_name}/y_{folder_name}_{str_id}.sbatch






    # So how does the preparation of scripts work:
       
    # - create an sbatch script for each model:

    sbatch_name = f"y_{curr_folder_name}.sbatch"
    to_sbatch = osp.join(to_run_folder, sbatch_name)



    sbatch = f"""#!/bin/bash
#SBATCH --job-name=ut_{str_id}
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G


mti={goal_train_iters - train_iter}

srun --output=x_unet_train_{str_id}.txt python3 unet_original_main.py --mti ${{mti}}  --ptd ./vein_sclera_data --sd {to_curr_folder} --yaml z_pipeline_unet/unet_original_0.yaml
"""

    # To escape { and } in fstrings, use double brackets: {{ and }}

    
    with open(to_sbatch, "w") as f:
        f.write(sbatch)













    """
    





    What do we have to do?

    abstract_folder_name = f"{model_name}_{folder_id}"    e.g. UNet_IPAD
    base_folder = f"{abstract_folder_name}_PrunedModels"    e.g. UNet_IPAD_PrunedModels

    - make base folder

    For each model:

    str_id = f"{train_iter}_{uniq_id}"
    curr_folder_name = f"{abstract_folder_name}_{str_id}"
    
    - make folder:  base_folder/curr_folder_name,   
        e.g.     UNet_IPAD_PrunedModels/UNet_IPAD_80_after_pruning
    - and subfolders saved_main and saved_model_wrapper
        

    For curr_folder_name/saved_model_wrapper/:

    From /SegNet_main/safety_copies/actual_safety_copies/:
    - copy SegNet_{str_id}.pth

    From /SegNet_main/saved_model_wrapper/:    
    - copy pruner_{str_id}.pkl

    - make a new previous_model_details.json: {"previous_model_filename":"SegNet_{str_id}.pth","previous_pruner_filename":"pruner_{str_id}.pkl"}

    

    For curr_folder_name/saved_main/:

    From /SegNet_main/saved_main/:
    - copy training_logs_{str_id}.pkl and pruning_logs_{str_id}.pkl
    - copy initial_train_iters.json

    From /SegNet_main/saved_main/copies/:
    - copy prev_training_logs_name_{str_id}.json and rename it to prev_training_logs_name.json
    - copy previous_pruning_logs_name_{str_id}.json and rename it to prev_pruning_logs_name.json


    




    Now the model copying is done.
    But we would also like to automatically create the sbatch files to run the programes with the correct parameters.

    We will create a run_folder {base_folder}/z_run_{abstract_folder_name}/
     with all the necessary sbatch scripts.

    Then you will be running:
    sbatch {base_folder}/z_run_{folder_name}/y_{folder_name}_{str_id}.sbatch
    


    So how does the preparation of scripts work:
       
    - create an sbatch script for each model:

    filename: y_{folder_name}_{str_id}.sbatch


    #!/bin/bash

    #SBATCH --job-name=ut_{str_id}
    #SBATCH --time=0-12:00:00

    #SBATCH -p frida
    #SBATCH -c 7
    #SBATCH --gpus=A100_80GB
    #SBATCH --mem-per-cpu=8G


    mti={goal_train_iters - train_iter}

    srun --output=x_unet_train_{str_id}.txt python3 unet_original_main.py --mti ${{mti}}  --ptd ./vein_sclera_data --sd {curr_folder_name} --yaml z_pipeline_unet/unet_original_0.yaml


    """
