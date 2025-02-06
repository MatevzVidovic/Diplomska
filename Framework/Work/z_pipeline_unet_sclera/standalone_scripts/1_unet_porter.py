


import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open(f"{osp.join('pylog_configs', 'active_logging_config.txt')}", 'r') as f:
    cfg_name = f.read()
    yaml_path = osp.join('pylog_configs', cfg_name)

log_config_path = osp.join(yaml_path)
do_log = False
if osp.exists(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
        file_log_setting = config.get(osp.basename(__file__), False)
        if file_log_setting:
            do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024, var_blacklist=["tree_ix_2_module", "mask_path"])
handlers = py_log_always_on.file_handler_setup(MY_LOGGER)








# python3 -m z_pipeline_unet_sclera.standalone_scripts.v_unet_porter



import os
import os.path as osp

import shutil as sh

import y_helpers.json_handler as jh

import pickle

import sys



import y_helpers.yaml_handler as yh

yaml_path = osp.join("z_pipeline_unet_sclera", "standalone_scripts", "trial_1_sclera.yaml")
YD = yh.read_yaml(yaml_path)




# trial_folder = "trial_1"
# pruning_methods = ["IPAD_eq", "IPAD1"]
# origin_folder_name_constants = ("unet_prune_", "_vein")

# origin_names = [f"{origin_folder_name_constants[0]}{i}{origin_folder_name_constants[1]}" for i in pruning_methods]
# origin_paths = [osp.join(trial_folder, i) for i in origin_names]
# goal_containers_paths = [osp.join(trial_folder, "pruned", f"{i}_pruned") for i in origin_names]

# retained_percents = [0.75, 0.5, 0.25, 0.03]
# goal_train_iters = 110
# resource_name = "flops_num"
# model_name = "UNet"


core_num = YD["core_num"]
trial_folder = YD["trial_name"]
pruning_methods = YD["pruning_methods"]
origin_prefix = YD["origin_prefix"]
origin_suffix = YD["origin_suffix"]

retained_percents = YD["retained_percents"]
goal_train_iters = YD["mtti"]
resource_name = YD["resource_name"]
model_name = "UNet"

yaml_id = YD["yaml_id"]


origin_names = [f"{origin_prefix}{i}{origin_suffix}" for i in pruning_methods]
origin_paths = [osp.join(trial_folder, i) for i in origin_names]
goal_containers_paths = [osp.join(trial_folder, "pruned", f"{i}_pruned") for i in origin_names]













def closest_perc_after_pruning_moment(model_path, percents, resource_name="flops_num"):

    try:
        main_save_path = osp.join(model_path, "saved_main")
        saved_model_wrapper_path = osp.join(model_path, "saved_model_wrapper")


        pl_j_path = osp.join(main_save_path, "prev_pruning_logs_name.json")
        pl_name = jh.load(pl_j_path)[f"prev_pruning_logs"]

        pruning_logs_path = osp.join(main_save_path, "pruning_logs", pl_name)
        irc_path = osp.join(saved_model_wrapper_path, "initial_conv_resource_calc.pkl")

        pruning_logs = pickle.load(open(pruning_logs_path, "rb"))
        initial_resource_calc = pickle.load(open(irc_path, "rb"))

        initial_resources = initial_resource_calc.get_resource_of_whole_model(resource_name)

        pruning_moments = [i[0] for i in pruning_logs.pruning_logs]
        resource_calcs = [i[2] for i in pruning_logs.pruning_logs]
        resources = [i.get_resource_of_whole_model(resource_name) for i in resource_calcs]

        resources_percents = [(i) / initial_resources for i in resources]


        def find_closest_index(lst, target):
            """Find the index of the closest value to the target in the list."""
            min_ix = 0
            min_val = abs(lst[0] - target)
            for i in range(1, len(lst)):
                if abs(lst[i] - target) < min_val:
                    min_val = abs(lst[i] - target)
                    min_ix = i
            return min_ix


        ixs = [find_closest_index(resources_percents, i) for i in percents]
        closest_moments = [pruning_moments[ix] for ix in ixs]
        model_percents = [resources_percents[ix] for ix in ixs]

        return closest_moments, model_percents

    except Exception as e:
        py_log.log_stack(passed_logger=MY_LOGGER)
        raise e


# for model_path in model_paths:
#     closest_id_nums, model_percents = closest_perc_after_pruning_moment(model_path, retained_percents)
#     closest_ids = [f"{i}_after_pruning" for i in closest_id_nums]
#     print(closest_ids)
#     print(model_percents)


# sys.exit()


def copy_file(src_path, dest_path):
    # try:

    os.makedirs(osp.dirname(dest_path), exist_ok=True)
    sh.copy2(src_path, dest_path)
    # copy_file(src_path, dest_path)
    # except Exception as e:
    #     py_log.log_stack(passed_logger=MY_LOGGER)
    #     raise e


def move_model(origin_path, target_path, str_id, model_name):
    """
    origin path e.g. trial_1/unet_prune_IPAD_eq_vein
    target path e.g. trial_1/pruned/unet_prune_IPAD_eq_vein_pruned/unet_prune_IPAD_eq_vein_pruned_0.75

    str_id e.g.   5_after_pruning
    """

    try:

        os.makedirs(target_path, exist_ok=True)




        to_saved_main = osp.join(target_path, "saved_main")
        os.makedirs(to_saved_main, exist_ok=True)
        to_saved_model_wrapper = osp.join(target_path, "saved_model_wrapper")
        os.makedirs(to_saved_model_wrapper, exist_ok=True)




        # For curr_folder_name/saved_model_wrapper/:

        # From /SegNet_main/safety_copies/actual_safety_copies/:

        # - copy SegNet_{str_id}.pth
        model_pth = f"{model_name}_{str_id}.pth"
        to_model_pth = osp.join(origin_path, "safety_copies", "actual_safety_copies", model_pth)
        to_destination = osp.join(to_saved_model_wrapper, 'models', model_pth)
        copy_file(to_model_pth, to_destination)
        

        # From /SegNet_main/saved_model_wrapper/:    

        # - copy pruner_{str_id}.pkl
        pruner_pkl = f"pruner_{str_id}.pkl"
        to_pruner_pkl = osp.join(origin_path, "saved_model_wrapper", "pruners", pruner_pkl)

        if osp.exists(to_pruner_pkl):
            to_destination = osp.join(to_saved_model_wrapper, 'pruners', pruner_pkl)
            copy_file(to_pruner_pkl, to_destination)
            previous_model_details = {"previous_model_filename": model_pth, "previous_pruner_filename": pruner_pkl}
        else:
            print(f"Could not find {to_pruner_pkl}. Skipping copying of {pruner_pkl}. This is expected if the model was not pruned.")
            previous_model_details = {"previous_model_filename": model_pth, "previous_pruner_filename": None}
        
        # - make a new previous_model_details.json: {"previous_model_filename":"SegNet_{str_id}.pth","previous_pruner_filename":"pruner_{str_id}.pkl"}
        to_previous_model_details = osp.join(to_saved_model_wrapper, "previous_model_details.json")
        jh.dump(to_previous_model_details, previous_model_details)

        
        # - copy initial_conv_resource_calc.pkl
        initial_conv_resource_calc = "initial_conv_resource_calc.pkl"
        to_initial_conv_resource_calc = osp.join(origin_path, "saved_model_wrapper", initial_conv_resource_calc)
        to_destination = osp.join(to_saved_model_wrapper, initial_conv_resource_calc)
        copy_file(to_initial_conv_resource_calc, to_destination)








        # For curr_folder_name/saved_main/:

        # From /SegNet_main/saved_main/:
        # - copy training_logs_{str_id}.pkl and pruning_logs_{str_id}.pkl
        # - copy initial_train_iters.json


        training_logs_pkl = f"training_logs_{str_id}.pkl"
        to_training_logs_pkl = osp.join(origin_path, "saved_main", "training_logs", training_logs_pkl)
        to_destination = osp.join(to_saved_main, 'training_logs', training_logs_pkl)
        copy_file(to_training_logs_pkl, to_destination)

        pruning_logs_pkl = f"pruning_logs_{str_id}.pkl"
        to_pruning_logs_pkl = osp.join(origin_path, "saved_main", "pruning_logs", pruning_logs_pkl)
        to_destination = osp.join(to_saved_main, 'pruning_logs', pruning_logs_pkl)
        copy_file(to_pruning_logs_pkl, to_destination)

        initial_train_iters = "initial_train_iters.json"
        to_initial_train_iters = osp.join(origin_path, "saved_main", initial_train_iters)
        to_destination = osp.join(to_saved_main, initial_train_iters)
        copy_file(to_initial_train_iters, to_destination)




        # From /SegNet_main/saved_main/copies/:
        # - copy prev_training_logs_name_{str_id}.json and rename it to prev_training_logs_name.json
        # - copy previous_pruning_logs_name_{str_id}.json and rename it to prev_pruning_logs_name.json


        prev_training_logs_name = f"prev_training_logs_name_{str_id}.json"
        to_prev_training_logs_name = osp.join(origin_path, "saved_main", "old_tl_jsons", prev_training_logs_name)
        to_destination = osp.join(to_saved_main, 'prev_training_logs_name.json')
        copy_file(to_prev_training_logs_name, to_destination)


        previous_pruning_logs_name = f"previous_pruning_logs_name_{str_id}.json"
        to_previous_pruning_logs_name = osp.join(origin_path, "saved_main", "old_pl_jsons", previous_pruning_logs_name)
        to_destination = osp.join(to_saved_main, 'prev_pruning_logs_name.json')
        copy_file(to_previous_pruning_logs_name, to_destination)


    except Exception as e:
        py_log.log_stack(passed_logger=MY_LOGGER)
        raise e











to_run_folder = osp.join(trial_folder, "run_files_2")
os.makedirs(to_run_folder, exist_ok=True)

to_out_folder = osp.join(trial_folder, "out_files_2")
os.makedirs(to_out_folder, exist_ok=True)


for ix in range(len(origin_paths)):

    after_pruning_train_iters, model_percents = closest_perc_after_pruning_moment(origin_paths[ix], retained_percents, resource_name)
    ids = [f"{i}_after_pruning" for i in after_pruning_train_iters]
    print(origin_paths[ix])
    print(ids)
    print(model_percents)


    target_folder_names = [f"{origin_names[ix]}_pruned_{i}" for i in retained_percents]
    target_paths = [osp.join(goal_containers_paths[ix], name) for name in target_folder_names]
    for i in range(len(ids)):
        move_model(origin_paths[ix], target_paths[i], ids[i], model_name)


        out_name = f"x_{target_folder_names[i]}_out.txt"
        to_out = osp.join(to_out_folder, out_name)

        sbatch_name = f"{target_folder_names[i]}.sbatch"
        sbatch = f"""#!/bin/bash
#SBATCH --job-name={target_paths[i]}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G

python3 unet_main.py --mtti {goal_train_iters}  --sd {target_paths[i]} --yaml z_pipeline_unet_sclera/{yaml_id}.yaml >> {to_out}  2>&1
"""
        


        ana_sbatch_name = f"ana_{target_folder_names[i]}.sbatch"
        ana_sbatch = f"""#!/bin/bash
#SBATCH --job-name={target_paths[i]}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100_80GB
#SBATCH --mem-per-cpu=8G

python3 unet_main.py --mtti {goal_train_iters}  --sd {target_paths[i]} --yaml z_pipeline_unet_sclera/{yaml_id}.yaml >> {to_out}  2>&1
"""

        # To escape { and } in fstrings, use double brackets: {{ and }}

        to_sbatch = osp.join(to_run_folder, sbatch_name)
        with open(to_sbatch, "w") as f:
            f.write(sbatch)

        to_ana_sbatch = osp.join(to_run_folder, ana_sbatch_name)
        with open(to_ana_sbatch, "w") as f:
            f.write(ana_sbatch)














# just have to_run_folder be set up correctly, and this whould go well:



# now make the automatic runnning mechanism
# We will have 2 sbatches that will run the python file.
# One will run normally, and one will run it on ana.
# Both sbatches have the max time limit of 2 days.

# The sbatch files take the arguments for the python file.

# Python file will be the runner.
# It has a yaml file in its run folder (creates it if it doesn't exist).
# Lists all the runner files in the run folder, with dicts: {"run": False, "finished": False}
# Before starting a run, it reads the run folder, figures out which in its sequence hasn't been run yet,
# writes to the yaml that thet file "run" is true, and runs the file.
# Upon finish it writes to the yaml that the file "finished" is true.

# The python file takes 2 args: 
# pos: start, mid_up, mid_down, end     # where in the sorted listdir it starts seeking to run files.
# max_run: int                          # how many files to run at most. Otherwise we are sure to surpass the time-limit at some point.

to_py_runner = osp.join(to_run_folder, "runner.py")
to_origin_runner = osp.join(osp.dirname(__file__), "runner.py")
sh.copy2(to_origin_runner, to_py_runner)

sbatch_name = "run_sbatch.sbatch"
sbatch = f"""#!/bin/bash

#SBATCH --job-name=runner
#SBATCH --time=2-00:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G

max_run=$1

python3 {to_py_runner} --max_run $max_run
"""

ana_sbatch_name = "ana_run_sbatch.sbatch"
sbatch = f"""#!/bin/bash

#SBATCH --job-name=runner
#SBATCH --time=2-00:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100_80GB
#SBATCH --mem-per-cpu=6G

max_run=$1

python3 {to_py_runner} --max_run $max_run
"""


to_sbatch = osp.join(to_run_folder, sbatch_name)
with open(to_sbatch, "w") as f:
    f.write(sbatch)

to_ana_sbatch = osp.join(to_run_folder, ana_sbatch_name)
with open(to_ana_sbatch, "w") as f:
    f.write(ana_sbatch)

print(f"{to_ana_sbatch}")





sys.exit()

