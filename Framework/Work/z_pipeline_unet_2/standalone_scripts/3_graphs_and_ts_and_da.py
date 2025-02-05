

import os
import os.path as osp


import y_helpers.yaml_handler as yh

import sys


yaml_path = osp.join("z_pipeline_unet_2", "standalone_scripts", "trial_1.yaml")
YD = yh.read_yaml(yaml_path)


pipeline_name = YD["pipeline_name"]
yaml_id = YD["yaml_id"]


# srun --pty -p dev -c 7 --gpus=A100 python3 v_graphs_and_ts_and_da.py

# or run me through multipurpose.sbatch




pruning_methods = YD["pruning_methods"]
retained_percents = YD["retained_percents"]

base_path = YD["trial_name"]
yaml_id = YD["yaml_id"]
origin_prefix = YD["origin_prefix"]
origin_suffix = YD["origin_suffix"]

folder_structure = {}

folder_structure[osp.join(base_path)] = [ (f"unet_full_train_{yaml_id}", "100%") ]


for pm in pruning_methods:

    method_folder_name = f"{origin_prefix}{pm}{origin_suffix}_pruned"
    curr_path = osp.join(base_path, "pruned", method_folder_name)

    folder_structure[curr_path] = []
    for rp in retained_percents:
        rp_perc = int(rp * 100)
        folder_structure[curr_path].append( (f"{method_folder_name}_{rp}", f"{pm}_{rp_perc}%") )











get_graphs_sh = osp.join(pipeline_name, "standalone_scripts", "z_get_graphs.sbatch")
get_da_sh = osp.join(pipeline_name, "standalone_scripts", "z_get_da.sbatch")
get_ts_sh = osp.join(pipeline_name, "standalone_scripts", "z_get_ts.sbatch")



count = 0
test_limit = 1
for base_f, main_fs in folder_structure.items():

    for main_f, _ in main_fs:
        sd = osp.join(base_f, main_f)
        os.system(f"bash {get_graphs_sh} {yaml_id} {sd}")
        os.system(f"bash {get_ts_sh} {yaml_id} {sd}")
        os.system(f"bash {get_da_sh} {yaml_id} {sd}")

        count += 1
        if count >= test_limit:
            sys.exit(0)


        # Fixing the mistake of not coppying stuff okay:

        # init_res_calc = osp.join(sd, "saved_model_wrapper", "initial_conv_resource_calc.pkl")
        # os.system(f"rm {init_res_calc}")

        # real_res_calc = osp.join("unet_prune_IPAD", "saved_model_wrapper", "initial_conv_resource_calc.pkl")
        # os.system(f"cp {real_res_calc} {init_res_calc}")