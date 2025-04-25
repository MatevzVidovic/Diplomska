



import os
import stat
import os.path as osp
import shutil as sh
import yaml
import argparse
import time
import subprocess
import fcntl
import sys

from sysrun.helpers.help import get_yaml, write_yaml, get_fresh_folder


from pathlib import Path

import tempfile

# with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#     print(f'Temporary file created: {temp_file.name}')
#     temp_file.write(b'Temporary data.')
# # The file remains after closing because delete=False
# os.remove(temp_file.name)  # Clean up the file if needed


temp_file_strs = {
    "save_and_stop": "s\nstop\n",
    "results_and_stop": "r\nstop\n",
    "graph_and_stop": "g\n\nstop\n",
    "resource_graph_and_stop": "resource_graph\nstop\n",
    "test_showcase": "ts\nall\nstop\nstop\n",
    "data_aug": "da\n\n\n\n\n1\n\n\n\n\n2\n\n\n\n\nstop\nstop\n",
    "save_preds": "sp\nstop\nstop\n",
    "batch_size_train": "bst\nstop\nstop\n",
    "batch_size_eval": "bse\nstop\nstop\n",
    "flops_and_weights": "fw\nstop\nstop\n",
}



# Function to delete a folder if it exists and then create it empty
def create_empty_folder(folder_name):

    if osp.exists(folder_name):
        sh.rmtree(folder_name)

    os.makedirs(folder_name, exist_ok=True)


def create_empty_file(file_name):


    if osp.exists(file_name):
        sh.rmtree(file_name)
    
    os.makedirs(Path(file_name).parent, exist_ok=True)
    
    with open(file_name, 'w') as file:
        pass





# Check if number of arguments is within acceptable range
def check_param_num(param_num, num_optional, args_list):
    
    
    min_param_num = param_num - num_optional

    if len(args_list) < min_param_num or len(args_list) > param_num:
        print(f"Error: Invalid number of parameters. Expected between {min_param_num} and {param_num} parameters, given {len(args_list)}. Given params: ", file=sys.stderr)
        for param in args_list:
            print(param, file=sys.stderr)
        sys.exit(1)


def get_yo_paths(pipeline_name, yo_ids):
    
    yo_paths=""
    # splits it by spaces
    for yo_id in yo_ids:
        yo_paths = f"{yo_paths} {pipeline_name}/overriding_yamls/{yo_id}.yaml"
    
    # print(f"yo_paths: {yo_paths}")
    return yo_paths


def get_yo_str(yo_ids):

    print(f"yo_ids: {yo_ids}", file=sys.stderr)
    
    yo_str=""
    # splits it by spaces
    for yo_id in yo_ids:
        yo_str = f"{yo_str}-{yo_id}"
    return yo_str








def main(yaml_path, module_path_of_bashpy_runfile):

    YD = get_yaml(yaml_path)["oth"]
    YD1 = YD["bashpy_args"]
    YD2 = YD["added_auto_main_args"]
    YD3 = YD["main_yaml"]


    yamls_str = ""
    for path in YD.get("yamls", []):
        yamls_str = f"{yamls_str}_{Path(path).stem}" # get file name + remove suffix

    sd_path = Path("active_models") / f"{YD1['model']}_{YD1['sbatch_id']}_{yamls_str}__{module_path_of_bashpy_runfile}"
    if not YD1['retain_savedir']:
        print(f"Maybe overwriting {sd_path}.", file=sys.stderr)
        create_empty_folder(sd_path)


        

    base_output_path = Path("zzz_outputs") / f"x_{sd_path}"
    output_folder_path = get_fresh_folder(base_output_path) # we get zzz_outputs/x_<sd_name>/999 for example

    # we make the temp main yaml
    main_yaml_path = Path("sysrun") / "bashpy_temp" / "temp.yaml"
    write_yaml(YD3, main_yaml_path)
    
    return main_yaml_path, output_folder_path, sd_path







# There is no need for this now, because it all goes through the main yaml.

# def main(pipeline_name, param_num, num_optional, sbatch_id, protect_out_files, retain_savedir):


#     check_param_num(param_num, num_optional, args_list)
#     yo_paths = get_yo_paths(pipeline_name, yo_ids_list)
#     yo_str = get_yo_str(yo_ids_list)

#     print(f"yo_paths: {yo_paths}")

#     sd_name = f"{model}_{sbatch_id}_{yaml_id}{yo_str}"
#     if not retain_savedir:
#         print(f"Maybe overwriting {sd_name}.", file=sys.stderr)
#         create_empty_folder(sd_name)
        


#     base_name = f"x_{sd_name}"
#     out_name = get_out_name(base_name, protect_out_files)
#     create_empty_file(out_name)

