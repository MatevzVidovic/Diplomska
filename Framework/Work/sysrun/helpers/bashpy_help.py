



import os
import os.path as osp
import shutil as sh
import sys
# import stat
# import yaml
# import argparse
# import time
# import subprocess
# import fcntl

from sysrun.helpers.help import get_yaml, write_yaml, get_fresh_folder


from pathlib import Path





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
    YD2 = YD["added_auto_main_args"] if "added_auto_main_args" in YD else {}
    YD3 = YD["main_yaml"]


    yamls_str = ""
    for path in YD.get("yamls", []):
        yamls_str = f"{yamls_str}_{Path(path).stem}" # get file name + remove suffix

    sd_path = Path("active_models") / f"{YD1['model']}_{YD1['sbatch_id']}_{yamls_str}__{module_path_of_bashpy_runfile}"
    if not YD1['retain_savedir']:
        print(f"Maybe overwriting {sd_path}.", file=sys.stderr)
        create_empty_folder(sd_path)


        
    base_output_path = Path("zzz_outputs") / sd_path
    output_folder_path = get_fresh_folder(base_output_path) # we get zzz_outputs/<sd_path>/999 for example
    outs_path = output_folder_path / "outs"
    os.makedirs(outs_path, exist_ok=True)
    errs_path = output_folder_path / "errs"
    os.makedirs(errs_path, exist_ok=True)

    # we make the temp main yaml
    main_yaml_path = Path("sysrun") / "bashpy_temp" / "temp.yaml"
    write_yaml(YD3, main_yaml_path)
    

    # Making a symlink to the latest output folder
    out_folder_path_absolute = output_folder_path.resolve()
    symlink_to_latest = out_folder_path_absolute.parent / "0_symlink_to_latest"
    # Uses Path object fns
    if symlink_to_latest.exists() or symlink_to_latest.is_symlink():
        symlink_to_latest.unlink()
    symlink_to_latest.symlink_to(out_folder_path_absolute, target_is_directory=True)

    
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

