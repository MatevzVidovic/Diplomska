



import os
import stat
import os.path as osp
import yaml
import subprocess
import io
# import shutil as sh
# import argparse
# import time
# import fcntl
# import sys

from pathlib import Path




def get_fresh_folder(path_to_parent_folder):
        
    counter=999
    
    folder_path = Path(path_to_parent_folder) / f"{counter}"
    while osp.exists(folder_path):
        counter -= 1
        folder_path = Path(path_to_parent_folder) / f"{counter}"

    os.makedirs(folder_path, exist_ok=True)
    return Path(folder_path)


def get_yaml(path):
    if osp.exists(path):
        with open(path, 'r') as f:
            YD = yaml.safe_load(f)

            # when yaml is empty
            if YD is None:
                YD = {}
    else:
        YD = {}
    return YD


def write_yaml(inp_dict, file_path, sort_keys=False, indent=4):
    os.makedirs(Path(file_path).parent, exist_ok=True)
    with open(file_path, 'w') as file:
        yaml.dump(inp_dict, file, sort_keys=sort_keys, indent=indent)





# ---------- Helper functions for gradually building the yaml dict ----------

# Updating a yaml with a new yaml
def recursive_update(initial_yaml_dict, new_yaml_dict):
    # For values that are dictionaries, we recursively update them
    # For values that are not dictionaries, we simply overwrite them
    # We need this recursion to make the yaml tree structure useful.

    initial_yaml_dict_copy = initial_yaml_dict.copy()

    for key, value in new_yaml_dict.items():
        if key in initial_yaml_dict_copy and isinstance(value, dict) and isinstance(initial_yaml_dict_copy[key], dict):
            initial_yaml_dict_copy[key] = recursive_update(initial_yaml_dict_copy[key], value)
        else:
            initial_yaml_dict_copy[key] = value

    return initial_yaml_dict_copy


# Used when an additional yaml path is specified. (for the "yamls" key in our dicts)
# We need this so we can warn if the file doesn't exist. And to make adding easier.
def added_yaml(initial_yaml_dict, yaml_path, dirpath):

    initial_yaml_dict_copy = initial_yaml_dict.copy()

    relative_yaml_path = dirpath / yaml_path # relative to dirpath
    absolute_yaml_path = Path(yaml_path) # path from root as a second option

    if osp.exists(relative_yaml_path):
        added_yaml = get_yaml(relative_yaml_path)
        initial_yaml_dict_copy = recursive_update(initial_yaml_dict_copy, added_yaml)
    elif osp.exists(absolute_yaml_path):
        added_yaml = get_yaml(absolute_yaml_path)
        initial_yaml_dict_copy = recursive_update(initial_yaml_dict_copy, added_yaml)
    else:
        print(f"Warning: {yaml_path} does not exist. Skipping.")
    
    
    return initial_yaml_dict_copy


def my_update(initial_yaml_dict, new_yaml_dict, dirpath, retain_yamls=False):

    initial_yaml_dict_copy = initial_yaml_dict.copy()

    # I have no idea why we would need to ever retain yamls.
    # Seems to me that it's only keeping the added yamls in the new_ymal_dict, 
    # and so the key also gets added to the initial_yaml_dict_copy, 
    # and so they might be added again later.
    # But I'm leaving it for now.
    if retain_yamls:
        additional_yamls = new_yaml_dict.get("yamls", None)
    else:
        additional_yamls = new_yaml_dict.pop("yamls", None)
    
    initial_yaml_dict_copy = recursive_update(initial_yaml_dict_copy, new_yaml_dict)


    if additional_yamls is not None:
    
        for yaml_path in additional_yamls:
            initial_yaml_dict_copy = added_yaml(initial_yaml_dict_copy, yaml_path, dirpath)

        # added for logging/observability purposes. YD gets saved in the end, and it's nice to know which yamls were added along the way.
        if "previously_added_yamls" in initial_yaml_dict_copy:
            initial_yaml_dict_copy["previously_added_yamls"].extend(additional_yamls)
        else:
            initial_yaml_dict_copy["previously_added_yamls"] = additional_yamls.copy()
    

    
    return initial_yaml_dict_copy
    


# Comparing with template yaml, to make sure we have all the keys
def recursive_check(yaml_dict, template_yaml_dict):

    missing_keys = []
    
    # This was the initial proof of concept for the top level.
    # for key in test_yaml.keys():
    #     if key not in YD:
    #         missing_keys.append(key)
    
    for key, value in template_yaml_dict.items():
        if key not in yaml_dict:
            missing_keys.append(key)
        elif isinstance(template_yaml_dict[key], dict):
            if isinstance(value, dict):
                deep_missing_keys = recursive_check(yaml_dict[key], template_yaml_dict[key])
                new_missing_keys = [f"{key}:{deep_key}" for deep_key in deep_missing_keys]
                missing_keys.extend(new_missing_keys)
            else:
                missing_keys.append(f"{key}-not-a-dict")
    
    return missing_keys









# ---------- Helper functions for actual running ----------

def make_executable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)

def run(args, stdout_path=None, stderr_path=None, stdin=None, shell=False):
    # is shel=True, args is a string and not a list

    # Open the specified files or default to subprocess.PIPE
    stdin_target = stdin if isinstance(stdin, io.IOBase) else subprocess.PIPE   # is stdin if stdin is any kind of file (text or buffered)
    input = stdin if isinstance(stdin, str) else None
    stdout_target = open(stdout_path, 'w') if stdout_path else subprocess.PIPE
    stderr_target = open(stderr_path, 'w') if stderr_path else subprocess.PIPE

    # you run scripts in a few ways:
    # sourcing: ['source', 'script.sh'] (but script.sh needs execute permission, so use make_executable beforehand)
    # bash:   ['bash', 'script.sh']
    # python: ['python', 'script.py']
    # sbatch: ['sbatch', 'script.sbatch']
    # srun:   ['srun', 'python', 'script.py']

    process = subprocess.Popen(args,
                            stdin=stdin_target,
                            stdout=stdout_target,
                            stderr=stderr_target,
                            text=True, shell=shell)
    
    # !!! Stdout and stderr will be strs only if you are not writing to file (you are using subprocess.PIPE)
    # !!! If you are writing to file, stdout and stderr will be file objects.
    stdout, stderr = process.communicate(input=input) # also waits for process to terminate
    exit_code = process.returncode # available after the process terminates
    
    return stdout, stderr, exit_code


def run_no_wait(args, stdout_path, stderr_path, stdin=None, shell=False):
    # is shel=True, args is a string and not a list

    # you run scripts in a few ways:
    # sourcing: ['source', 'script.sh'] (but script.sh needs execute permission, so use make_executable beforehand)
    # bash:   ['bash', 'script.sh']
    # python: ['python', 'script.py']
    # sbatch: ['sbatch', 'script.sbatch']
    # srun:   ['srun', 'python', 'script.py']

    
    stdout_target = open(stdout_path, 'w')
    stderr_target = open(stderr_path, 'w')

    process = subprocess.Popen(args,
                            stdin=stdin,
                            stdout=stdout_target,
                            stderr=stderr_target,
                            text=True, shell=shell)

    # use process.poll() to check if it's finished or process.wait() to wait for its completion at a later point
    return process