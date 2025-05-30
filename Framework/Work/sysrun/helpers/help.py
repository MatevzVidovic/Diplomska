



import os
import stat
import os.path as osp
import yaml
import subprocess
import io
import sys
# import shutil as sh
# import argparse
# import time
# import fcntl

from pathlib import Path


# ---------- General helper functions ----------


def symlink_safe_preresolve(path: Path, prevent_starting_double_dot=True):
    """
    In a path like:
    zzzzz_runners_simple/sclera/unet/test/../model_yamls/sclera_fake.yaml

    where test/ is actually a symlink to zzzzz_runners_simple/0_shared/test/

    what actualy happens when pathlib Path resolves it, is:
    - first resolving up to the relative element .. :
    zzzzz_runners_simple/sclera/unet/test/    so we end up in zzzzz_runners_simple/0_shared/test/
    - then resolving the relative element .. :
    so we end up in zzzzz_runners_simple/0_shared/
    - then resolving the rest:
    model_yamls/sclera_fake.yaml
    So we would end up in zzzzz_runners_simple/0_shared/model_yamls/sclera_fake.yaml

    But there is no model_yamls/ in 0_shared. It's in sclera/unet/

    So to prevent stuff like this from happening, we manually check for .. in the path and
    remove the parent folder in the path, so we resolve correctly.

    !!!!!
    ALSO
    We prevent going past the root of our project.
    For example, using a path like:
    ../model_yamls/sclera_fake.yaml
    Would, in sysrun.py, try being tboth the absolute path (from root of project) and the relative path (from where the runner.py file resides).
    In the absolute case, it would try to go to the parent folder of the root of the project, which is not desired.

    """

    new_path = []
    for folder in path.parts:
        if folder == "..":
            if new_path: # if nonempty
                new_path.pop()
            elif not prevent_starting_double_dot:
                new_path.append(folder)
        else:
            new_path.append(folder)

    return Path(*new_path)




    pass




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








# ---------- Helper functions for getting fresh folders ----------

def get_fresh_folder(path_to_parent_folder, do_cleanup=True):
    # we will only keep 4 of the latest folders in path_to_parent_folder
    # Older ones should be moved to an old/ folder.
    # This makes the file system cleaner and easier to navigate,
    # especially in vscode.

    counter=999
    
    folder_path = Path(path_to_parent_folder) / f"{counter}"
    folder_path_in_old = Path(path_to_parent_folder) / "old" / f"{counter}"
    while osp.exists(folder_path) or osp.exists(folder_path_in_old):
        counter -= 1
        folder_path = Path(path_to_parent_folder) / f"{counter}"
        folder_path_in_old = Path(path_to_parent_folder) / "old" / f"{counter}"

    if do_cleanup:
        cleanup_by_moving_to_old(path_to_parent_folder, keep_latest=3) 
    # we do the above first, to make absolutely sure the new folder is fresh and not in the old/ folder.
    os.makedirs(folder_path, exist_ok=True)
    return Path(folder_path)

def cleanup_by_moving_to_old(path_to_parent_folder, keep_latest=4):
    # Moves all folders in path_to_parent_folder to old/ folder, except the latest keep_latest ones.
    # If old/ folder doesn't exist, it will be created.
    
    parent_path = Path(path_to_parent_folder)
    old_path = parent_path / "old"
    os.makedirs(old_path, exist_ok=True)

    folders = sorted([f for f in parent_path.iterdir() if f.is_dir() and f.name.isdigit()], reverse=False)

    for folder in folders[keep_latest:]:
        new_folder_path = old_path / folder.name
        if not new_folder_path.exists():
            folder.rename(new_folder_path)

def get_fresh_folder_basic(path_to_parent_folder):
        
    counter=999
    
    folder_path = Path(path_to_parent_folder) / f"{counter}"
    while osp.exists(folder_path):
        counter -= 1
        folder_path = Path(path_to_parent_folder) / f"{counter}"

    os.makedirs(folder_path, exist_ok=True)
    return Path(folder_path)







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

def write_sbatch_file(sbatch_dict, sbatch_path, python_command, no_sys_args=False):
    
    if no_sys_args:
        sys_args_string = ""
    else:
        sys_args = sbatch_dict.get("sys", None)
        sys_args_string = ""
        for key, value in sys_args.items():
            if key.startswith("--"):
                # To make: #SBATCH --gpus=A100
                sys_args_string += f"#SBATCH {key}{value}\n"
            else:
                # To make: #SBATCH -c 16
                # We need the extra space with single - args.
                sys_args_string += f"#SBATCH {key} {value}\n"


    sbatch_file = f"""#!/bin/bash

{sys_args_string}

{python_command}
"""
    
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_file)



def make_executable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)

def run(args, stdout_path=None, stderr_path=None, stdin=None, shell=False, terminal_inp=False):
    
    """
    if shel=True, args is a string and not a list. If a list is given, we will join it into a string.
    
    Since we run python -m sysrun.sysrun in the terminal, subprocess.PIPE is for the python subprocess.
    So we can't use terminal input to steer our program.
    So we need to bypass this, by making stdin_target sys.stdin directly - if terminal_inp is True.
    The output of the program, however, is still read from the file - you won't see it in the terminal.
    """

    # --------- Open the specified files or default to subprocess.PIPE --------- 

    # what happens is:
    # sysrun proc -> run() -> bash (script) process -> py process -> run()
    # By default, the bash stdin gets passed to the python process, because that's how bash works.
    # We give the run() in sysrun sys.stdin explicitly so it always reads the terminal.
    # But sys.stdin for the pyprocess is, I think, the bash process stdin.
    # Even sys.__stdin__ (which gives the "original" stdin) is the bash process stdin.
    
    stdin_target = None
    if terminal_inp:
        stdin_target = sys.__stdin__
    else:
        stdin_target = stdin if isinstance(stdin, io.IOBase) else subprocess.PIPE

    # A long experimentation on why terminal input could not be passed to the program:
    """
    # stdin_target = sys.__stdin__

    # this works:
    # (i think this worked because stdin_target stayed None in sysrun.py, 
    # which surprisingly actually ended up being the exact same as sys.__stdin__ )
    # stdin_target = None
    # if terminal_inp:
    #     stdin_target = sys.__stdin__
    
    # this doesn't work:
    # stdin_target = None
    # if terminal_inp:
    #     stdin_target = sys.__stdin__
    # else:
    #     stdin_target = stdin if isinstance(stdin, io.IOBase) else subprocess.PIPE

    # print(f"stdin_target is {stdin_target} (of type {type(stdin_target)})")
    # print(f"terminal_inp is {terminal_inp}")

    # whyyyyyyyyyyyyyyyyy!!!!!!

    # stdin_target = stdin if isinstance(stdin, io.IOBase) else sys.__stdin__ if terminal_inp else subprocess.PIPE

    # Now I know why this didn't work.
    # I didn't set stdin=sys.stdin in sysrun.py, so it was None.
    # Our system looks like this:
    # sysrun proc -> run() -> bash (script) process -> py process -> run()
    # So changing the code actually changed it in both sysrun.py and the script.py that is run.
    # And in sysrun it had terminal_inp=False, so stdin was None.
    # SO if we set stdin_target to sys.stdin without any condition, it got changed there too, and that's when it all worked.
    # But
    # Why didn't:
    # if terminal_inp:
    #     stdin_target = sys.__stdin__
    # work?
    # It sets it to the original stdin, right?
    # Well, I think it actually set it to the bash stdin.
    # And so things went like this:
    # sysrun (terminal) stdin -> nowhere.   nothing -> bash_stdin --(this arrow i think happens by default when you run a bash script)--> pyproc_stdin  
    # But now:
    # sysrun (terminal) stdin -> bash_stdin --(this arrow i think happens by default when you run a bash script)--> pyproc_stdin

    # This works:
    # stdin_target = sys.stdin

    # And this works:
    # stdin_target = stdin if isinstance(stdin, io.IOBase) else sys.stdin

    # This code didn't work:
    # I suspect it's because after we even call SUBPROCESS.PIPE, the stdin is its stdin? Or sth? No idea.
    # stdin_target = stdin if isinstance(stdin, io.IOBase) else subprocess.PIPE   # is stdin if stdin is any kind of file (text or buffered)
    # if terminal_inp: stdin_target = sys.stdin

    # And this doesn't work:
    # stdin_target = stdin if isinstance(stdin, io.IOBase) else sys.stdin if terminal_inp else subprocess.PIPE

    # And for some ungodly reason, this doesn't work either:
    # stdin_target = None
    # if stdin is None:
    #     if terminal_inp:
    #         stdin_target = sys.stdin
    #     else:
    #         stdin_target = subprocess.PIPE
    # elif isinstance(stdin, io.IOBase):
    #     # If stdin is a file-like object, we use it directly
    #     stdin_target = stdin
    # elif isinstance(stdin, str):
    #     # If stdin is a string, we will pass it as input to the process
    #     stdin_target = subprocess.PIPE
    # else:
    #     raise ValueError(f"stdin must be None, a file-like object, or a string. Got {type(stdin)} instead.")

    """

    input = stdin if isinstance(stdin, str) else None
    stdout_target = open(stdout_path, 'w') if stdout_path else subprocess.PIPE
    stderr_target = open(stderr_path, 'w') if stderr_path else subprocess.PIPE

    # you run scripts in a few ways:
    # sourcing: ['source', 'script.sh'] (but script.sh needs execute permission, so use make_executable beforehand)
    # bash:   ['bash', 'script.sh']
    # python: ['python', 'script.py']
    # sbatch: ['sbatch', 'script.sbatch']
    # srun:   ['srun', 'python', 'script.py']





    if shell:
        if isinstance(args, list):
            # If shell=True, args should be a single string command
            args = ' '.join(str(i) for i in args)

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