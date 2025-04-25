



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
import io

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


def make_executable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)

def run(args, stdout_path=None, stderr_path=None, stdin=None, shell=False):
    # is shel=True, args is a string and not a list

    # Open the specified files or default to subprocess.PIPE
    stdin_target = stdin if isinstance(stdin, io.IOBase) else subprocess.PIPE   # is stdin if stdin is any kind of file (text or buffered)
    input = input if isinstance(stdin, str) else None
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