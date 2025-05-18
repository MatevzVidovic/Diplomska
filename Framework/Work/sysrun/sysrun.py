



"""

Note:

    If you're executing a shell command that includes shell-specific syntax
      (like pipes | or redirection >), you should set shell=True and pass the command 
      as a single string. For example:
      process = subprocess.Popen("ls -l | grep py", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

"""


import os
import os.path as osp
import argparse
# import shutil as sh

from pathlib import Path

from sysrun.helpers.help import get_yaml, write_yaml, run, run_no_wait, get_fresh_folder, recursive_update, my_update, recursive_check


SYSTEMS_RUN_COMMAND = "bash" # "sbatch"
# would be nice to alias sbatch as bash for local use, 
# but python Popen won't recognise it unless shell=True, wo we can't do it

# python3 -m sysrun.sysrun a/b/c/y_train.py --um








parser = argparse.ArgumentParser()
parser.add_argument("path_to_run", type=str) #, required=True)
parser.add_argument("--um", action="store_true", default=False, help="unconstrained mode")
parser.add_argument("--yamls", nargs='+', help="paths to additional yamls to add to the constructed yaml dict", default=[])
parser.add_argument("--args", nargs='+', help="args we overwrite in the end. Passing sys:gpus:a100_80GB will change that arg.", default=[])
parser.add_argument("--test_yaml", type=str, help="Pass path from the root of the project to a template yaml. If any field, \
                    which is present in the template yaml, isn't present in the constructed yaml, we tell you about it. \
                    Either way, we stop before running anything.", default=None)

# parser.add_argument("--hours_time_limit", type=int, default=48)
args = parser.parse_args()

path_to_run = args.path_to_run
unconstrained_mode = args.um



path_to_run = path_to_run.split("/")
dirs_to_run = [Path(path_to_run[0])]
for i in range(1, len(path_to_run)):
    dirs_to_run.append(dirs_to_run[i-1] / Path(path_to_run[i]))

path_to_run = dirs_to_run.pop(-1)




# print(f"Paths to run: {dirs_to_run}")
# for path in dirs_to_run:
#     os.makedirs(path/"test", exist_ok=True)
#     os.makedirs(path/"yamls", exist_ok=True)
# input()
# for path in dirs_to_run:
#     sh.rmtree(path/"test")




# ---------- Building the yaml dict ----------

YD = {}

main_dir_yaml = get_yaml("dirdef.yaml")
YD = my_update(YD, main_dir_yaml, Path("."))

for dirpath in dirs_to_run:
    dirdef_path = dirpath / "dirdef.yaml"
    dirdef_yaml = get_yaml(dirdef_path)
    YD = my_update(YD, dirdef_yaml, dirpath)





file_yaml_path = path_to_run.with_suffix(".yaml")
file_yaml = get_yaml(file_yaml_path)
# I have no idea why we would want retain_yamls=True here.
# We don't use the file_yaml anywhere anyway. And it just means that YD will then get the yamls_key added to it as well.
# Maybe just so that when the YD is stored into a file for logging/observability purposes that the yamls are also there?
# And maybe we don't want to do this in previous dirdef steps, 
# so that in that process we don't keep the yamls key which could cause later unintended relative path yaml overrides?
YD = my_update(YD, file_yaml, path_to_run.parent, retain_yamls=True)



for path in args.yamls:
    
    # absolute in the sense that it is relative to the project root
    absolute_yaml_path = Path(path)

    # relative yaml path (relative to where the runfile is)
    relative_yaml_path = dirs_to_run[-1] / path
    
    if osp.exists(absolute_yaml_path):
        new_yaml = get_yaml(absolute_yaml_path)
        YD = recursive_update(YD, new_yaml)
    elif osp.exists(relative_yaml_path):
        new_yaml = get_yaml(relative_yaml_path)
        YD = recursive_update(YD, new_yaml)
    else:
        print(f"Warning: {path} does not exist. Skipping.")



args_dict = {}
for arg in args.args:

    # e.g. sys:gpus:a100_80GB
    arg = arg.split(":")
    
    curr_dict = args_dict
    for i in range(len(arg)-2):
        key = arg[i]
        if key not in curr_dict:
            curr_dict[key] = {}
        curr_dict = curr_dict[key]

    curr_dict[arg[-2]] = arg[-1]

YD = recursive_update(YD, args_dict)









# ---------- Testing in comparing with template yaml ----------

if args.test_yaml is not None:
    test_yaml = get_yaml(Path(args.test_yaml))

    missing_keys = recursive_check(YD, test_yaml)    
    
    if len(missing_keys) > 0:
        print(f"Missing keys in constructed yaml: {missing_keys}")
        exit(1)
    else:
        print("All keys are present in the constructed yaml.")
        exit(0)







# ---------- Running stuff ----------

sysrun_path = Path("sysrun") / "sysrun_temp"
os.makedirs(sysrun_path, exist_ok=True)
sysrun_yaml_path = sysrun_path  / "sysrun.yaml"
write_yaml(YD, sysrun_yaml_path)


out_folder_path = get_fresh_folder("sysrun_runner_outputs")
outfile_path = out_folder_path / "out.txt"
err_path = out_folder_path / "err.txt"


# Making a symlink to the latest output folder
out_folder_path_absolute = out_folder_path.resolve()
symlink_to_latest = out_folder_path_absolute.parent / "0_symlink_to_latest"
# Uses Path object fns
if symlink_to_latest.exists() or symlink_to_latest.is_symlink():
    symlink_to_latest.unlink()
symlink_to_latest.symlink_to(out_folder_path_absolute, target_is_directory=True)




# We have to do this, so that our bashrun.py is run as a module, so it's cwd is the project root, and it can import other packages, such as sysrun.helpers.bashpy_help
module_path_to_run = str(path_to_run).replace("/", ".").removesuffix(".py") # we also have to rmove the suffix for the module run
# print(f"path_to_run: {path_to_run}")
# print(f"module_path_to_run: {module_path_to_run}")

if not unconstrained_mode:

    sysrun_sbatch_path = sysrun_path / "sysrun.sbatch"
    sys_args = YD.get("sys", None)
    sys_args_string = ""
    for key, value in sys_args.items():
        sys_args_string += f"#SBATCH {key}{value}\n"


    sbatch_file = f"""#!/bin/bash

{sys_args_string}

python3 -m {module_path_to_run} {sysrun_yaml_path} {module_path_to_run}
"""
    
    with open(sysrun_sbatch_path, 'w') as f:
        f.write(sbatch_file)
    
    # make_executable(sysrun_sbatch_path)

    args = [SYSTEMS_RUN_COMMAND, sysrun_sbatch_path, sysrun_yaml_path, sysrun_sbatch_path]

    # Here I think we can actually wait. Because when you run sbatch, you immediately get a response "Submited with id 31985" or whatever.
    # You arent actually waiting on the sbatch process itself.
    stdout, stderr, exit_code = run(args, stdout_path=outfile_path, stderr_path=err_path)
    # run_no_wait(args)


else:

    args = ["python3", "-m", module_path_to_run, sysrun_yaml_path, module_path_to_run]

    # # We don't want to be waiting.
    # # Otherwise, this will keep being an alive process on the login node, and noone wants that.
    # stdout, stderr, exit_code = run(args, stdout_path=outfile_path, stderr_path=err_path)
    # print(f"stdout: {stdout}")
    # print(f"stderr: {stderr}")
    # print(f"exit_code: {exit_code}")
    
    run_no_wait(args, stdout_path=outfile_path, stderr_path=err_path)





