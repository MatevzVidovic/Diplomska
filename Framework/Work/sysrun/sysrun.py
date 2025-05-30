



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
import sys
import datetime
# import shutil as sh

from pathlib import Path

from sysrun.helpers.help import get_yaml, write_yaml, run, run_no_wait, get_fresh_folder, recursive_update, my_update, recursive_check, write_sbatch_file, symlink_safe_preresolve


SYSTEMS_RUN_COMMAND = "bash" # "sbatch"
# would be nice to alias sbatch as bash for local use, 
# but python Popen won't recognise it unless shell=True, wo we can't do it

# python3 -m sysrun.sysrun a/b/c/y_train.py --um








parser = argparse.ArgumentParser()
parser.add_argument("path_to_run", type=str) #, required=True)
parser.add_argument("--um", action="store_true", default=False, help="unconstrained mode")
parser.add_argument("--ns", action="store_true", default=False, help="nested sysrun - inside of an already running sysrun")
parser.add_argument("--bash", action="store_true", default=False, help="""Enforce bash. If set, no sys_args (frida args) are set.
                    And SYSTEMS_RUN_COMMAND is set to bash. This is useful running sysrun.sysrun inside of sysrun.sysrun.
                    For example, you might make sth like a bash cript that makes graphs for the model. In your runner script
                    you do some training and pruning. And then you want to run that bash-script-like runner (e.g. z_get_graphs.py).
                    And it needs to be run with sysrun.sysrun. This will make a new temp sth.sbatch and a new temp yaml,
                    and it will go run it. But you don't want it to be run with sbatch, because that would make it return instantly and run in the background.
                    So you set this flag and make it run with bash.""")
parser.add_argument("--no_run", action="store_true", default=False, help="if set, we will not run the command, just construct the yaml and .sbatch and exit. \
                    You then run the command manually later. It gives you more control.")
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
    absolute_yaml_path = symlink_safe_preresolve(Path(path))

    # relative yaml path (relative to where the runfile is)
    relative_yaml_path = symlink_safe_preresolve(dirs_to_run[-1] / path)

    print(f"Attempting to add yaml from {absolute_yaml_path} or {relative_yaml_path} to the constructed yaml dict.")
    
    if osp.exists(absolute_yaml_path):
        new_yaml = get_yaml(absolute_yaml_path)
        YD = recursive_update(YD, new_yaml)
    elif osp.exists(relative_yaml_path):
        new_yaml = get_yaml(relative_yaml_path)
        YD = recursive_update(YD, new_yaml)
    else:
        print(3*"!!!!!!!!!!\n" + f"!!!Warning!!!: {path} does not exist. Skipping." + 3*"!!!!!!!!!!\n")
        sys.exit(1) # to prevent erroneous runs



args_dict = {}
for arg in args.args:
    """
    Basic behaviour:
    args = [sys:gpus:a100_80GB sys:-p:"frida"]

    We still have many problems.
    This can only be used for string args.
    Ints, floats, lists, dicts, etc. are not supported.

    
    Also not supported, just a possible idea for the future:

    Also, having a lot of args starts to be unreadable.
    It would be nice to not duplicate the paths for args in the same subdict.
    So we could do:
    args = [sys:{gpus:a100_80GB!&&!-p:frida}]
    This special sequence could delimit the args in the same subdict.
    But it would be a bit complex.  
    
    """

    
    # Basic behaviour:
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
    
# cleanup moves all but last 4 sysrun dirs to old/
# but if we did 4 nested sysruns, then the original sysrun dir would now be moved to old/
# and this make all the paths stored in the running process wrong and crashes the program.
out_folder_path = get_fresh_folder("sysrun_runner_outputs", do_cleanup=(not args.ns)).absolute()
outfile_path = out_folder_path / "runner_out.txt"
err_path = out_folder_path / "runner_err.txt"
print(f" sysrun_runner_outputs_out_folder_path:[{out_folder_path.name}] \n") # .name is just the last part of the path, so it is the folder name

YD["sysrun_info"] = {}
YD["sysrun_info"]["sysrun_runner_outputs_out_folder_path"] = str(out_folder_path)

sysrun_path = Path(out_folder_path) / "sysrun_temp"
os.makedirs(sysrun_path, exist_ok=True)
sysrun_yaml_path = sysrun_path  / "sysrun.yaml"
write_yaml(YD, sysrun_yaml_path)




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

python_command = f"python3 -m {module_path_to_run} {sysrun_yaml_path} {module_path_to_run}"
running_time = datetime.datetime.now()
with open(out_folder_path / "sysrun_info.txt", "w") as f:
    f.write(f"Running time: {running_time}\n")
    f.write(f"Python command: {python_command}\n")

if not unconstrained_mode:

    sysrun_sbatch_path = sysrun_path / "sysrun.sbatch"
    write_sbatch_file(YD, sysrun_sbatch_path, python_command, no_sys_args=args.bash)
    
    # make_executable(sysrun_sbatch_path)
    if args.bash:
        command_args = ["bash", sysrun_sbatch_path, sysrun_yaml_path, sysrun_sbatch_path]
    else:
        command_args = [SYSTEMS_RUN_COMMAND, sysrun_sbatch_path, sysrun_yaml_path, sysrun_sbatch_path]

    # Here I think we can actually wait. Because when you run sbatch, you immediately get a response "Submited with id 31985" or whatever.
    # You arent actually waiting on the sbatch process itself.
    if not args.no_run:
        stdout, stderr, exit_code = run(command_args, stdout_path=outfile_path, stderr_path=err_path, stdin=sys.stdin)
    # run_no_wait(args)


else:

    command_args = ["python3", "-m", module_path_to_run, sysrun_yaml_path, module_path_to_run]

    # # We don't want to be waiting.
    # # Otherwise, this will keep being an alive process on the login node, and noone wants that.
    # stdout, stderr, exit_code = run(args, stdout_path=outfile_path, stderr_path=err_path)
    # print(f"stdout: {stdout}")
    # print(f"stderr: {stderr}")
    # print(f"exit_code: {exit_code}")
    if not args.no_run:
        run_no_wait(command_args, stdout_path=outfile_path, stderr_path=err_path)





