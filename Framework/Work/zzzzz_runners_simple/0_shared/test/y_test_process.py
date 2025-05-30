


import argparse
from sysrun.bashpy_boilerplate.diplomska_boilerplate import run_me, boilerplate, temp_file_strs
import sysrun.bashpy_boilerplate.shared_remote_debug as debug
debug.start_remote_debug()

parser = argparse.ArgumentParser()
parser.add_argument("path_to_yaml", type=str)
parser.add_argument("module_path_to_this_file", type=str, help="""e.g. a.b.c.y_train.""")
args = parser.parse_args()

whole_yaml, out_folder_path, sysrun_runner_outputs_path, main_name, sd_path, main_yaml_path = boilerplate(args.path_to_yaml, args.module_path_to_this_file).values()

# Here, the necessary bashpy_args are to be asserted and processed.
if (model_yaml := whole_yaml["oth"]["bashpy_args"].get("model_yaml", None)) is None:
    raise ValueError("model_yaml not given in bashpy_args.")
added_args = []
if "--ifn" not in whole_yaml["oth"]["bashpy_args"]:
    added_args = ["--ifn", whole_yaml["oth"]["bashpy_args"]["main_ifn"]]
else:
    added_args = whole_yaml["oth"]["bashpy_args"]["--ifn"]



# pruning_phase
command = ["python3", main_name, "--map", "1", "--ntibp", "1", "--tras", "2", "-p", "--sd", sd_path, "--yaml", main_yaml_path] + added_args
run_me(command, out_folder_path, sysrun_runner_outputs_path)

from pathlib import Path
relative_path_to_this_file = args.module_path_to_this_file.replace(".", "/")

path_to_z_get_graphs = Path(relative_path_to_this_file).parent.parent / "standalone_scripts" / "z_get_graphs.py" # relative path from root of project
command = ["python3", "-m", "sysrun.sysrun", "--ns", path_to_z_get_graphs, "--bash", "--yamls", f"../model_yamls/{model_yaml}.yaml", "../hpc_yamls/basic.yaml", "--args", f"oth:bashpy_args:--sd_path:{sd_path}"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

path_to_z_get_fw = Path(relative_path_to_this_file).parent.parent / "standalone_scripts" / "z_get_fw.py" # relative path from root of project
command = ["python3", "-m", "sysrun.sysrun", "--ns", path_to_z_get_fw, "--bash", "--yamls", f"../model_yamls/{model_yaml}.yaml", "../hpc_yamls/basic.yaml", "--args", f"oth:bashpy_args:--sd_path:{sd_path}"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)


command = ["python3", main_name, "--ips", "0", "--sd", sd_path, "--yaml", main_yaml_path]
run_me(command, out_folder_path, sysrun_runner_outputs_path, stdin=temp_file_strs["save_and_stop"])
