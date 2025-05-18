


import argparse
from sysrun.bashpy_boilerplate.diplomska_boilerplate import run_me, boilerplate, temp_file_strs
import y_helpers.shared_debug as debug
debug.start_debug()

parser = argparse.ArgumentParser()
parser.add_argument("path_to_yaml", type=str)
parser.add_argument("module_path_to_this_file", type=str, help="""e.g. a.b.c.y_train    From root of project (should be your cwd in the terminal also).""")
args = parser.parse_args()

main_name, auto_main_args, sd_path, main_yaml_path, out_folder_path = boilerplate(args.path_to_yaml, args.module_path_to_this_file).values()

# Here, the necessary auto_main_args are to be asserted.




command = ["python3", main_name] + auto_main_args + ["-p", "--sd", sd_path, "--yaml", main_yaml_path]
run_me(command, out_folder_path)

