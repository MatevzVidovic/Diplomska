


import argparse
from sysrun.bashpy_boilerplate.diplomska_boilerplate import run_me, boilerplate, temp_file_strs
import sysrun.bashpy_boilerplate.shared_remote_debug as debug
debug.start_remote_debug()

parser = argparse.ArgumentParser()
parser.add_argument("path_to_yaml", type=str)
parser.add_argument("module_path_to_this_file", type=str, help="""e.g. a.b.c.y_train.""")
args = parser.parse_args()

whole_yaml, out_folder_path, main_name, sd_path, main_yaml_path = boilerplate(args.path_to_yaml, args.module_path_to_this_file).values()

# Here, the necessary bashpy_args are to be asserted and processed.
assert "--mti" in whole_yaml["oth"]["bashpy_args"]
added_args = ["--mti", whole_yaml["oth"]["bashpy_args"]["--mti"]]



command = ["python3", main_name, "--sd", sd_path, "--yaml", main_yaml_path] + added_args
run_me(command, out_folder_path)

