

import argparse

from sysrun.helpers.help import run
from sysrun.bashpy_boilerplate.diplomska_boilerplate import boilerplate, get_novel_out_and_err



parser = argparse.ArgumentParser()
parser.add_argument("path_to_yaml", type=str)
parser.add_argument("module_path_to_this_file", type=str, help="e.g. a.b.c.y_train")
args = parser.parse_args()

main_name, auto_main_args, sd_path, main_yaml_path, out_folder_path = boilerplate(args.path_to_yaml, args.module_path_to_this_file).values()





import y_helpers.shared_debug as debug
debug.start_debug()


# auto_main_args needs added_auto_main_args: --ifn: IPAD1
assert "--mti" in auto_main_args

command = ["python3", main_name] + auto_main_args + ["--sd", sd_path, "--yaml", main_yaml_path]
run(command, **get_novel_out_and_err(out_folder_path))

