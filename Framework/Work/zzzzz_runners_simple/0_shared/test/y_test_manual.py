


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


# Meant for manual steering. So running it in your terminal with bash and then writing the stdin by hand is necessary.
command = ["python3", main_name, "--ips", "0", "--tras", "2", "--mti", "2", "--sd", sd_path, "--yaml", main_yaml_path]
run_me(command, out_folder_path, sysrun_runner_outputs_path, terminal_inp=True)


# # pruning_phase
# for i in range(1):
#     command = ["python3", main_name, "--ntibp", "1", "--ptp", "0.0000001", "--map", "1", "--tras", "2", "--tp", "--ifn", "uniform", "-p", "--mti", "2", "--sd", sd_path, "--yaml", main_yaml_path]
#     run_me(command, out_folder_path, sysrun_runner_outputs_path)
