

"""
basic_run

y_prune
y_train

standalone_scripts

z_get_bs
z_get_da
z_get_fw
z_get_graphs
z_get_sp
z_get_ts



test

y_test_main
y_test_manual
y_test_process
y_test_quick
y_test_uniform


multi
unet

sclera
segnet
unet


veins
segnet
unet
"""

"""
# python3 -m sysrun.sysrun zzzzz_runners_simple/sclera/segnet/test/y_test_manual.py --yamls ../model_yamls/sclera_fake.yaml ../hpc_yamls/basic.yaml

"zzzzz_runners_simple"


"basic_run"

"model_yamls/sclera_fake.yaml",
"hpc_yamls/basic.yaml",

"y_train"
"--args",
"oth:bashpy_args:--mti:2",

"y_prune"
"--args",
"oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.y_train",
"oth:bashpy_args:--map:2",




"standalone_scripts"

"../model_yamls/sclera_fake.yaml" 
"../hpc_yamls/basic.yaml"

"z_get_bs"
"z_get_da"
"z_get_fw"
"z_get_graphs"
"z_get_sp"
"z_get_ts"



"test"

"y_test_main"
"y_test_manual"
"y_test_process"
"y_test_quick"
"y_test_uniform"


"multi"
"unet"

"sclera"
"segnet"
"unet"


"veins"
"segnet"
"unet"
"""






from pathlib import Path


# python3 -m sysrun.sysrun zzzzz_runners_simple/sclera/segnet/test/y_test_manual.py --yamls ../model_yamls/sclera_fake.yaml ../hpc_yamls/basic.yaml

base = "zzzzz_runners_simple"

"multi"
"unet"

"sclera"
"segnet"
"unet"


"veins"
"segnet"
"unet"

temp_paths = [("multi", "unet"),
              ("sclera", "segnet"),
                ("sclera", "unet"),
              ("veins", "segnet"),
                ("veins", "unet"),]
              

base_paths = []
for i in temp_paths:
    base_paths.append(Path(base) / i[0] / i[1])






run_args = []



run_paths = []


a = "basic_run"


for i in base_paths:
    b = [f"../model_yamls/{i.parts[-2]}_fake.yaml", "../hpc_yamls/basic.yaml"]
    path = i / a / "y_train.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", "oth:bashpy_args:--mti:2"))

    path = i / a / "y_prune.py"
    sd_path = f"active_models/{i.parts[-1]}_train___zzzzz_runners_simple.{i.parts[-2]}.{i.parts[-1]}.basic_run.y_train"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:--sd_path:{sd_path}", "oth:bashpy_args:--map:2"))





a = "standalone_scripts"

for i in base_paths:
    b = [f"../model_yamls/{i.parts[-2]}_fake.yaml", "../hpc_yamls/basic.yaml"]

    # this always gets an error, because it's supposed to run for long enough until it gets one.
    # Also, it takwes a bunch of time. One successful test was enough.
    # path = i / a / "z_get_bs.py"
    # run_args.append((path, "--bash", "--yamls", *b))

    sd_path = f"active_models/{i.parts[-1]}_train___zzzzz_runners_simple.{i.parts[-2]}.{i.parts[-1]}.basic_run.y_train"

    path = i / a / "z_get_da.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:--sd_path:{sd_path}"))

    path = i / a / "z_get_fw.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:--sd_path:{sd_path}"))

    path = i / a / "z_get_graphs.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:--sd_path:{sd_path}"))

    path = i / a / "z_get_sp.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:--sd_path:{sd_path}"))

    path = i / a / "z_get_ts.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:--sd_path:{sd_path}"))




a = "test"
for i in base_paths:
    b = [f"../model_yamls/{i.parts[-2]}_fake.yaml", "../hpc_yamls/basic.yaml"]

    path = i / a / "y_test_main.py"
    run_args.append((path, "--bash", "--yamls", *b))

    path = i / a / "y_test_process.py"
    run_args.append((path, "--bash", "--yamls", *b, "--args", f"oth:bashpy_args:model_yaml:{i.parts[-2]}_fake"))

    path = i / a / "y_test_quick.py"
    run_args.append((path, "--bash", "--yamls", *b))




"test"

"y_test_main"
# "y_test_manual" # manual steering
# "y_test_uniform" # takes too long
"y_test_process"
"y_test_quick"
"y_test_uniform" # takes too long









final_file = """



import argparse
from sysrun.bashpy_boilerplate.diplomska_boilerplate import run_me, basic_boilerplate, temp_file_strs
import sysrun.bashpy_boilerplate.shared_remote_debug as debug
debug.start_remote_debug()

parser = argparse.ArgumentParser()
parser.add_argument("path_to_yaml", type=str)
parser.add_argument("module_path_to_this_file", type=str, help="e.g. a.b.c.y_train.")
args = parser.parse_args()

whole_yaml, out_folder_path, sysrun_runner_outputs_path = basic_boilerplate(args.path_to_yaml, args.module_path_to_this_file).values()

# Here, the necessary bashpy_args are to be asserted and processed.



"""


for i in run_args:
    curr_args = list(i)
    args_str = ", ".join([f'"{str(arg)}"' for arg in curr_args])
    final_file += f"""
command = ["python3", "-m", "sysrun.sysrun", "--ns", {args_str}]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)
"""

with open("test_runners/final_script.py", "w") as f:
    f.write(final_file)