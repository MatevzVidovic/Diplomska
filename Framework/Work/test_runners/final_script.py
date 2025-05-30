



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




command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/basic_run/y_train.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--mti:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/basic_run/y_prune.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.multi.unet.basic_run.y_train", "oth:bashpy_args:--map:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/basic_run/y_train.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--mti:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/basic_run/y_prune.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.basic_run.y_train", "oth:bashpy_args:--map:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/basic_run/y_train.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--mti:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/basic_run/y_prune.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.sclera.unet.basic_run.y_train", "oth:bashpy_args:--map:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/basic_run/y_train.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--mti:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/basic_run/y_prune.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.veins.segnet.basic_run.y_train", "oth:bashpy_args:--map:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/basic_run/y_train.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--mti:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/basic_run/y_prune.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.veins.unet.basic_run.y_train", "oth:bashpy_args:--map:2"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/standalone_scripts/z_get_da.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.multi.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/standalone_scripts/z_get_fw.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.multi.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/standalone_scripts/z_get_graphs.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.multi.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/standalone_scripts/z_get_sp.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.multi.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/standalone_scripts/z_get_ts.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.multi.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/standalone_scripts/z_get_da.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/standalone_scripts/z_get_fw.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/standalone_scripts/z_get_graphs.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/standalone_scripts/z_get_sp.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/standalone_scripts/z_get_ts.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.sclera.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/standalone_scripts/z_get_da.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.sclera.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/standalone_scripts/z_get_fw.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.sclera.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/standalone_scripts/z_get_graphs.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.sclera.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/standalone_scripts/z_get_sp.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.sclera.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/standalone_scripts/z_get_ts.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.sclera.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/standalone_scripts/z_get_da.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.veins.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/standalone_scripts/z_get_fw.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.veins.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/standalone_scripts/z_get_graphs.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.veins.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/standalone_scripts/z_get_sp.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.veins.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/standalone_scripts/z_get_ts.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/segnet_train___zzzzz_runners_simple.veins.segnet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/standalone_scripts/z_get_da.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.veins.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/standalone_scripts/z_get_fw.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.veins.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/standalone_scripts/z_get_graphs.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.veins.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/standalone_scripts/z_get_sp.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.veins.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/standalone_scripts/z_get_ts.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:--sd_path:active_models/unet_train___zzzzz_runners_simple.veins.unet.basic_run.y_train"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/test/y_test_main.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/test/y_test_process.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:model_yaml:multi_fake"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/multi/unet/test/y_test_quick.py", "--bash", "--yamls", "../model_yamls/multi_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/test/y_test_main.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/test/y_test_process.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:model_yaml:sclera_fake"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/segnet/test/y_test_quick.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/test/y_test_main.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/test/y_test_process.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:model_yaml:sclera_fake"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/sclera/unet/test/y_test_quick.py", "--bash", "--yamls", "../model_yamls/sclera_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/test/y_test_main.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/test/y_test_process.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:model_yaml:veins_fake"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/segnet/test/y_test_quick.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/test/y_test_main.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/test/y_test_process.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml", "--args", "oth:bashpy_args:model_yaml:veins_fake"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)

command = ["python3", "-m", "sysrun.sysrun", "--ns", "zzzzz_runners_simple/veins/unet/test/y_test_quick.py", "--bash", "--yamls", "../model_yamls/veins_fake.yaml", "../hpc_yamls/basic.yaml"]
run_me(command, out_folder_path, sysrun_runner_outputs_path, is_nested_sysrun=True)
