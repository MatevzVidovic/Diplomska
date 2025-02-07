



import os
import os.path as osp
import shutil as sh
import sys


import y_helpers.yaml_handler as yh


yaml_path = osp.join(osp.dirname(__file__), "trial.yaml")
YD = yh.read_yaml(yaml_path)



# pipeline_name = "z_pipeline_unet_veins"

# yaml_id = "vein"

# trial_name = "trial_1"

# origin_dir_name = "unet_train_vein"

# versions_to_make = ["random", "uniform", "IPAD_eq", "IPAD1_L1", "IPAD2_L2", "IPAD1", "IPAD2", "L1", "L2"]

main_name = YD["main_name"]
model_name = YD["model_name"]

pipeline_name = YD["pipeline_name"]

yaml_id = YD["yaml_id"]

trial_folder = YD["trial_name"]

origin_dir_name = YD["origin_dir_name"]

versions_to_make = YD["versions_to_make"]

mtti = YD["mtti"]

core_num = YD["core_num"]

origin_prefix = YD["origin_prefix"]
origin_suffix = YD["origin_suffix"]






origin_dir_path = osp.join(origin_dir_name)
save_dirs = [f"{origin_prefix}{v}{origin_suffix}" for v in versions_to_make]




to_run_folder = osp.join(trial_folder, "run_files_1")
os.makedirs(to_run_folder, exist_ok=True)

to_out_folder = osp.join(trial_folder, "out_files_1")
os.makedirs(to_out_folder, exist_ok=True)




# first one is the full train

sd = f"{model_name}_full_train{origin_suffix}"

sd_path = osp.join(trial_folder, sd)
sh.copytree(origin_dir_path, sd_path)

out_name = f"x_{sd}_out.txt"
to_out = osp.join(to_out_folder, out_name)

sbatch_name = f"{sd}.sbatch"
sbatch = f"""#!/bin/bash

#SBATCH --job-name={sd}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G


python3 {main_name} --mtti {mtti} --sd {trial_folder}/{sd} --yaml {pipeline_name}/{yaml_id}.yaml >> {to_out} 2>&1

"""


ana_sbatch_name = f"ana_{sd}.sbatch"
ana_sbatch = f"""#!/bin/bash

#SBATCH --job-name={sd}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100_80GB
#SBATCH --mem-per-cpu=6G


python3 {main_name} --mtti {mtti} --sd {trial_folder}/{sd} --yaml {pipeline_name}/{yaml_id}.yaml >> {to_out} 2>&1

"""


to_sbatch = osp.join(to_run_folder, sbatch_name)
with open(to_sbatch, "w") as f:
    f.write(sbatch)

to_ana_sbatch = osp.join(to_run_folder, ana_sbatch_name)
with open(to_ana_sbatch, "w") as f:
    f.write(ana_sbatch)

print(f"Made {sd}")






for sd in save_dirs:
    # if osp.exists(sd):
    #     shutil.rmtree(sd)
    # os.makedirs(sd)

    sd_path = osp.join(trial_folder, sd)
    # os.makedirs(sd_path, exist_ok=True)
    sh.copytree(origin_dir_path, sd_path)


    # os.system(f"cp -r {origin_dir_path}/* {sd}")

    # # Fixing the mistake of not coppying stuff okay:

    # init_res_calc = osp.join(sd, "saved_model_wrapper", "initial_conv_resource_calc.pkl")
    # os.system(f"rm {init_res_calc}")

    # real_res_calc = osp.join("unet_prune_IPAD", "saved_model_wrapper", "initial_conv_resource_calc.pkl")
    # os.system(f"cp {real_res_calc} {init_res_calc}")

    # print(f"Made {sd}")






for sd, version in zip(save_dirs, versions_to_make):
    out_name = f"x_{sd}_out.txt"
    to_out = osp.join(to_out_folder, out_name)

    sbatch_name = f"{sd}.sbatch"
    sbatch = f"""#!/bin/bash

#SBATCH --job-name={sd}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G


python3 {main_name} --ifn {version} -p --sd {trial_folder}/{sd} --yaml {pipeline_name}/{yaml_id}.yaml >> {to_out} 2>&1

"""
    
    ana_sbatch_name = f"ana_{sd}.sbatch"
    ana_sbatch = f"""#!/bin/bash

#SBATCH --job-name={sd}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G


python3 {main_name} --ifn {version} -p --sd {trial_folder}/{sd} --yaml {pipeline_name}/{yaml_id}.yaml >> {to_out} 2>&1

"""
    
    to_sbatch = osp.join(to_run_folder, sbatch_name)
    with open(to_sbatch, "w") as f:
        f.write(sbatch)

    to_ana_sbatch = osp.join(to_run_folder, ana_sbatch_name)
    with open(to_ana_sbatch, "w") as f:
        f.write(ana_sbatch)

    print(f"Made {sd}")






# just have to_run_folder be set up correctly, and this whould go well:



# now make the automatic runnning mechanism
# We will have 2 sbatches that will run the python file.
# One will run normally, and one will run it on ana.
# Both sbatches have the max time limit of 2 days.

# The sbatch files take the arguments for the python file.

# Python file will be the runner.
# It has a yaml file in its run folder (creates it if it doesn't exist).
# Lists all the runner files in the run folder, with dicts: {"run": False, "finished": False}
# Before starting a run, it reads the run folder, figures out which in its sequence hasn't been run yet,
# writes to the yaml that thet file "run" is true, and runs the file.
# Upon finish it writes to the yaml that the file "finished" is true.

# The python file takes 2 args: 
# pos: start, mid_up, mid_down, end     # where in the sorted listdir it starts seeking to run files.
# max_run: int                          # how many files to run at most. Otherwise we are sure to surpass the time-limit at some point.

to_py_runner_yaml = osp.join(to_run_folder, "runner.yaml")
with open(to_py_runner_yaml, "w") as f:
    f.write("")


to_py_runner = osp.join(to_run_folder, "runner.py")
to_origin_runner = osp.join(osp.dirname(__file__), "runner.py")
sh.copy2(to_origin_runner, to_py_runner)


sbatch_name = "run_sbatch.sbatch"
sbatch = f"""#!/bin/bash

#SBATCH --job-name=runner
#SBATCH --time=2-00:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G

max_run=$1

python3 {to_py_runner} --max_run $max_run
"""

ana_sbatch_name = "ana_run_sbatch.sbatch"
sbatch = f"""#!/bin/bash

#SBATCH --job-name=runner
#SBATCH --time=2-00:00:00

#SBATCH -p frida
#SBATCH -c {core_num}
#SBATCH --gpus=A100_80GB
#SBATCH --mem-per-cpu=6G

max_run=$1

python3 {to_py_runner} --max_run $max_run
"""


to_sbatch = osp.join(to_run_folder, sbatch_name)
with open(to_sbatch, "w") as f:
    f.write(sbatch)

to_ana_sbatch = osp.join(to_run_folder, ana_sbatch_name)
with open(to_ana_sbatch, "w") as f:
    f.write(ana_sbatch)

