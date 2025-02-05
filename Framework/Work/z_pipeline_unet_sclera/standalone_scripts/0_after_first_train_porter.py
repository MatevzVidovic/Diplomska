



import os
import os.path as osp
import shutil
import sys


import y_helpers.yaml_handler as yh


yaml_path = osp.join("z_pipeline_unet_sclera", "standalone_scripts", "trial_1.yaml")
YD = yh.read_yaml(yaml_path)



# pipeline_name = "z_pipeline_unet_sclera"

# yaml_id = "vein"

# trial_name = "trial_1"

# origin_dir_name = "unet_train_vein"

# versions_to_make = ["random", "uniform", "IPAD_eq", "IPAD1_L1", "IPAD2_L2", "IPAD1", "IPAD2", "L1", "L2"]


pipeline_name = YD["pipeline_name"]

yaml_id = YD["yaml_id"]

trial_name = YD["trial_name"]

origin_dir_name = YD["origin_dir_name"]

versions_to_make = YD["versions_to_make"]

mtti = YD["mtti"]






origin_dir_path = osp.join(origin_dir_name)
save_dirs = [f"unet_prune_{v}_vein" for v in versions_to_make]




run_files_path = osp.join(trial_name, "run_files_1")

os.makedirs(run_files_path, exist_ok=True)




# first one is the full train

sd = "unet_full_train_vein"

sd_path = osp.join(trial_name, sd)
shutil.copytree(origin_dir_path, sd_path)

out_name = f"x_{sd}_out.txt"

run_file = f"""#!/bin/bash

#SBATCH --job-name={sd}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G


python3 unet_main.py --mtti {mtti} --sd {trial_name}/{sd} --yaml {pipeline_name}/{yaml_id}.yaml >> {out_name} 2>&1

"""
    
with open(osp.join(run_files_path, f"y_{sd}.sbatch"), "w") as f:
    f.write(run_file)

print(f"Made {sd}")







for sd in save_dirs:
    # if osp.exists(sd):
    #     shutil.rmtree(sd)
    # os.makedirs(sd)

    sd_path = osp.join(trial_name, sd)
    # os.makedirs(sd_path, exist_ok=True)
    shutil.copytree(origin_dir_path, sd_path)


    # os.system(f"cp -r {origin_dir_path}/* {sd}")

    # # Fixing the mistake of not coppying stuff okay:

    # init_res_calc = osp.join(sd, "saved_model_wrapper", "initial_conv_resource_calc.pkl")
    # os.system(f"rm {init_res_calc}")

    # real_res_calc = osp.join("unet_prune_IPAD", "saved_model_wrapper", "initial_conv_resource_calc.pkl")
    # os.system(f"cp {real_res_calc} {init_res_calc}")

    # print(f"Made {sd}")






for sd, version in zip(save_dirs, versions_to_make):
    out_name = f"x_{sd}_out.txt"

    run_file = f"""#!/bin/bash

#SBATCH --job-name={sd}
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=6G


python3 unet_main.py --ifn {version} -p --sd {trial_name}/{sd} --yaml {pipeline_name}/{yaml_id}.yaml >> {out_name} 2>&1

"""
    
    with open(osp.join(run_files_path, f"y_{sd}.sbatch"), "w") as f:
        f.write(run_file)

    print(f"Made {sd}")

