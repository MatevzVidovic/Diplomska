
import os
import os.path as osp
import shutil as sh
import yaml
import argparse

import fcntl

# import contextlib

# @contextlib.contextmanager
# def file_lock(filename):
#     with open(filename, 'r') as f:
#         try:
#             fcntl.flock(f.fileno(), fcntl.LOCK_EX)
#             yield
#         finally:
#             fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            


parser = argparse.ArgumentParser()
parser.add_argument("--max_run", type=int, required=True)
args = parser.parse_args()
max_run = args.max_run



# python3 -m trial_1_sclera.run_files_1.run_all --max_run 3


def get_yaml(path):
    if osp.exists(path):
        with open(path, 'r') as f:
            YD = yaml.safe_load(f)

            # when yaml is empty
            if YD is None:
                YD = {}
    else:
        YD = {}
    return YD



# The space between yaml get and yaml dump needs to be enclosed in a lock
# to prevent race conditions.




# path = osp.join("trial_1_sclera", "run_files_1")
# yaml_path = osp.join("trial_1_sclera", "run_files_1", "run_all.yaml")

# print(osp.abspath(osp.join(".", "run_all.yaml")))
# get path of file
path = osp.dirname(__file__)
yaml_path = osp.join(path, "runner.yaml")


with open(yaml_path, 'r+') as f:



    all_files = os.listdir(path)

    all_files = [f for f in all_files if not f.startswith("ana_") and not f.startswith("run_") and f.endswith(".sbatch")]
    all_files = sorted(all_files)


    run_count = 0

    for run_file in all_files:


        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        YD = get_yaml(yaml_path)
        file_info = YD.get(run_file, {})
        file_run = file_info.get("run", False)

        if file_run:
            continue

        YD[run_file] = {"run": True, "finished": False}
        yaml.dump(YD, open(yaml_path, 'w'))
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)



        print(f"Running {run_file}")
        os.system(f"bash {osp.join(path, run_file)}")
        run_count += 1


        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        YD = get_yaml(yaml_path)
        YD[run_file]["finished"] = True
        yaml.dump(YD, open(yaml_path, 'w'))
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)



        if run_count >= max_run:
            break