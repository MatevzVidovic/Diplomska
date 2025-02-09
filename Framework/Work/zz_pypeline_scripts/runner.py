
import os
import stat
import os.path as osp
import shutil as sh
import yaml
import argparse
import time
import subprocess
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

"""
In the folder of this runner, find all .sbatch files (not starting with ana_    so we don't do the same task twice),
and run them.
Use locking to atomically update the yaml file, knowing:
- if the task has been started
- if the task has been finished
- the runtime of the task
- the exit code of the task

We know not to do an sbatch if yaml says the task has been started.

We can set max_run to limit the number of sbatch runs.

We also implement time limit cancel prevention, by checking the avg time of previous sbatches,
and stopping if we would probably exceed the time limit with the next task.
"""
            


parser = argparse.ArgumentParser()
parser.add_argument("--max_run", type=int, required=True)
parser.add_argument("--hours_time_limit", type=int, default=48)
args = parser.parse_args()
max_run = args.max_run
hours_time_limit = args.hours_time_limit
seconds_time_limit = hours_time_limit * 60 * 60



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


def make_executable(file_path):
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)


def get_seconds(start_seconds):
    return time.perf_counter() - start_seconds

def seconds_to_dhms(seconds):
    """Convert seconds to days, hours, minutes, seconds format"""
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    seconds = int(seconds)
    to_str = f"{days}d-{hours}h-{minutes}m-{seconds}s"
    return to_str

def update_avg_time(prev_avg_time, new_time):

    prev_avg_seconds, prev_run_count = prev_avg_time
    new_run_count = prev_run_count + 1

    new_avg_seconds = (prev_avg_seconds * prev_run_count + new_time) / new_run_count

    new_avg_time = (new_avg_seconds, new_run_count)
    return new_avg_time



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
    avg_time = (0, 0) # avg_seconds, run_count
    start_time = time.perf_counter()

    for run_file in all_files:


        elapsed_since_start = get_seconds(start_time)
        if elapsed_since_start + avg_time[0] >= seconds_time_limit:
            print(f"Close to time limit. Stopping to prevent cancelation.")
            break


        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        YD = get_yaml(yaml_path)
        file_info = YD.get(run_file, {})
        file_run = file_info.get("run", False)

        if file_run:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            continue

        YD[run_file] = {"run": True, "finished": False}
        yaml.dump(YD, open(yaml_path, 'w'))
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)



        print(f"Running {run_file}")
        run_file_path = osp.join(path, run_file)
        make_executable(run_file_path)    # since we arent running it with "bash {command}"
        run_start_time = time.perf_counter()
        # os.system(f"bash {osp.join(path, run_file)}")
        process = subprocess.Popen([f'{run_file_path}'], 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True)
        stdout, stderr = process.communicate() # also waits for process to terminate
        exit_code = process.returncode # available after the process terminates
        run_count += 1

        run_seconds = get_seconds(run_start_time)
        avg_time = update_avg_time(avg_time, run_seconds)
        run_time_str = seconds_to_dhms(run_seconds)



        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        YD = get_yaml(yaml_path)
        YD[run_file]["finished"] = True
        YD[run_file]["run_time"] = run_time_str
        YD[run_file]["exit_code"] = exit_code
        if exit_code != 0:
            failed_files = YD.get("failed_files", None)
            if failed_files is None:
                failed_files = []
            YD["failed_files"] = failed_files + [run_file]
        yaml.dump(YD, open(yaml_path, 'w'))
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)



        if run_count >= max_run:
            break