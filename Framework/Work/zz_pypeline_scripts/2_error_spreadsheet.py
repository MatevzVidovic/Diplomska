

import pickle

import argparse
import pandas as pd
import numpy as np

import os.path as osp

import sys



import y_helpers.yaml_handler as yh



# srun python3 -m zz_pypeline_scripts.2_error_spreadsheet z_pipeline_unet_veins/standalone_scripts/trial.yaml

# srun python3 -m zz_pypeline_scripts.2_error_spreadsheet z_pipeline_segnet_veins/standalone_scripts/trial.yaml

# srun python3 -m zz_pypeline_scripts.2_error_spreadsheet z_pipeline_unet_sclera/standalone_scripts/trial.yaml

# srun python3 -m zz_pypeline_scripts.2_error_spreadsheet z_pipeline_segnet_sclera/standalone_scripts/trial.yaml




import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument("yaml_path", type=str)
args = argparser.parse_args()
yaml_path = args.yaml_path

YD = yh.read_yaml(yaml_path)



# srun --pty -p dev -c 7 --gpus=A100 python3 v_error_spreadsheet.py



training_logs_name = YD["training_logs_name"]

trial_name = YD["trial_name"]
csv_name = YD["csv_name"]
csv_path = osp.join(trial_name, csv_name)



model_name = YD["model_name"]

pruning_methods = YD["pruning_methods"]
retained_percents = YD["retained_percents"]

base_path = YD["trial_name"]
yaml_id = YD["yaml_id"]
origin_prefix = YD["origin_prefix"]
origin_suffix = YD["origin_suffix"]

folder_structure = {}

folder_structure[osp.join(base_path)] = [ (f"{model_name}_full_train_{yaml_id}", "100%") ]


for pm in pruning_methods:

    method_folder_name = f"{origin_prefix}{pm}{origin_suffix}_pruned"
    curr_path = osp.join(base_path, "pruned", method_folder_name)

    folder_structure[curr_path] = []
    for rp in retained_percents:
        rp_perc = int(rp * 100)
        folder_structure[curr_path].append( (f"{method_folder_name}_{rp}", f"{pm}_{rp_perc}%") )



print(folder_structure)


def add_row(df_np, row):
    
    if df_np is None:
        df_np = np.array([row])
    else:
        df_np = np.vstack([df_np, row])

    return df_np


models_np = None
errors_np = None
is_best_np = None


# err_titles = ["val MCDL", "val IoU", "test MCDL", "test IoU"]

sections = ["val", "test"]
loss_names = ["loss", "F1", "IoU", "precision", "recall"]

err_titles = []
for s in sections:
    for ln in loss_names:
        err_titles.append(f"{s} {ln}")

is_best_titles = [f"is best {et}" for et in err_titles]


for base_f, main_fs in folder_structure.items():

    for main_f, model in main_fs:


        try:

            to_tl = osp.join(base_f, main_f, "saved_main", "training_logs", training_logs_name)
            with open(to_tl, 'rb') as f:
                training_logs = pickle.load(f)

            last_error = training_logs.last_log


            print(f"{last_error=}")


            errs = []

            for s in sections:
                for ln in loss_names:
                    errs.append( float(last_error[f"{s}_err"][f"{ln}"]) )

            
            models_np = add_row(models_np, model)
            errors_np = add_row(errors_np, errs)

        except Exception as e:
            print(e)
            print(f"Error in {base_f}/{main_f}")



print(models_np)
print(errors_np)







# Create an array of False with the same shape as 'errors_np'
is_best_np = np.zeros_like(errors_np, dtype=bool)


for i in range(is_best_np.shape[1]):
    errors = errors_np[:, i]
    errors = errors.astype(float)
    # print(errors)
    # loss should be minimal, IoU should be maximal
    if i in [0, len(loss_names)]:
        best_ix = np.argmin(errors)
    else:
        best_ix = np.argmax(errors)
    is_best_np[best_ix, i] = True


collected_data = np.hstack([models_np, errors_np, is_best_np])

try:
    df = pd.DataFrame(collected_data, columns=["model"] + err_titles + is_best_titles)
except:
    print("Error in creating DataFrame")
    df = pd.DataFrame(collected_data)

# to csv
df.to_csv(csv_path, index=False)








