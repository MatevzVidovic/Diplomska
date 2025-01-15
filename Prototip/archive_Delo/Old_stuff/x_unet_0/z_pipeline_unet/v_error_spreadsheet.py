

import pickle

import argparse
import pandas as pd
import numpy as np

import os.path as osp

import sys





# srun --pty -p dev -c 7 --gpus=A100 python3 v_error_spreadsheet.py



folder_structure = {
    "" : [ ("unet_train2", "100%") ],
    "pruned_models_unet_IPAD" : [ ("unet_IPAD_490_after_pruning", "IPAD_75%"), ("unet_IPAD_740_after_pruning", "IPAD_50%"), ("unet_IPAD_940_after_pruning", "IPAD_30%") ],
    "pruned_models_unet_random" : [ ("unet_random_490_after_pruning", "random_75%"), ("unet_random_740_after_pruning", "random_50%"), ("unet_random_940_after_pruning", "random_30%") ],
    "pruned_models_unet_uniform" : [ ("unet_uniform_490_after_pruning", "uniform_75%"), ("unet_uniform_740_after_pruning", "uniform_50%"), ("unet_uniform_940_after_pruning", "uniform_30%") ],
}


training_logs_name = "training_logs_1200_.pkl"

csv_path = 'unet_1200_models_errors.csv'


def add_row(df_np, row):
    
    if df_np is None:
        df_np = np.array([row])
    else:
        df_np = np.vstack([df_np, row])

    return df_np


models_np = None
errors_np = None
is_best_np = None


err_titles = ["val MCDL", "val IoU", "test MCDL", "test IoU"]
is_best_titles = ["is best val MCDL", "is best val IoU", "is best test MCDL", "is best test IoU"]



for base_f, main_fs in folder_structure.items():

    for main_f, model in main_fs:


        try:

            to_tl = osp.join(base_f, main_f, "saved_main", training_logs_name)
            with open(to_tl, 'rb') as f:
                training_logs = pickle.load(f)

            last_error = training_logs.last_error




            # err_names = ["MCDL", "IoU"]
            err_ixs = [0, 3]# we get the avg_loss (MCDL) and the real IoU
            val_err = [float(i) for ix, i in enumerate(last_error[0]) if ix in err_ixs]
            test_err = [float(i) for ix, i in enumerate(last_error[1]) if ix in err_ixs]
            errs = val_err + test_err
            
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
    # MCDL should be minimal, IoU should be maximal
    if i in [0, 2]:
        best_ix = np.argmin(errors)
    else:
        best_ix = np.argmax(errors)
    is_best_np[best_ix, i] = True


collected_data = np.hstack([models_np, errors_np, is_best_np])

df = pd.DataFrame(collected_data, columns=["model"] + err_titles + is_best_titles)

# to csv
df.to_csv(csv_path, index=False)








