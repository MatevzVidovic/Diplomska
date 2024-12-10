

import pickle

import argparse
import pandas as pd
import numpy as np

import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--inp', type=str, default="", help='Training logs path')

args = parser.parse_args()
inp = args.inp


hardcoded_list = [
                  "DonePrunedModels/SegNet_main_60/saved_main/training_logs_140_.pkl",
                  "DonePrunedModels/SegNet_main_76/saved_main/training_logs_140_.pkl",
                  "DonePrunedModels/SegNet_main_96/saved_main/training_logs_140_.pkl",
                  "DonePrunedModels/SegNet_main_112/saved_main/training_logs_140_.pkl",
]
# hardcoded_list = [
#                   "DonePrunedModels/UNet_main_60/saved_main/training_logs_140_.pkl",
#                   "DonePrunedModels/UNet_main_76/saved_main/training_logs_140_.pkl",
#                   "DonePrunedModels/UNet_main_92/saved_main/training_logs_140_.pkl",
#                   "DonePrunedModels/UNet_main_104/saved_main/training_logs_140_.pkl",
#                   ]

if hardcoded_list == []:
    hardcoded_list = [inp]


for inp in hardcoded_list:




    with open(inp, 'rb') as f:
        training_logs = pickle.load(f)

    last_error = training_logs.last_error


    model = last_error[3]
    split = model.split('/')
    model = split[1] + "/" + split[2]

    val_err = [float(i) for ix, i in enumerate(last_error[0]) if ix not in [1,2]]
    test_err = [float(i) for ix, i in enumerate(last_error[1]) if ix not in [1,2]]

    len_of_error_tuple = len(val_err)

    is_best_val = tuple([False] * len_of_error_tuple)
    is_best_test = tuple([False] * len_of_error_tuple)

    num_of_errors = 2*len_of_error_tuple









    csv_path = 'final_models_errors.csv'

    if osp.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_np = np.array(df)
        df_np[:, 1:1+num_of_errors] = df_np[:, 1:1+num_of_errors].astype(float)
    else:
        df_np = None











    new_row = [model, *val_err, *test_err, *is_best_val, *is_best_test]

    print(new_row)

    if df_np is None:
        df_np = np.array([new_row])
    else:
        df_np = np.vstack([df_np, new_row])



    df_np[:, 1+num_of_errors : 1+num_of_errors+num_of_errors] = False

    for i in range(1, 1+num_of_errors):
        errors = df_np[:, i]
        errors = errors.astype(float)
        # print(errors)
        if i in [1, 4]:
            best_ix = np.argmin(errors)
        else:
            best_ix = np.argmax(errors)
        df_np[best_ix, i+num_of_errors] = True




    df = pd.DataFrame(df_np) #, columns=['Model', 'Val error', 'Test error', "IsBestVal", "IsBestTest"])

    # to csv
    df.to_csv(csv_path, index=False)








