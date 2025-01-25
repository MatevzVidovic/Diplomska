




import argparse

import os.path as osp

import y_helpers.helper_json_handler as jh


import pickle


import argparse




# In a multi-purpose sbatch:
#!/bin/bash

#SBATCH --job-name=mp
#SBATCH --time=1-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G


# srun --output=x_percents_unet_prune_IPAD.txt  python3 v_find_closest_pruned_percents.py -p unet_prune_IPAD




argparser = argparse.ArgumentParser(description='Porter Stemmer')
argparser.add_argument('-p', '--model_path', required=True, type=str)

args = argparser.parse_args()
model_path = args.model_path

main_save_path = osp.join(model_path, "saved_main")
saved_model_wrapper_path = osp.join(model_path, "saved_model_wrapper")


pl_j_path = osp.join(main_save_path, "prev_pruning_logs_name.json")
pl_name = jh.load(pl_j_path)[f"prev_pruning_logs"]

pruning_logs_path = osp.join(main_save_path, pl_name)
irc_path = osp.join(saved_model_wrapper_path, "initial_conv_resource_calc.pkl")

pruning_logs = pickle.load(open(pruning_logs_path, "rb"))
initial_resource_calc = pickle.load(open(irc_path, "rb"))

initial_flops = initial_resource_calc.get_resource_of_whole_model("flops_num")
initial_weights = initial_resource_calc.get_resource_of_whole_model("weights_num")

pruning_moments = [i[0] for i in pruning_logs.pruning_logs]
resource_calcs = [i[2] for i in pruning_logs.pruning_logs]
flops = [i.get_resource_of_whole_model("flops_num") for i in resource_calcs]
weights = [i.get_resource_of_whole_model("weights_num") for i in resource_calcs]

flops_percents = [(i) / initial_flops for i in flops]
weights_percents = [(i) / initial_weights for i in weights]


def find_closest_index(lst, target):
    """Find the index of the closest value to the target in the list."""
    min_ix = 0
    min_val = abs(lst[0] - target)
    for i in range(1, len(lst)):
        if abs(lst[i] - target) < min_val:
            min_val = abs(lst[i] - target)
            min_ix = i
    return min_ix

def print_percentages(pruning_moments, percents, label):
    """Print the percentages and corresponding pruning moments."""
    print(f"{label}:")
    possible_percents = [ 1-0.01*i for i in range(100)]
    for percent in possible_percents:
        closest_index = find_closest_index(percents, percent)
        print(f"{percent:.2f}: {pruning_moments[closest_index]}, {percents[closest_index]:.5f}")


print_percentages(pruning_moments, flops_percents, "FLOPs")
print_percentages(pruning_moments, weights_percents, "Weights")
