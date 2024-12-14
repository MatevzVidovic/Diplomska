

import matplotlib.pyplot as plt
import pickle

import os.path as osp

import argparse

from helper_model_eval_graphs import show_results, resource_graph

argparser = argparse.ArgumentParser()
argparser.add_argument("--mp", type=str, default=None, help="Model path. e.g. ./test_uniform_SegNet_pruning/      Used if msp and smwp are standard and trivial (saved_main and saved_model_wrapper)")

# Only used if --mp is not given
argparser.add_argument("--msp", type=str, help="main save path.  e.g.  ./test_uniform_SegNet_pruning/saved_main/")
argparser.add_argument("--smwp", type=str, help="saved model wrapper path.  e.g.  ./test_uniform_SegNet_pruning/saved/saved_model_wrapper/")

args = argparser.parse_args()

if args.mp is not None:
    main_save_path = osp.join(args.mp, "saved_main")
    saved_model_wrapper_path = osp.join(args.mp, "saved_model_wrapper")
else:
    main_save_path = args.msp
    saved_model_wrapper_path = args.smwp

fig, ax = show_results(main_save_path)
fig, ax, res_dict = resource_graph(main_save_path, saved_model_wrapper_path)

input("Press Enter to continue...")