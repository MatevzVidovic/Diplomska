
import matplotlib.pyplot as plt
import pickle

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--inp", type=str, required=True, help="./test_uniform_SegNet_pruning/saved_main/8_show_results.pkl")
inp = argparser.parse_args().inp

with open(inp, "rb") as f:
    data = pickle.load(f)

data.show()

input("Press Enter to continue...")