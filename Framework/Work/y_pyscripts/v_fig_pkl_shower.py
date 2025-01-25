
import matplotlib.pyplot as plt
import pickle

import argparse

from conv_resource_calc import ConvResourceCalc

argparser = argparse.ArgumentParser()
argparser.add_argument("--inp", type=str, required=True, help="./test_uniform_SegNet_pruning/saved_main/8_show_results.pkl")
inp = argparser.parse_args().inp

with open(inp, "rb") as f:
    data = pickle.load(f)



print(data)


if data is ConvResourceCalc:

    conv_calc: ConvResourceCalc = data.pruning_logs[0][2]
    ker_num = conv_calc.get_resource_of_whole_model("kernels_num")
    print(f"kernels_num after first pruning: {ker_num}")

data.show()

input("Press Enter to continue...")