

import pickle


import os


import matplotlib.pyplot as plt


import argparse

from ConvResourceCalc import ConvResourceCalc



argparser = argparse.ArgumentParser()
argparser.add_argument("--inp", type=str, required=True, help="this can be path to training_logs or conv_res_calc (curr or initial)")
inp = argparser.parse_args().inp

with open(inp, "rb") as f:
    data = pickle.load(f)



print(data)


if data is ConvResourceCalc:

    conv_calc: ConvResourceCalc = data.pruning_logs[0][2]
    ker_num = conv_calc.get_resource_of_whole_model("kernels_num")
    print(f"kernels_num after first pruning: {ker_num}")


# fig, ax = plt.subplots()
# plt.plot([0,3,2,3,5,7])


# os.makedirs(os.path.join(".", "for_fig_shower"), exist_ok=True)

# fig.savefig(os.path.join(".", "for_fig_shower", "fig_shower_of_pkl.png"))
# with open(os.path.join(".", "for_fig_shower", "fig_shower_of_pkl.pkl"), "wb") as f:
#     pickle.dump(fig, f)
