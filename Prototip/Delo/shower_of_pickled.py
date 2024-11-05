

import pickle


import os

from main import TrainingLogs

inp1 = input("Model folder name (Like UNet): ")
# inp = input("What to add between 'training_logs_' and '.pkl' ?")

if inp1 == "":
    inp1 = "UNet"

training_logs = pickle.load(open(os.path.join(".", inp1, "saved_main", "training_logs.pkl"), "rb"))

print(training_logs)




from ConvResourceCalc import ConvResourceCalc


pruning_train_iter = pickle.load(open(os.path.join(".", inp1, "saved_main", "pruning_train_iter.pkl"), "rb"))

print(pruning_train_iter)

conv_calc: ConvResourceCalc = pruning_train_iter[0][2]
ker_num = conv_calc.get_resource_of_whole_model("kernels_num")
print(f"kernels_num after first pruning: {ker_num}")





initial_resource_calc = pickle.load(open(os.path.join(".", inp1, "saved", "initial_conv_resource_calc.pkl"), "rb"))
initial_resource_calc: ConvResourceCalc
kernel_num = initial_resource_calc.get_resource_of_whole_model("kernels_num")
print(f"kernels_num: {kernel_num}")
