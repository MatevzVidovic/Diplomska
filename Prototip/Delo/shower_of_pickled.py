

import pickle




from main import TrainingLogs

inp = input("What to add between 'training_logs_' and '.pkl' ?")

training_logs = pickle.load(open(("./saved_main/training_logs_" + inp + ".pkl"), "rb"))

print(training_logs)




from ConvResourceCalc import ConvResourceCalc


pruning_train_iter = pickle.load(open(("./saved_main/pruning_train_iter.pkl"), "rb"))

print(pruning_train_iter)

conv_calc: ConvResourceCalc = pruning_train_iter[0][2]
ker_num = conv_calc.get_resource_of_whole_model("kernels_num")
print(f"kernels_num after first pruning: {ker_num}")





initial_resource_calc = pickle.load(open(("./saved/initial_conv_resource_calc.pkl"), "rb"))
initial_resource_calc: ConvResourceCalc
kernel_num = initial_resource_calc.get_resource_of_whole_model("kernels_num")
print(f"kernels_num: {kernel_num}")
