


import os.path as osp
import pickle
import json_handler as jh


import matplotlib.pyplot as plt

import argparse


# !!!!!!!!!!!!!!!!!!!!!!!!!!!
# There is a main() at the bottom!!!!!!!!



def show_results(main_save_path):

    try:

        # inp = input("Enter what comes between pruner_ and .pkl: ")
        # pruner_instance = pickle.load(open("./saved/pruner_" + inp + ".pkl", "rb"))
        # pruning_logs_dict = pruner_instance.pruning_logs


        tl_j_path = osp.join(main_save_path, "prev_training_logs_name.json")
        pl_j_path = osp.join(main_save_path, "prev_pruning_logs_name.json")

        if not osp.exists(tl_j_path) or not osp.exists(pl_j_path):
            print("The logs don't exist.")
            return
        

        tl_name = jh.load(tl_j_path)[f"prev_training_logs"]
        pl_name = jh.load(pl_j_path)[f"prev_pruning_logs"]

        training_logs_path = osp.join(main_save_path, tl_name)
        pruning_logs_path = osp.join(main_save_path, pl_name)

        training_logs = pickle.load(open(training_logs_path, "rb"))
        pruning_logs = pickle.load(open(pruning_logs_path, "rb"))
        
        
        
        pruning_moments = [i[0] for i in pruning_logs.pruning_logs]


        
        print(training_logs)
        print(pruning_logs)

        model_errors = training_logs.errors + training_logs.deleted_models_errors
        model_errors = [log for log in model_errors if log is not None]
        model_errors = [log for log in model_errors if not log[5]]
        model_errors.sort(key = lambda x: x[2])

        val_errors = [error[0] for error in model_errors]
        test_errors = [error[1] for error in model_errors]


        val_losses = [error[0] for error in val_errors]
        # error[1] is approx IoU
        val_1minus_F1s = [1 - error[2] for error in val_errors]
        val_1minus_IoUs = [1 - error[3] for error in val_errors]

        

        test_losses = [error[1] for error in test_errors]
        # error[1] is approx IoU
        test_1minus_F1s = [1 - error[2] for error in test_errors]
        test_1minus_IoUs = [1 - error[3] for error in test_errors]


        # make a plot of the val errors and test errors over time
        # make vertical lines for when pruning happened (in pruning moments)

        # make new plt
        fig, ax = plt.subplots()


        plt.plot(val_losses, label="avg cross entropy (validation)")
        plt.plot(test_losses, label="avg cross entropy (test)")
        plt.plot(val_1minus_F1s, label="1-F1 (validation)")
        plt.plot(test_1minus_F1s, label="1-F1 (test)")
        plt.plot(val_1minus_IoUs, label="1-IoU (validation)")
        plt.plot(test_1minus_IoUs, label="1-IoU (test)")

        # plt.ylim(0, 1)

        plt.ylabel("Error")
        plt.xlabel("training iterations")

        for moment in pruning_moments:
            plt.axvline(x=moment, color="blue", linestyle="--")
            




        from matplotlib.patches import Patch

        # Create the initial legend
        plt.legend()

        # Get current handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dummy handle for the arbitrary string
        extra = Patch(facecolor='none', edgecolor='none', label='Arbitrary String')

        # Append the new handle and label
        # handles.append(extra)
        # labels.append("Green line: 10 filters pruned")
        handles.append(extra)
        labels.append("Blue line: 1 pruning")

        # Update the legend with the new entries
        plt.legend(handles=handles, labels=labels)

        plt.show(block=False)

        return fig, ax

    except Exception as e:
        # Print the exception message
        print("An exception occurred in show_results():", e)
        # Print the type of the exception
        print("Exception type:", type(e).__name__)
        print("Continuing operation.")
        return None, None



def resource_graph(main_save_path, saved_model_wrapper_path):

    try:

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

        fig, ax = plt.subplots()
        plt.plot(pruning_moments, flops_percents, label="FLOPs")
        plt.plot(pruning_moments, weights_percents, label="Weights")

        plt.ylabel("Resource percent")
        plt.xlabel("training iterations")

        plt.legend()


        plt.show(block=False)

        to_pkl = []
        for i in range(len(pruning_moments)):
            to_pkl.append((pruning_moments[i], flops_percents[i], weights_percents[i]))

        res_dict = {"(pruning_moment, flops_percent, weights_percent)": to_pkl}

        return fig, ax, res_dict



    except Exception as e:

        # raise e

        # Print the exception message
        print("An exception occurred in resource_graph():", e)
        # Print the type of the exception
        print("Exception type:", type(e).__name__)
        print("Continuing operation.")
        return None, None, None




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Show the results of the training and pruning.")
    parser.add_argument("--msp", type=str, help="main_save_path. The path to the main save folder of the model. e.g. ./scp_1/SegNet_random_main_60/saved_main")
    parser.add_argument("--smp", type=str, help="saved_model_wrapper_path. The path to the saved_model_wrapper folder. e.g. ./scp_1/SegNet_random_main_60/saved_model_wrapper")

    args = parser.parse_args()

    main_save_path = args.msp
    saved_model_wrapper_path = args.smp

    fig, ax = show_results(main_save_path)
    fig, ax, res_dict = resource_graph(main_save_path, saved_model_wrapper_path)

    input("Press enter to close the program.")










