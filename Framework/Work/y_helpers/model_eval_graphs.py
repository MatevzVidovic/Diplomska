
import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open(f"{osp.join('pylog_configs', 'active_logging_config.txt')}", 'r') as f:
    cfg_name = f.read()
    yaml_path = osp.join('pylog_configs', cfg_name)

log_config_path = osp.join(yaml_path)
do_log = False
if osp.exists(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
        file_log_setting = config.get(osp.basename(__file__), False)
        if file_log_setting:
            do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)




import pickle
import argparse

import y_helpers.shared as shared
if not shared.PLT_SHOW: # For more info, see shared.py
    import matplotlib
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import y_helpers.json_handler as jh

from y_framework.log_handlers import TrainingLogs, PruningLogs






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
        

        tl_name = jh.load(tl_j_path)["prev_training_logs"]
        pl_name = jh.load(pl_j_path)["prev_pruning_logs"]

        training_logs_path = osp.join(main_save_path, "training_logs", tl_name)
        pruning_logs_path = osp.join(main_save_path, "pruning_logs", pl_name)

        training_logs: TrainingLogs = pickle.load(open(training_logs_path, "rb"))
        pruning_logs = pickle.load(open(pruning_logs_path, "rb"))
        
        
        
        pruning_moments = [i[0] for i in pruning_logs.pruning_logs]


        
        print(training_logs)
        print(pruning_logs)

        model_logs = training_logs.logs + training_logs.deleted_models_logs
        model_logs = [log for log in model_logs if log is not None]
        model_logs = [log for log in model_logs if not log["is_not_automatic"]]
        model_logs.sort(key = lambda x: x["train_iter"])

        val_errors = [error["val_err"] for error in model_logs]
        test_errors = [error["test_err"] for error in model_logs]


        val_losses = [error["loss"] for error in val_errors]
        # error[1] is approx IoU
        val_1minus_F1s = [1 - error["F1"] for error in val_errors]
        val_1minus_IoUs = [1 - error["IoU"] for error in val_errors]

        

        test_losses = [error["loss"] for error in test_errors]
        # error[1] is approx IoU
        test_1minus_F1s = [1 - error["F1"] for error in test_errors]
        test_1minus_IoUs = [1 - error["IoU"] for error in test_errors]


        # make a plot of the val errors and test errors over time
        # make vertical lines for when pruning happened (in pruning moments)

        # make new plt
        fig, ax = plt.subplots()


        plt.plot(val_losses, label="avg loss (validation)")
        plt.plot(test_losses, label="avg loss entropy (test)")
        plt.plot(val_1minus_F1s, label="1-F1 (validation)")
        plt.plot(test_1minus_F1s, label="1-F1 (test)")
        plt.plot(val_1minus_IoUs, label="1-IoU (validation)")
        plt.plot(test_1minus_IoUs, label="1-IoU (test)")

        # plt.ylim(0, 1)

        plt.ylabel("Error")
        plt.xlabel("training iterations")

        for moment in pruning_moments:
            plt.axvline(x=moment, color="blue", linestyle="--")
            




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
        if len(pruning_moments) >= 1:
            labels.append("Blue line: 1 pruning")

        # Update the legend with the new entries
        plt.legend(handles=handles, labels=labels)

        if shared.PLT_SHOW:
            plt.show(block=False)

        return fig, ax
    
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e

    # except Exception as e:
    #     # Print the exception message
    #     print("An exception occurred in show_results():", e)
    #     # Print the type of the exception
    #     print("Exception type:", type(e).__name__)
    #     print("Continuing operation.")
    #     return None, None



def resource_graph(main_save_path, saved_model_wrapper_path):

    try:

        pl_j_path = osp.join(main_save_path, "prev_pruning_logs_name.json")
        pl_name = jh.load(pl_j_path)["prev_pruning_logs"]
        pruning_logs_path = osp.join(main_save_path, "pruning_logs", pl_name)

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

        if shared.PLT_SHOW:
            plt.show(block=False)

        to_pkl = []
        for i in range(len(pruning_moments)):
            to_pkl.append((pruning_moments[i], flops_percents[i], weights_percents[i]))

        res_dict = {"(pruning_moment, flops_percent, weights_percent)": to_pkl}

        return fig, ax, res_dict

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


    # except Exception as e:

    #     # raise e

    #     # Print the exception message
    #     print("An exception occurred in resource_graph():", e)
    #     # Print the type of the exception
    #     print("Exception type:", type(e).__name__)
    #     print("Continuing operation.")
    #     return None, None, None




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










