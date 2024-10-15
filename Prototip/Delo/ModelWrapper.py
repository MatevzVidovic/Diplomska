

import torch
from torch import nn

import pickle



from unet import UNet

from TrainingWrapper import TrainingWrapper

from ConvResourceCalc import ConvResourceCalc

from pruner import pruner

from min_resource_percentage import min_resource_percentage

from model_vizualization import model_graph


import pandas as pd





# Logging preparation:

import logging
import sys
import os

# Assuming the submodule is located at 'python_logger'
submodule_path = os.path.join(os.path.dirname(__file__), 'python_logger')
sys.path.insert(0, submodule_path)

import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string instead of __name__. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)
# def file_handler_setup(logger, path_to_python_logger_folder, add_stdout_stream: bool = False)













class ModelWrapper:

    @py_log.log(passed_logger=MY_LOGGER)
    def __init__(self, model_class, model_parameters: dict, dataloader_dict: dict, learning_dict: dict, input_example):

        self.tree_ix_2_hook_handle = None

        self.model_class = model_class
        
        self.save_path = os.path.join(os.path.dirname(__file__), "saved")

        os.makedirs(self.save_path, exist_ok=True)


        if os.path.exists(os.path.join(self.save_path, "previous_model_details.csv")):
            prev_model_details = pd.read_csv(os.path.join(self.save_path, "previous_model_details.csv"))
            self.prev_serial_num = prev_model_details["previous_serial_num"][0]
            self.prev_model_path = prev_model_details["previous_model_path"][0]
            self.prev_pruner_path = prev_model_details["previous_pruner_path"][0]
        else:
            self.prev_serial_num = None
            self.prev_model_path = None
            self.prev_pruner_path = None




        if self.prev_model_path is not None and os.path.exists(self.prev_model_path):
            self.model = torch.load(self.prev_model_path)
        else:
            self.model = self.model_class(**model_parameters)
        









        learning_rate = learning_dict["learning_rate"]
        optimizer_class = learning_dict["optimizer_class"]

        optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

        new_learning_dict = {
            "optimizer": optimizer,
            "loss_fn": learning_dict["loss_fn"],
        }

        self.wrap_model = TrainingWrapper(self.model, dataloader_dict, new_learning_dict)





        self.resource_calc = ConvResourceCalc(self.wrap_model)

        self.input_example = input_example
        self.resource_calc.calculate_resources(self.input_example)

        initial_resource_calc_path = os.path.join(self.save_path, "initial_conv_resource_calc.pkl")
        if os.path.exists(os.path.join(initial_resource_calc_path)):
            with open(initial_resource_calc_path, "rb") as f:
                self.initial_resource_calc = pickle.load(f)
        else:
            # pickle resource_dict
            with open(initial_resource_calc_path, "wb") as f:
                self.initial_resource_calc = self.resource_calc.get_copy_for_pickle()
                pickle.dump(self.initial_resource_calc, f)



    def get_tree_ix_2_name(self):
        return self.resource_calc.module_tree_ix_2_name




    def initialize_pruning(self, importance_fn, connection_fn, inextricable_connection_fn, FLOPS_min_res_percents, weights_min_res_percents):

        if self.tree_ix_2_hook_handle is not None:
            self.remove_hooks()

        self.FLOPS_min_res_percents = FLOPS_min_res_percents
        self.weights_min_res_percents = weights_min_res_percents


        self.conv_tree_ixs = self.resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("Conv2d")
        self.lowest_level_modules = self.resource_calc.get_lowest_level_module_tree_ixs()
        self.batch_norm_ixs = self.resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("BatchNorm2d")

        if self.prev_pruner_path is not None:
            with open(self.prev_pruner_path, "rb") as f:
                self.pruner_instance = pickle.load(f)
                # We need to load this, if the user changed it between the two runs.
                self.pruner_instance.FLOPS_min_resource_percentage_dict = self.FLOPS_min_res_percents
                self.pruner_instance.weights_min_resource_percentage_dict = self.weights_min_res_percents
        else:
            self.pruner_instance = pruner(self.FLOPS_min_res_percents, self.weights_min_res_percents, self.initial_resource_calc, connection_fn, inextricable_connection_fn, self.conv_tree_ixs, self.batch_norm_ixs, self.lowest_level_modules)


        self.importance_fn = importance_fn




        self.activations = {}
        self.set_activations_hooks(self.activations, self.resource_calc, self.conv_tree_ixs)





    def set_activations_hooks(self, activations: dict, resource_calc: ConvResourceCalc, tree_ixs: list):
            
        
        def get_activation(tree_ix):
            
            def hook(module, input, output):
                if tree_ix not in activations:
                    activations[tree_ix] = []
                # activations[tree_ix].append(output.detach())
                activations[tree_ix] = [output.detach()]

            return hook

        tree_ix_2_hook_handle = {}
        for tree_ix in tree_ixs:
            module = resource_calc.module_tree_ix_2_module_itself[tree_ix]
            tree_ix_2_hook_handle[tree_ix] = module.register_forward_hook(get_activation(tree_ix))
        
        self.tree_ix_2_hook_handle = tree_ix_2_hook_handle
        

    

    def remove_hooks(self):
        for hook_handle in self.tree_ix_2_hook_handle.values():
            hook_handle.remove()
        
        self.tree_ix_2_hook_handle = None


    @py_log.log(passed_logger=MY_LOGGER)
    def train(self, epochs=1):
        for _ in range(epochs):
            self.wrap_model.train()


    def validation(self):
        # This is necessary, so we don't medle with the activations.
        self.remove_hooks()
        val_results = self.wrap_model.validation()
        self.set_activations_hooks(self.activations, self.resource_calc, self.conv_tree_ixs)
        return val_results


    # set_activations_hooks(activations, conv_modules_tree_ixs, resource_calc)
    def test(self):
        # This is necessary, so we don't medle with the activations.
        self.remove_hooks()
        test_result = self.wrap_model.test()
        self.set_activations_hooks(self.activations, self.resource_calc, self.conv_tree_ixs)
        return test_result


    def reset_activations(self):

        self.remove_hooks()
        self.activations = {}
        self.set_activations_hooks(self.activations, self.resource_calc, self.conv_tree_ixs)

    def prune(self, num_of_prunes: int = 1):

        for _ in range(num_of_prunes):
            # pruner needs the current state of model resources to know which modules shouldn't be pruned anymore
            self.resource_calc.calculate_resources(self.input_example)
            importance_dict = self.importance_fn(self.activations, self.conv_tree_ixs)
            self.pruner_instance.prune(importance_dict, self.resource_calc, self.wrap_model)
        
        self.resource_calc.calculate_resources(self.input_example)
        

        # I don't know why
        # self.activations.clear()
        # doesn't work. But for some reason, on the second pass of pruning, self.activations is empty.
        # Maybe the hooks are somehow hellbound to the previous activations dictionary.
        # I do not know.
        self.reset_activations()
        
        return
    


    def model_graph(self):
        model_graph(self.resource_calc, self.initial_resource_calc, self.pruner_instance)
        

    def print_logs(self):
        
        conv = self.pruner_instance.pruning_logs["conv"]
        batch_norm = self.pruner_instance.pruning_logs["batch_norm"]
        following = self.pruner_instance.pruning_logs["following"]
        print("conv_ix/batch_norm_ix , real_(kernel/input_slice)_ix, initial_-||-")
        print("Conv ||  BatchNorm2d ||  Following")
        for i in range(len(conv)):
            print(f"{conv[i]}, || {batch_norm[i]} || {following[i]}")
            print("\n")

    @py_log.log(passed_logger=MY_LOGGER)
    def save(self, str_identifier: str = ""):

        self.remove_hooks()

        curr_serial_num = self.prev_serial_num + 1 if self.prev_serial_num is not None else 0

        new_model_path = os.path.join(self.save_path , self.model_class.__name__ + "_" + str(curr_serial_num) + "_" + str_identifier + ".pth")

        torch.save(self.model, new_model_path)

        new_pruner_path = os.path.join(self.save_path, f"pruner_{curr_serial_num}_" + str_identifier + ".pkl")
        with open(new_pruner_path, "wb") as f:
            pickle.dump(self.pruner_instance, f)
        
        
        new_df = pd.DataFrame({"previous_serial_num": [curr_serial_num], "previous_model_path": new_model_path, "previous_pruner_path": new_pruner_path})
        new_df.to_csv(os.path.join(self.save_path, "previous_model_details.csv"))

        self.set_activations_hooks(self.activations, self.resource_calc, self.conv_tree_ixs)
        
        return (new_model_path, new_pruner_path)






