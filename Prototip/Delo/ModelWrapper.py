



import logging
import python_logger.log_helper as py_log

MY_LOGGER = logging.getLogger("prototip")
MY_LOGGER.setLevel(logging.DEBUG)



import os
import torch
import pickle
import pandas as pd
import shutil


from TrainingWrapper import TrainingWrapper
from ConvResourceCalc import ConvResourceCalc
from pruner import pruner
from model_vizualization import model_graph









class ModelWrapper:

    @py_log.log(passed_logger=MY_LOGGER)
    def __init__(self, model_class, model_parameters: dict, dataloader_dict: dict, learning_dict: dict, input_example, save_path):

        self.model_class = model_class
        
        # self.save_path = os.path.join(os.path.dirname(__file__), "saved")
        self.save_path = os.path.join(save_path, "saved_model_wrapper")

        os.makedirs(self.save_path, exist_ok=True)


        if os.path.exists(os.path.join(self.save_path, "previous_model_details.csv")):
            prev_model_details = pd.read_csv(os.path.join(self.save_path, "previous_model_details.csv"))
            self.prev_serial_num = prev_model_details["previous_serial_num"][0]
            self.prev_model_path = prev_model_details["previous_model_path"][0]
            self.prev_pruner_path = prev_model_details["previous_pruner_path"][0]

            # To help with migration after I changed pruner.py right after the training phase 
            # (no prunings had happened so I could just create a new pruner instance)
            if self.prev_pruner_path == "remake_pruner":
                self.prev_pruner_path = None

        else:
            self.prev_serial_num = None
            self.prev_model_path = None
            self.prev_pruner_path = None




        if self.prev_model_path is not None and os.path.exists(self.prev_model_path):
            self.model = torch.load(self.prev_model_path)
        else:
            self.model = self.model_class(**model_parameters)
        









        self.learning_rate = learning_dict["learning_rate"]
        self.optimizer_class = learning_dict["optimizer_class"]

        new_learning_dict = {
            "loss_fn": learning_dict["loss_fn"],
        }

        self.wrap_model = TrainingWrapper(self.model, dataloader_dict, new_learning_dict)

        self.initialize_optimizer()



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



    def initialize_optimizer(self):
        self.wrap_model.initialize_optimizer(self.optimizer_class, self.learning_rate)




    def initialize_pruning(self, get_importance_dict_fn, input_slice_connection_fn, kernel_connection_fn, FLOPS_min_res_percents, weights_min_res_percents):

        self.FLOPS_min_res_percents = FLOPS_min_res_percents
        self.weights_min_res_percents = weights_min_res_percents


        self.conv_tree_ixs = self.resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("Conv2d")
        self.lowest_level_modules = self.resource_calc.get_lowest_level_module_tree_ixs()
        self.batch_norm_ixs = self.resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("BatchNorm2d")

        if self.prev_pruner_path is not None:
            with open(self.prev_pruner_path, "rb") as f:
                self.pruner_instance = pickle.load(f)
                # We need to load this, if the user changed it between the two runs.
                self.pruner_instance.FLOPS_min_resource_percentage_dict = self.FLOPS_min_res_percents.min_resource_percentage_dict
                self.pruner_instance.weights_min_resource_percentage_dict = self.weights_min_res_percents.min_resource_percentage_dict
        else:
            self.pruner_instance = pruner(self.FLOPS_min_res_percents, self.weights_min_res_percents, self.initial_resource_calc, input_slice_connection_fn, kernel_connection_fn, self.conv_tree_ixs, self.batch_norm_ixs, self.lowest_level_modules)

        self.get_importance_dict_fn = get_importance_dict_fn







    @py_log.log(passed_logger=MY_LOGGER)
    def train(self, epochs=1):
        for _ in range(epochs):
            self.wrap_model.train()






    
    



    def epoch_pass(self):
        self.wrap_model.epoch_pass(dataloader_name="train")


    def _prune_one(self):

        importance_dict = self.get_importance_dict_fn(self)

        self.pruner_instance.prune(importance_dict, self.resource_calc, self.wrap_model)

        # This needs to be done so the gradient computation graph is updated.
        # Otherwise it expects gradients of the old shapes.
        self.initialize_optimizer()


    def prune(self, prune_by_original_percent = False, num_of_prunes: int = 1, resource_name = "flops_num", original_percent_to_prune: float = 0.1):

        if not prune_by_original_percent:
            for _ in range(num_of_prunes):
                self._prune_one()


        else:
            initial_resource_value = self.initial_resource_calc.get_resource_of_whole_model(resource_name)
            value_to_prune = initial_resource_value * original_percent_to_prune
            
            starting_resource_value = self.resource_calc.get_resource_of_whole_model(resource_name)
            curr_resource_value = starting_resource_value

            goal_resource_value = starting_resource_value - value_to_prune
            print(f"Goal resource value: {goal_resource_value}")
            
            while curr_resource_value > goal_resource_value:
                self._prune_one()
                self.resource_calc.calculate_resources(self.input_example)
                curr_resource_value = self.resource_calc.get_resource_of_whole_model(resource_name)
                print(f"Current resource value: {curr_resource_value}")
        
        self.resource_calc.calculate_resources(self.input_example)
        return self.resource_calc.get_copy_for_pickle()




    def validation(self):
        val_results = self.wrap_model.validation()
        return val_results


    def test(self):
        test_result = self.wrap_model.test()
        return test_result
    


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

        curr_serial_num = self.prev_serial_num + 1 if self.prev_serial_num is not None else 0

        new_model_path = os.path.join(self.save_path , self.model_class.__name__ + "_" + str(curr_serial_num) + "_" + str_identifier + ".pth")

        torch.save(self.model, new_model_path)

        new_pruner_path = os.path.join(self.save_path, f"pruner_{curr_serial_num}_" + str_identifier + ".pkl")
        with open(new_pruner_path, "wb") as f:
            pickle.dump(self.pruner_instance, f)
        
        
        new_df = pd.DataFrame({"previous_serial_num": [curr_serial_num], "previous_model_path": new_model_path, "previous_pruner_path": new_pruner_path})
        new_df.to_csv(os.path.join(self.save_path, "previous_model_details.csv"))
        
        return (new_model_path, new_pruner_path)
    

    def create_safety_copy_of_existing_models(self, str_identifier: str):

        model_name = str(self.model_class.__name__)

        # get dirname of savepath


        parent_dir_path = os.path.dirname(self.save_path)
        safety_path = os.path.join(parent_dir_path, "safety_copies")
        os.makedirs(safety_path, exist_ok=True)

        safety_copy_dir = os.path.join(safety_path, f"saved_safety_copies_{str_identifier}")

        try:
            # Create the safety copy directory if it doesn't exist
            os.makedirs(safety_copy_dir, exist_ok=True)

            # Iterate through all files in self.save_path
            for filename in os.listdir(self.save_path):
                if filename.startswith(model_name):
                    src_file = os.path.join(self.save_path, filename)
                    dst_file = os.path.join(safety_copy_dir, filename)
                    
                    # Copy the file
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {filename} to {safety_copy_dir}")

        except OSError as e:
            print(f"Error creating safety copies: {e}")



        





