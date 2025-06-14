



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




import os
# import os.path as osp
import pickle
import shutil


import torch


import y_helpers.json_handler as jh
from y_helpers.model_vizualization import model_graph

from y_framework.params_dataclasses import *
from y_framework.training_wrapper import TrainingWrapper
from pruning_framework.conv_resource_calc import ConvResourceCalc
from pruning_framework.pruner import Pruner









class ModelWrapper:

    @py_log.autolog(passed_logger=MY_LOGGER)
    def __init__(self, model_wrapper_params: ModelWrapperParams, model_parameters: dict, dataloader_dict: dict, training_wrapper_params: dict):
        self.init_dict = {
            "model_wrapper_params": model_wrapper_params,
            "model_parameters": model_parameters,
            "dataloader_dict": dataloader_dict,
            "training_wrapper_params": training_wrapper_params
        }
        self.init()

    @py_log.autolog(passed_logger=MY_LOGGER)
    def init(self):

        try:

            model_wrapper_params = self.init_dict["model_wrapper_params"]
            model_parameters = self.init_dict["model_parameters"]
            dataloader_dict = self.init_dict["dataloader_dict"]
            training_wrapper_params = self.init_dict["training_wrapper_params"]
            
            self.params: ModelWrapperParams = model_wrapper_params
                
            self.model_class = self.params.model_class
            
            # self.save_path = osp.join(osp.dirname(__file__), "saved")
            save_path = self.params.save_path
            self.save_path = osp.join(save_path, "saved_model_wrapper")
            os.makedirs(self.save_path, exist_ok=True)





            j_path = osp.join(self.save_path, "previous_model_details.json")
            j_dict = jh.load(j_path)

            if j_dict is None:
                self.prev_model_path = None
                self.prev_pruner_path = None
            else:

                model_filename = j_dict["previous_model_filename"]
                pruner_filename = j_dict["previous_pruner_filename"]
                
                self.prev_model_path = osp.join(self.save_path, "models", model_filename)
                
                if pruner_filename is None:
                    self.prev_pruner_path = None
                else:
                    self.prev_pruner_path = osp.join(self.save_path, "pruners", pruner_filename)
            


                
            # To help with migration after I changed pruner.py right after the training phase 
            # (no prunings had happened so I could just create a new pruner instance)
            if self.prev_pruner_path == "remake_pruner":
                self.prev_pruner_path = None
            
            # This is None until we initialize pruning
            self.pruner_instance = None



            


            device = self.params.device

            if self.prev_model_path is not None and osp.exists(self.prev_model_path):
                print("Loaded model path: ", self.prev_model_path)
                # you can specify the map_location parameter to map the model to the CPU. This tells PyTorch to load the model onto the CPU, even if it was originally saved on a GPU.
                # Otherwise: RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.
                self.model = torch.load(self.prev_model_path, map_location=torch.device(device))
            else:
                self.model = self.model_class(**model_parameters)
                print("Created new model instance.")
            













            self.learning_rate = self.params.learning_rate
            self.optimizer_class = self.params.optimizer_class


            self.training_wrapper: TrainingWrapper = TrainingWrapper(training_wrapper_params, self.model, dataloader_dict)

            self.initialize_optimizer()



            if self.params.is_resource_calc_ready:

                self.resource_calc = ConvResourceCalc(self.training_wrapper)

                self.input_example = self.params.input_example
                self.resource_calc.calculate_resources(self.input_example)

                # these two are set here because they are useful in some other tasks in the main script - such as when setting disallowed conv layers
                # or in functions that take model_wrapper as an argument.
                # Just ctrl f  for model_wrapper.conv_tree_ixs
                self.conv_tree_ixs = self.resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("Conv2d")
                self.lowest_level_modules = self.resource_calc.get_lowest_level_module_tree_ixs()



                initial_resource_calc_path = osp.join(self.save_path, "initial_conv_resource_calc.pkl")
                if osp.exists(initial_resource_calc_path):
                    with open(initial_resource_calc_path, "rb") as f:
                        self.initial_resource_calc = pickle.load(f)
                else:
                    # pickle resource_dict
                    with open(initial_resource_calc_path, "wb") as f:
                        self.initial_resource_calc = self.resource_calc.get_copy_for_pickle()
                        pickle.dump(self.initial_resource_calc, f)



        
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e



    @py_log.autolog(passed_logger=MY_LOGGER)
    def get_tree_ix_2_name(self):
        return self.resource_calc.module_tree_ix_2_name


    @py_log.autolog(passed_logger=MY_LOGGER)
    def initialize_optimizer(self):
        self.training_wrapper.initialize_optimizer(self.optimizer_class, self.learning_rate)



    @py_log.autolog(passed_logger=MY_LOGGER)
    def initialize_pruning(self, get_importance_dict_fn, input_slice_connection_fn, kernel_connection_fn, pruning_disallowments, other_zeroth_dim_LLM_ixs=None):

        if other_zeroth_dim_LLM_ixs is None:
            other_zeroth_dim_LLM_ixs = []



        batchnorm_ixs = self.resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("BatchNorm2d")
        other_zeroth_dim_ixs = []
        # print(other_zeroth_dim_LLM_ixs)
        for ix in other_zeroth_dim_LLM_ixs:
            curr = self.lowest_level_modules[ix]
            # print("curr", curr)
            other_zeroth_dim_ixs.append(curr)
        other_zeroth_dim_ixs = batchnorm_ixs + other_zeroth_dim_ixs
        # print("other_zeroth_dim_ixs", other_zeroth_dim_ixs)
        

        if self.prev_pruner_path is not None:
            with open(self.prev_pruner_path, "rb") as f:
                self.pruner_instance = pickle.load(f)
                # We need to load this, if the user changed it between the two runs.
                self.pruner_instance.pruning_disallowments = pruning_disallowments
        else:
            self.pruner_instance = Pruner(pruning_disallowments, self.initial_resource_calc, input_slice_connection_fn, kernel_connection_fn, self.conv_tree_ixs, other_zeroth_dim_ixs, self.lowest_level_modules, self.input_example)

        self.get_importance_dict_fn = get_importance_dict_fn







    @py_log.autolog(passed_logger=MY_LOGGER)
    def train(self, epochs=1):
        returnings = []
        for _ in range(epochs):
            returned = self.training_wrapper.train()
            returnings.append(returned)
        return returnings






    
    


    @py_log.autolog(passed_logger=MY_LOGGER)
    def epoch_pass(self, dataloader_name="train"):
        self.training_wrapper.epoch_pass(dataloader_name=dataloader_name)



    def _prune_n_kernels(self, n, resource_limitation_dict=None):

        importance_dict = self.get_importance_dict_fn(self) # this does epoch pass in it

        # this already does resource_calc.calculate_resources(self.input_example)
        are_there_more_to_prune_in_the_future = self.pruner_instance.prune(n, importance_dict, self.resource_calc, resource_limitation_dict)

        # This needs to be done so the gradient computation graph is updated.
        # Otherwise it expects gradients of the old shapes.
        self.initialize_optimizer()

        return are_there_more_to_prune_in_the_future
    

    def get_resource_info(self, resource_name):
        
        initial_resource_value = self.initial_resource_calc.get_resource_of_whole_model(resource_name)
        curr_resource_value = self.resource_calc.get_resource_of_whole_model(resource_name)
        curr_percent = curr_resource_value / initial_resource_value

        return {"initial": initial_resource_value, "current": curr_resource_value, "percentage": curr_percent}

    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune(self, prune_n_kernels_at_once=1, prune_by_original_percent = False, num_of_prunes: int = 1, resource_name = "flops_num", original_proportion_to_prune: float = 0.1):

        # making sure it is correct
        self.resource_calc.calculate_resources(self.input_example)

        if not prune_by_original_percent:
            num_pruned = 0
            while num_pruned < num_of_prunes:
                num_to_prune = min(num_of_prunes - num_pruned, prune_n_kernels_at_once)
                are_there_more_to_prune_in_the_future = self._prune_n_kernels(num_to_prune)
                num_pruned += num_to_prune
                if not are_there_more_to_prune_in_the_future:
                    break


        else:
            
            """
            We don't just prune by this percent, because that get's us bad results.
            Every time we prune, we prune e.g. 1 percent. Because of pnkao we overshoot by a little. So next time, if we prune by 1 percent again, we will overshoot by a little again, and the overshoots compound.
            So we will instead prune in this way: get in which bracket of this percent we are so far (eg, we have 79.9 percent of original weights), then we will prune to 79 percent and pnkao will overshoot a little.
            

            Old code:
            initial_resource_value = self.initial_resource_calc.get_resource_of_whole_model(resource_name)
            value_to_prune = initial_resource_value * original_proportion_to_prune
            
            starting_resource_value = self.resource_calc.get_resource_of_whole_model(resource_name)
            curr_resource_value = starting_resource_value

            goal_resource_value = starting_resource_value - value_to_prune
            print(f"Goal resource value: {goal_resource_value}")
            
            """



            initial_resource_value = self.initial_resource_calc.get_resource_of_whole_model(resource_name)

            bracket_size = initial_resource_value * original_proportion_to_prune
            
            starting_resource_value = self.resource_calc.get_resource_of_whole_model(resource_name)
            starting_bracket = starting_resource_value / bracket_size
            lower_lim_starting_bracket = int(starting_bracket)
            residual_part = starting_bracket - lower_lim_starting_bracket

            # if we are too close to the next bracket, that is somewhat suspicious, so just to prevent problems like:
            # we are at bracket 79.01 so we go prune to 79 but we again just stop and end up at 79.01, so we never prune anything.
            # Not that this has happened, just that it can happen in general.
            # Pnkao should never overshoot by 0.9 o a bracket size, so this is a safe assumption. If it does overshoot by that much, it's a much bigger problem than just this skipping of a bracket,
            # and it should obviously be made much lower.

            if residual_part < 0.1:
                lower_lim_starting_bracket -= 1
            
            goal_resource_value = bracket_size * lower_lim_starting_bracket
            


            curr_resource_value = starting_resource_value



            resource_limitation_dict = {
                "resource_name": resource_name,
                "goal_resource_value": goal_resource_value,
            }
            
            print(f"Goal resource value: {goal_resource_value}")

            goal_resource_value_percentage = goal_resource_value / initial_resource_value

            print(f"Goal resource value percentage: {goal_resource_value_percentage}")





            info_dict = self.get_resource_info("flops_num")
            network_flops_percentage = info_dict["percentage"]

            info_dict = self.get_resource_info("weights_num")
            network_weights_percentage = info_dict["percentage"]

            print(f"Curr network_flops_percentage: {network_flops_percentage}")
            print(f"Curr network_weights_percentage: {network_weights_percentage}")
            


            while curr_resource_value > goal_resource_value:
                are_there_more_to_prune_in_the_future = self._prune_n_kernels(prune_n_kernels_at_once, resource_limitation_dict) # this already does resource_calc.calculate_resources(self.input_example) 
                info_dict = self.get_resource_info("flops_num")
                curr_resource_value, curr_percentage = info_dict["current"], info_dict["percentage"]
                print(f"Current resource value: {curr_resource_value}")
                print(f"Current resource value percentage: {curr_percentage}")
                if not are_there_more_to_prune_in_the_future:
                    break
        
        self.resource_calc.calculate_resources(self.input_example) # just in case - but should have been done in the prune function already
        return self.resource_calc.get_copy_for_pickle(), are_there_more_to_prune_in_the_future



    @py_log.autolog(passed_logger=MY_LOGGER)
    def validation(self):
        val_results = self.training_wrapper.validation()
        return val_results

    @py_log.autolog(passed_logger=MY_LOGGER)
    def test(self):
        test_result = self.training_wrapper.test()
        return test_result
    

    def model_graph(self, model_graph_params_dict):
        list_of_fig_ax_tuples = model_graph(self.resource_calc, self.initial_resource_calc, self.pruner_instance, **model_graph_params_dict)
        return list_of_fig_ax_tuples

    def print_logs(self):
        
        conv = self.pruner_instance.pruning_logs["conv"]
        batch_norm = self.pruner_instance.pruning_logs["batch_norm"]
        following = self.pruner_instance.pruning_logs["following"]
        print("conv_ix/batch_norm_ix , real_(kernel/input_slice)_ix, initial_-||-")
        print("Conv ||  BatchNorm2d ||  Following")
        for ix, curr_conv in enumerate(conv):
            print(f"{curr_conv}, || {batch_norm[ix]} || {following[ix]}")
            print("\n")

    @py_log.autolog(passed_logger=MY_LOGGER)
    def save(self, str_identifier: str = ""):

        model_filename = self.model_class.__name__ + "_" + str_identifier + ".pth"

        os.makedirs(osp.join(self.save_path, "models"), exist_ok=True)
        new_model_path = osp.join(self.save_path, "models", model_filename)

        torch.save(self.model, new_model_path)

        if self.pruner_instance is not None:
            pruner_filename = "pruner_" + str_identifier + ".pkl"
            os.makedirs(osp.join(self.save_path, "pruners"), exist_ok=True)
            new_pruner_path = osp.join(self.save_path, "pruners", pruner_filename)
            with open(new_pruner_path, "wb") as f:
                pickle.dump(self.pruner_instance, f)
        else:
            pruner_filename = None
        

        j_path = osp.join(self.save_path, "previous_model_details.json")
        j_dict = { 
            "previous_model_filename": model_filename,
            "previous_pruner_filename": pruner_filename
        }
        jh.dump(j_path, j_dict)
        
        return (model_filename, pruner_filename)
    
    @py_log.autolog(passed_logger=MY_LOGGER)
    def create_safety_copy_of_existing_models(self, str_identifier: str):

        model_name = str(self.model_class.__name__)

        # get dirname of savepath


        parent_dir_path = osp.dirname(self.save_path)

        safety_path = osp.join(parent_dir_path, "safety_copies")
        os.makedirs(safety_path, exist_ok=True)

        safety_copy_dir = osp.join(safety_path, "actual_safety_copies")
        os.makedirs(safety_copy_dir, exist_ok=True)



        try:
            # Create the safety copy directory if it doesn't exist

            copied_filenames = []

            # Iterate through all files in self.save_path
            to_models = osp.join(self.save_path, "models")
            for filename in os.listdir(to_models):
                if filename.startswith(model_name):
                    src_file = osp.join(to_models, filename)
                    dst_file = osp.join(safety_copy_dir, filename)

                    copied_filenames.append(filename)
                    
                    if osp.exists(dst_file):
                        print(f"File {filename} already exists in safety copies, skipping.")
                    else:
                        shutil.copy2(src_file, dst_file)
                        print(f"Copied {filename} to {safety_copy_dir}")
            

            j_path_no_suffix = osp.join(safety_path, f"safety_copies_{str_identifier}")
            j_dict = {"copied_filenames": copied_filenames}
            jh.dump_no_overwrite(j_path_no_suffix, j_dict)
            



            

        except OSError as e:
            print(f"Error creating safety copies: {e}")




        





