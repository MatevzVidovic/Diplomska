



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



import copy

import torch.nn as nn

from y_helpers.model_sorting import sort_tree_ixs




# create object for calculating model's flops
class ConvResourceCalc():
    
    def __init__(self, wrapper_model, initial_conv_resource_calc=None, target_modules=None):
        self.wrapper_model = wrapper_model
        self.initial_conv_resource_calc = initial_conv_resource_calc

        if target_modules is None:
            self.target_modules = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout, nn.Upsample, nn.ConvTranspose2d)
        else:
            self.target_modules = target_modules
        
        self.module_tree_ixs = set()
        self.module_tree_ix_2_children_tree_ix_list = {}
        self.module_tree_ix_2_all_parents_to_root_tree_ix_list = {}
        self.module_tree_ix_2_name = {}
        self.module_tree_ix_2_module_itself = {}

        self.calculate_layer_tree_ix_set_bug_prevention = set()
    

        self.resource_name_2_resource_dict = {
            "flops_num": {},
            "weights_num": {},
            "weights_dimensions": {},
            "kernels_num": {}
        }

        

    
    def get_copy_for_pickle(self):

        # deepcopy uses pickle in the background, so we have to do this:
        temp_wrapper_model = self.wrapper_model
        self.wrapper_model = None

        # It pickles without doing the following, but that is useless anyways,
        #  because it's pointing to an old model.
        # So better to cause an error than to just be a bug.
        temp_dict = self.module_tree_ix_2_module_itself
        self.module_tree_ix_2_module_itself = None

        pickleable_object = copy.deepcopy(self)
        
        
        self.wrapper_model = temp_wrapper_model
        self.module_tree_ix_2_module_itself = temp_dict

        
        return pickleable_object


    
    def _get_len_of_generator(self, gen):
        
        return sum(1 for x in gen)

    
    def _is_leaf(self, model):
        
        return self._get_len_of_generator(model.children()) == 0



    
    def calculate_layer(self, layer, tree_ix, x, *args, **kwargs):


        if tree_ix in self.calculate_layer_tree_ix_set_bug_prevention:
            print("Tree_ix already in set.")
            print(f"tree_ix: {tree_ix}")
            print(f"layer: {layer}")
            print(f"args: {args}")
            print(f"kwargs: {kwargs}")
            print(f"x: {x}")
            print(f"self.module_tree_ix_2_name[tree_ix]: {self.module_tree_ix_2_name[tree_ix]}")
            print(f"self.calculate_layer_tree_ix_set_bug_prevention: {self.calculate_layer_tree_ix_set_bug_prevention}")
            print(f"{self.calculate_layer_tree_ix_set_bug_prevention=}")
            
            raise ValueError("Tree_ix already in set.")


        self.calculate_layer_tree_ix_set_bug_prevention.add(tree_ix)


        y = layer.old_forward(x, *args, **kwargs)

        if isinstance(layer, nn.Conv2d):
            
            # print(f"y.shape: {y.shape}")
            # print(f"y: {y}")
            # print(f"assert: {y.shape[1] == layer.weight.size(0)}") # is true



            # dimensions of layer weights:
            # number of kernels (output chanels), depth of kernels (input chanels), kernel height, kernel width
            cur_weights = layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
            self.resource_name_2_resource_dict["weights_num"][tree_ix] = cur_weights

            self.resource_name_2_resource_dict["weights_dimensions"][tree_ix] = list(layer.weight.shape)

            self.resource_name_2_resource_dict["kernels_num"][tree_ix] = layer.weight.size(0)


            # dimensions of the output:
            # batch_size, number of kernels (output chanels), height, width
            h = y.shape[2]
            w = y.shape[3]

            cur_flops = h * w * cur_weights
            self.resource_name_2_resource_dict["flops_num"][tree_ix] = cur_flops


            
            # This is false, because the kernels are now not zeroed out, but actually removed:
            # (the word filter is used, because this is from the previous code, where it was used) 
            # (Also, I think that would have been wrong also, because (layer.weight.size(0) - n_removed_filters) would be correct)
            # self.cur_flops += h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)
            #self.original_flops += h * w * layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
            #self.n_removed_filters += n_removed_filters


        elif isinstance(layer, nn.BatchNorm2d):

            # Dovolj FLOPs in weights dovolj malo, da ne upostevam.
            for _, curr_dict in self.resource_name_2_resource_dict.items():
                curr_dict[tree_ix] = 0

            self.resource_name_2_resource_dict["weights_dimensions"][tree_ix] = list(layer.weight.shape)

        
        elif isinstance(layer, nn.ConvTranspose2d):


            # Prekompleksno se to upostevat za flops.
            for _, curr_dict in self.resource_name_2_resource_dict.items():
                curr_dict[tree_ix] = 0

            self.resource_name_2_resource_dict["weights_dimensions"][tree_ix] = list(layer.weight.shape)

        

        else:
            for _, curr_dict in self.resource_name_2_resource_dict.items():
                curr_dict[tree_ix] = 0
        



        
        return y
    

    
    def calculate_resources(self, input_example):
        # tale ubistvu spremeni forward tako, da poklice trace_layer na vsakem. V trace nardis dejansko forward, poleg tega pa se
        # izracunas stevilo flopov.
        
                


        
        def modify_forward(module, curr_tree_ix=(0,), current_path_list=None):
            if current_path_list is None:
                current_path_list = []

            module_name = type(module).__name__    #.lower()

            self.module_tree_ixs.add(curr_tree_ix)
            self.module_tree_ix_2_children_tree_ix_list[curr_tree_ix] = []
            self.module_tree_ix_2_name[curr_tree_ix] = module_name
            self.module_tree_ix_2_module_itself[curr_tree_ix] = module

            self.module_tree_ix_2_all_parents_to_root_tree_ix_list[curr_tree_ix] = current_path_list
            children_path_list = current_path_list + [curr_tree_ix]




            # We have to only change it for leaves, because some modules take skip connections and 
            # the lambda function takes more parameters than we made it for.
            """
            x = self.up1(x5, x4)
            TypeError: ConvResourceCalc.calculate_resources.<locals>.modify_forward.<locals>.new_forward.<locals>.lambda_forward() takes 1 positional argument but 2 were given
            """
            if self._is_leaf(module):
                
                def new_forward(m):
                    
                    def lambda_forward(x, *args, **kwargs):
                        
                        return self.calculate_layer(m, curr_tree_ix, x, *args, **kwargs)
                    
                    return lambda_forward

                module.old_forward = module.forward
                module.forward = new_forward(module)





            for ix, child in enumerate(module.children()):

                new_tree_ix = (curr_tree_ix, ix)

                self.module_tree_ix_2_children_tree_ix_list[curr_tree_ix].append(new_tree_ix)

                modify_forward(child, new_tree_ix, children_path_list)

            """
            # Direct approach:
                        
            for submodule in module.modules():
                if isinstance(submodule, self.target_modules):

                    def new_forward(layer):
                        def lambda_forward(x):
                            return self.calculate_layer(layer, x, submodule)

                        return lambda_forward

                    submodule.old_forward = submodule.forward
                    submodule.forward = new_forward(submodule)
            """



        
        def restore_forward(module):
                        
            for child in module.children():
                # leaf node
                if self._is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

            """
            # Direct approach:

            for submodule in module.modules():
                print(submodule)
                if isinstance(submodule, self.target_modules) and hasattr(submodule, 'old_forward'):
                    submodule.forward = submodule.old_forward
                    submodule.old_forward = None


            print(10*"\n" + "Children:")
            for child in module.children():
                print(child)
            """
        

        # We have calculations for the leaves - they are already in the dict.
        # Now we have to calculate the resources of the middle modules,
        # by simply summing the resources of their children.
        def recursively_calculate(tree_ix_2_resource_dict_to_populate, curr_tree_ix=(0,)):

            # print(self.module_tree_ix_2_children_tree_ix_list[curr_tree_ix])
            children_tree_ix_lists = self.module_tree_ix_2_children_tree_ix_list[curr_tree_ix]

            # If leaf, return what we have calculated.
            if len(children_tree_ix_lists) == 0:
                
                return tree_ix_2_resource_dict_to_populate[curr_tree_ix]


            running_sum = 0
            for child_tree_ix in children_tree_ix_lists:
                running_sum += recursively_calculate(tree_ix_2_resource_dict_to_populate, child_tree_ix)

            tree_ix_2_resource_dict_to_populate[curr_tree_ix] = running_sum
            
            
            return running_sum


            

        self.calculate_layer_tree_ix_set_bug_prevention = set() # empty it out

        modify_forward(self.wrapper_model.model)
        input_example = input_example.to(self.wrapper_model.device)
        _ = self.wrapper_model.model.forward(input_example)
        restore_forward(self.wrapper_model.model)


        # print(f"self.module_tree_ix_2_name: {self.module_tree_ix_2_name}")
        # print(self.resource_name_2_resource_dict["flops_num"])


        recursively_calculate(self.resource_name_2_resource_dict["flops_num"])
        recursively_calculate(self.resource_name_2_resource_dict["weights_num"])
        recursively_calculate(self.resource_name_2_resource_dict["kernels_num"])





    def get_resource_of_whole_model(self, resource_name):
        return self.resource_name_2_resource_dict[resource_name][(0,)]




    def get_lowest_level_module_tree_ixs(self):

        lowest_level_modules_tree_ixs = []
        for tree_ix, children_list in self.module_tree_ix_2_children_tree_ix_list.items():
            if len(children_list) == 0:
                lowest_level_modules_tree_ixs.append(tree_ix)
        
        return lowest_level_modules_tree_ixs



    def get_ordered_list_of_tree_ixs_for_layer_name(self, layer_name):
        

        applicable_tree_ixs = []
        for tree_ix, module_name in self.module_tree_ix_2_name.items():
            if module_name == layer_name:
                applicable_tree_ixs.append(tree_ix)
        
        assert len(applicable_tree_ixs) > 0, f"No module with name {layer_name} found."

        sorted_applicable_tree_ixs = sort_tree_ixs(applicable_tree_ixs)

        
        return sorted_applicable_tree_ixs
    
