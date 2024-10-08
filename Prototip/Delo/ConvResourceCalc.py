

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy




# create object for calculating model's flops
class ConvResourceCalc():
    def __init__(self, wrapper_model, initial_conv_resource_calc=None, target_modules=None):
        self.wrapper_model = wrapper_model
        self.initial_conv_resource_calc = initial_conv_resource_calc

        if target_modules is None:
            self.target_modules = (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Dropout, nn.Upsample, nn.ConvTranspose2d)
        else:
            self.target_modules = target_modules
    

    def get_copy_for_pickle(self):

        # deepcopy uses pickle in the background, so we have to do this:
        temp_wrapper_model = self.wrapper_model
        self.wrapper_model = None

        # It pickles without doing the following, but that is useless anyways,
        #  because it's pointing to an old model.
        # So better to cause an error than to just be a bug.
        temp_dict = self.module_tree_ixs_2_modules_themselves
        self.module_tree_ixs_2_modules_themselves = None

        pickleable_object = copy.deepcopy(self)
        
        
        self.wrapper_model = temp_wrapper_model
        self.module_tree_ixs_2_modules_themselves = temp_dict

        return pickleable_object


    def _get_len_of_generator(self, gen):
        return sum(1 for x in gen)

    def _is_leaf(self, model):
        return self._get_len_of_generator(model.children()) == 0



    def calculate_layer(self, layer, x, tree_ix):


        y = layer.old_forward(x)

        if isinstance(layer, nn.Conv2d):
            
            # print(f"y.shape: {y.shape}")
            # print(f"y: {y}")
            # print(f"assert: {y.shape[1] == layer.weight.size(0)}") # is true

            # dimensions of the output:
            # batch_size, number of filters (output chanels), height, width
            h = y.shape[2]
            w = y.shape[3]

            # dimensions of layer weights:
            # number of filters (output chanels), number of filters (input chanels), kernel height, kernel width
            cur_weights = layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
            cur_flops = h * w * cur_weights


            # self.module_resources_dict[module] = cur_flops
            self.module_tree_ixs_2_flops_dict[tree_ix] = cur_flops
            self.all_flops_num += cur_flops
            self.module_tree_ix_2_weights_dimensions[tree_ix] = list(layer.weight.shape)
            self.module_tree_ix_2_weights_num[tree_ix] = cur_weights
            self.all_weights_num += cur_weights

            # This is false, because the filters are now not zeroed out, but actually removed:
            # (Also, I think that would have been wrong also, because (layer.weight.size(0) - n_removed_filters) would be correct)
            # self.cur_flops += h * w * layer.weight.size(0) * (layer.weight.size(1) - n_removed_filters) * layer.weight.size(2) * layer.weight.size(3)
            #self.original_flops += h * w * layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(2) * layer.weight.size(3)
            #self.n_removed_filters += n_removed_filters

        # elif isinstance(layer, nn.BatchNorm2d):
        #     # ne upostevas, ker ne uporabis pri inferenci ene slike, na podlagi katere delas flop count.
        #     # rezanje tega je samo posledica rezanja konvolucije, nikoli ne mores samo tega rezat, zato ga ne sestevam..
        #     self.module_tree_ixs_2_flops_dict[tree_ix] = 0
        #     self.module_tree_ix_2_weights_num[tree_ix] = 0
        #     # pass


        else:
            self.module_tree_ixs_2_flops_dict[tree_ix] = 0
            self.module_tree_ix_2_weights_num[tree_ix] = 0

        



        return y
    

    def calculate_resources(self, input_example):
        # tale ubistvu spremeni forward tako, da poklice trace_layer na vsakem. V trace nardis dejansko forward, poleg tega pa se
        # izracunas stevilo flopov.
        #self.original_flops = 0
        self.all_flops_num = 0
        self.all_weights_num = 0
        #self.n_removed_filters = 0
        self.module_resources_dict = {}

        self.module_tree_ixs_2_flops_dict = {}
        self.module_tree_ix_2_weights_num = {}

        self.module_tree_ixs_2_children_tree_ix_lists = {}
        self.module_tree_ixs_2_all_parents_to_root_tree_ix_list = {}
        self.module_tree_ixs_2_name = {}
        self.module_tree_ixs_2_modules_themselves = {}
        self.module_tree_ix_2_weights_dimensions = {}


        def modify_forward(module, curr_tree_ix=(0,), current_path_list=[]):

            module_name = type(module).__name__    #.lower()

            self.module_tree_ixs_2_children_tree_ix_lists[curr_tree_ix] = []
            self.module_tree_ixs_2_name[curr_tree_ix] = module_name
            self.module_tree_ixs_2_modules_themselves[curr_tree_ix] = module

            self.module_tree_ixs_2_all_parents_to_root_tree_ix_list[curr_tree_ix] = current_path_list
            children_path_list = current_path_list + [curr_tree_ix]




            # We have to only change it for leaves, because some modules take skip connections and 
            # the lambda function takes more parameters than we made it for.
            """
            x = self.up1(x5, x4)
            TypeError: ConvResourceCalc.calculate_resources.<locals>.modify_forward.<locals>.new_forward.<locals>.lambda_forward() takes 1 positional argument but 2 were given
            """
            if self._is_leaf(module):
                def new_forward(m):
                    def lambda_forward(x):
                        return self.calculate_layer(m, x, curr_tree_ix)
                    return lambda_forward

                module.old_forward = module.forward
                module.forward = new_forward(module)





            for ix, child in enumerate(module.children()):

                new_tree_ix = (curr_tree_ix, ix)

                self.module_tree_ixs_2_children_tree_ix_lists[curr_tree_ix].append(new_tree_ix)

                modify_forward(child, new_tree_ix, children_path_list)


            # # Direct approach:
                        
            # for submodule in module.modules():
            #     if isinstance(submodule, self.target_modules):

            #         def new_forward(layer):
            #             def lambda_forward(x):
            #                 return self.calculate_layer(layer, x, submodule)

            #             return lambda_forward

            #         submodule.old_forward = submodule.forward
            #         submodule.forward = new_forward(submodule)



        def restore_forward(model):
            
            model_name = type(model).__name__.lower()
            
            for child in model.children():
                # leaf node
                if self._is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)


            # # Direct approach:

            # for submodule in module.modules():
            #     print(submodule)
            #     if isinstance(submodule, self.target_modules) and hasattr(submodule, 'old_forward'):
            #         submodule.forward = submodule.old_forward
            #         submodule.old_forward = None


            # print(10*"\n" + "Children:")
            # for child in module.children():
            #     print(child)
        


        # We have the FLOPs for the leaves. Elsewhere it is 0.
        # Now we recursively calculate the FLOPs of middle modules.
        def recursively_populate_flops(curr_tree_ix=(0,)):

            # print(self.module_tree_ixs_2_children_tree_ix_lists[curr_tree_ix])
            children_tree_ix_lists = self.module_tree_ixs_2_children_tree_ix_lists[curr_tree_ix]

            # If leaf, return what we have calculated.
            if len(children_tree_ix_lists) == 0:
                return self.module_tree_ixs_2_flops_dict[curr_tree_ix]


            cur_flops = 0
            for child_tree_ix in children_tree_ix_lists:
                cur_flops += recursively_populate_flops(child_tree_ix)

            self.module_tree_ixs_2_flops_dict[curr_tree_ix] = cur_flops
            
            return cur_flops




        def recursively_populate_weights_nums(curr_tree_ix=(0,)):

            # print(self.module_tree_ixs_2_children_tree_ix_lists[curr_tree_ix])
            children_tree_ix_lists = self.module_tree_ixs_2_children_tree_ix_lists[curr_tree_ix]

            # If leaf, return what we have calculated.
            if len(children_tree_ix_lists) == 0:
                return self.module_tree_ix_2_weights_num[curr_tree_ix]


            cur_weights_num = 0
            for child_tree_ix in children_tree_ix_lists:
                cur_weights_num += recursively_populate_weights_nums(child_tree_ix)

            self.module_tree_ix_2_weights_num[curr_tree_ix] = cur_weights_num
            
            return cur_weights_num
            


        modify_forward(self.wrapper_model.model)
        input_example = input_example.to(self.wrapper_model.device)
        y = self.wrapper_model.model.forward(input_example)
        restore_forward(self.wrapper_model.model)
        # print(self.module_tree_ixs_2_name)
        # print(self.module_tree_ixs_2_flops_dict)
        recursively_populate_flops()
        # print(self.module_tree_ixs_2_flops_dict)
        # print(self.all_flops_num)
        recursively_populate_weights_nums()
