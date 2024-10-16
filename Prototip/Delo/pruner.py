


import torch
import ConvResourceCalc

from model_vizualization import model_graph

from TrainingWrapper import TrainingWrapper



import logging
import sys
import os

# Assuming the submodule is located at 'python_logger'
submodule_path = os.path.join(os.path.dirname(__file__), 'python_logger')
sys.path.insert(0, submodule_path)

import python_logger.log_helper as py_log

MY_LOGGER = logging.getLogger("prototip")


class pruner:



    @py_log.log(passed_logger=MY_LOGGER)
    def __init__(self, FLOPS_min_resource_percentage, weights_min_resource_percentage, initial_conv_resource_calc, connection_lambda, inextricable_connection_lambda, conv_tree_ixs, batch_norm_ixs, lowest_level_modules):
        self.initial_conv_resource_calc = initial_conv_resource_calc
        self.FLOPS_min_resource_percentage_dict = FLOPS_min_resource_percentage.min_resource_percentage_dict
        self.weights_min_resource_percentage_dict = weights_min_resource_percentage.min_resource_percentage_dict
        self.connection_lambda = connection_lambda
        self.inextricable_connection_lambda = inextricable_connection_lambda
        self.conv_tree_ixs = conv_tree_ixs
        self.lowest_level_modules = lowest_level_modules


        self.tree_ix_2_list_of_initial_kernel_ixs = {}
        for tree_ix in conv_tree_ixs + batch_norm_ixs:
            # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
            # 0. dim is the number of kernels of this layer.
            # 0. dim also works for batchnorm, for which this is only needed for display of what kernels have been pruned.
            kernel_num = initial_conv_resource_calc.module_tree_ix_2_weights_dimensions[tree_ix][0]
            self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix] = list(range(kernel_num))


        self.tree_ix_2_list_of_initial_input_slice_ixs = {}
        for tree_ix in conv_tree_ixs:
            # 1. dim is the number input size of this layer.
            input_slice_num = initial_conv_resource_calc.module_tree_ix_2_weights_dimensions[tree_ix][1]
            self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix] = list(range(input_slice_num))

        
        self.pruning_logs = {
            "conv" : [],
            "batch_norm" : [],
            "following" : []
        }
        
        
    

    @py_log.log(passed_logger=MY_LOGGER)
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                py_log.log_locals(passed_logger=MY_LOGGER)
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        py_log.log_locals(passed_logger=MY_LOGGER)
        return -1  # Target not found
    




    @py_log.log(passed_logger=MY_LOGGER)
    def prune_current_layer(self, tree_ix, real_kernel_ix, wrapper_model, tree_ix_2_module):

        # get the module
        module = tree_ix_2_module[tree_ix]
        old_weights = module.weight.data

        # print(f"old_weights.shape: {old_weights.shape}")
        # input()

        # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
        new_weights = torch.cat([old_weights[:real_kernel_ix, :, :, :], old_weights[real_kernel_ix+1:, :, :, :]], dim=0)
        module.weight.data = new_weights # torch.nn.Parameter(new_weights)

        # apparently better to clear out between two forward-backward passes
        # I suspect it would work without this too
        module.weight.grad = None



        if module.bias is not None:
            old_bias = module.bias.data
            new_bias = torch.cat([old_bias[:real_kernel_ix], old_bias[real_kernel_ix+1:]], dim=0)
            module.bias.data = new_bias # torch.nn.Parameter(new_bias)
            module.bias.grad = None

        # Now we have to update the list of initial kernels
        print(f"Pruned {tree_ix}, real_kernel_ix: {real_kernel_ix}, initial_kernel_ix: {self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix]}")
        self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix].pop(real_kernel_ix)


        inextricable_following_to_prune = self.inextricable_connection_lambda(tree_ix, real_kernel_ix, self.conv_tree_ixs, self.lowest_level_modules)
        for tree_ix, real_input_slice_ix in inextricable_following_to_prune:
            self.prune_inextricably_following_layer(tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module)




        # prune.random_unstructured(module, name="weight", amount=0.3)

        # # zadnje poglavje tega si bo treba prebrat:
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#pruning-multiple-parameters-in-a-model

        # Pač ta class bo treba razširit in potem njegove metode uporabljat.
        # Veliko bolje, kot da sam delam surgery po utežeh.
        # Samo ta compute_mask() bo treba spremenit. Pa pogledat je treba kaj je ta global, structured, unstructured
        # Pomoje bo treba imet samo pač structured, ker je local, ampak zdajle ne vem zadosti.
        # pass

        py_log.log_locals(passed_logger=MY_LOGGER)
        return

    @py_log.log(passed_logger=MY_LOGGER)
    def prune_inextricably_following_layer(self, tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module):

        # get the module
        module = tree_ix_2_module[tree_ix]
        
        if type(module) == torch.nn.BatchNorm2d:

            # For the comments of the lines, go look at pruner.prune_current_layer()
            # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]

            old_weights = module.weight.data
            new_weights = torch.cat([old_weights[:real_input_slice_ix], old_weights[real_input_slice_ix+1:]], dim=0)
            module.weight.data = new_weights
            module.weight.grad = None

            old_bias = module.bias.data
            new_bias = torch.cat([old_bias[:real_input_slice_ix], old_bias[real_input_slice_ix+1:]], dim=0)
            module.bias.data = new_bias
            module.bias.grad = None

            old_running_mean = module.running_mean.data
            new_running_mean = torch.cat([old_running_mean[:real_input_slice_ix], old_running_mean[real_input_slice_ix+1:]], dim=0)
            module.running_mean.data = new_running_mean
            module.running_mean.grad = None

            old_running_var = module.running_var.data
            new_running_var = torch.cat([old_running_var[:real_input_slice_ix], old_running_var[real_input_slice_ix+1:]], dim=0)
            module.running_var.data = new_running_var
            module.running_var.grad = None


            initial_input_slice_ix = self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_input_slice_ix]
            self.pruning_logs["batch_norm"].append((tree_ix, real_input_slice_ix, initial_input_slice_ix))
            print(f"Pruned {tree_ix}, real kernel ix (in code real_input_slice_ix): {real_input_slice_ix}, initial_input_slice_ix: {initial_input_slice_ix}")
            self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix].pop(real_input_slice_ix)

        py_log.log_locals(passed_logger=MY_LOGGER)
        return









    @py_log.log(passed_logger=MY_LOGGER)
    def prune_following_layer(self, tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module):


        # get the module
        module = tree_ix_2_module[tree_ix]



        # pruning convolutional layers:

        old_weights = module.weight.data

        # print(f"old_weights.shape: {old_weights.shape}")
        # input()

        # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
        # Here real_input_slice_ix is refering to the input channels (each channel is the result of one kernel. Which one of those are we pruning.)
        new_weights = torch.cat([old_weights[:, :real_input_slice_ix, :, :], old_weights[:, real_input_slice_ix+1:, :, :]], dim=1)
        module.weight.data = new_weights # torch.nn.Parameter(new_weights)

        # apparently better to clear out between two forward-backward passes
        # I suspect it would work without this too
        module.weight.grad = None


        print(f"Pruned {tree_ix}, real_input_slice_ix: {real_input_slice_ix}, initial_input_slice_ix: {self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix][real_input_slice_ix]}")
        self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix].pop(real_input_slice_ix)

        
        py_log.log_locals(passed_logger=MY_LOGGER)
        return




    
    def prune(self, importance_dict, curr_conv_resource_calc: ConvResourceCalc, wrapper_model: TrainingWrapper):

        

        
        # We sort the kernels by their importance

        # importance_dict is:  {tree_ix: 1d_tensor_of_importances}
        # We have to convert these 1d tensors to
        # lists of tuples of the form (tree_ix, real_kernel_ix, importance).
        # Concat all these lists. Then sort by the importance.

        sortable_list = []
        for tree_ix, importance_tensor in importance_dict.items():
            for real_kernel_ix, importance in enumerate(importance_tensor):
                sortable_list.append((tree_ix, real_kernel_ix, float(importance)))
        
        sortable_list.sort(key=lambda x: x[2])

        # print(sortable_list)
        # input()

        # Then we find out which tree_ixs are disallowed due to resource thresholds

        # This would be better if we had conv_resource_calc.module_tree_ix_2_all_children_conv_tree_ixs_list
        # We would only go through all tree_ixs once and record which conv_tree_ixs are disallowed.
        # Now we go through all tree_ixs once, and then for all conv tree ix-s we go to the root. Not as nice.

        # first we find all the tree_ixs which are disallowed directly

        print(self.FLOPS_min_resource_percentage_dict)

        disallowed_directly = set()
        for tree_ix in self.FLOPS_min_resource_percentage_dict:
            try:
                curr_flops_percentage = curr_conv_resource_calc.module_tree_ix_2_flops_num[tree_ix] / self.initial_conv_resource_calc.module_tree_ix_2_flops_num[tree_ix]
            except ZeroDivisionError:
                curr_flops_percentage = 0
            print(self.FLOPS_min_resource_percentage_dict[tree_ix])
            print(curr_flops_percentage)
            if curr_flops_percentage < self.FLOPS_min_resource_percentage_dict[tree_ix]:
                disallowed_directly.add(tree_ix)

        for tree_ix in self.weights_min_resource_percentage_dict:
            try:
                curr_weights_percentage = curr_conv_resource_calc.module_tree_ix_2_weights_num[tree_ix] / self.initial_conv_resource_calc.module_tree_ix_2_weights_num[tree_ix]
            except ZeroDivisionError:
                curr_weights_percentage = 0
                
            if curr_weights_percentage < self.weights_min_resource_percentage_dict[tree_ix]:
                disallowed_directly.add(tree_ix)
        

        # Then we find all conv tree_ixs (only those are relevant to us) which are disallowed directly or due to their parents
        disallowed_tree_ixs = set()
        for tree_ix in self.conv_tree_ixs:
            if tree_ix in disallowed_directly:
                disallowed_tree_ixs.add(tree_ix)
            else:
                parents = curr_conv_resource_calc.module_tree_ix_2_all_parents_to_root_tree_ix_list[tree_ix]
                for parent in parents:
                    if parent in disallowed_directly:
                        disallowed_tree_ixs.add(tree_ix)
                        break
        
        print(10*"-")
        print(f"sortable_list[:5]: {sortable_list[:5]}")
        print(f"disallowed_directly: {disallowed_directly}")
        print(f"disallowed_tree_ixs: {disallowed_tree_ixs}")


        # Then we pick the least important one which isn't disallowed
        to_prune = None
        for tree_ix, real_kernel_ix, importance in sortable_list:
            if tree_ix in disallowed_tree_ixs:
                continue
            to_prune = (tree_ix, real_kernel_ix)
            break

        if to_prune is None:
            print("No more to prune.")
            py_log.log_locals(passed_logger=MY_LOGGER)
            return

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # THIS HAS TO HAPPEN BEFORE ANY PRUNING TAKES PLACE.
        # BEFORE IT DIDN'T - AND THIS MESSES UP THE ORDERING OF
        # self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]][to_prune[1]]
        # because one element is alredy popped because of the pruning.
        # And everything is messed up.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        

        # to find the next (tree_ix, real_input_slice_ix) to prune, we have to go through the connection lambda
        # however, the lambda works on the indexes of the initial unpruned model.
        # So, we have to find out the initial_kernel_ix of the real kernel_ix.
        # Then put that in the lambda to get the goal's initial_input_slice_ix.
        # Then change that to the real_input_slice_ix (the ix of where the initial slice is right now).

        initial_kernel_ix = self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]][to_prune[1]]
        following_to_prune = self.connection_lambda(to_prune[0], initial_kernel_ix, self.conv_tree_ixs, self.lowest_level_modules)


        conv_tree_ixs = curr_conv_resource_calc.get_ordered_list_of_tree_ixs_for_layer_name("Conv2d")
        self.pruning_logs["conv"].append((conv_tree_ixs.index(to_prune[0]), to_prune[1], initial_kernel_ix))


        # Transform the initial_kernel_ixs to real_kernel_ixs:
        # Fix the following_to_prune to be in the form (tree_ix, real_input_slice_ix, inital_input_slice_ix)
        # TODO: could do self.binary search for speed, but it is for later
        following_to_prune = [(i, self.tree_ix_2_list_of_initial_input_slice_ixs[i].index(j), j) for i,j in following_to_prune]
        self.pruning_logs["following"].append([(conv_tree_ixs.index(i), j, k) for i,j,k in following_to_prune])




        # And we prune it.

        tree_ix_2_module = curr_conv_resource_calc.module_tree_ix_2_module_itself

        try:
            # method for pruning current
            self.prune_current_layer(to_prune[0], to_prune[1], wrapper_model, tree_ix_2_module)
        except KeyError as e:
            print(f"Pruning {to_prune} failed.")
            self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]]
            py_log.log_locals(passed_logger=MY_LOGGER)
            raise e

        # on those the method of next to be pruned (its a different pruning method)
        for tree_ix, real_input_slice_ix, _ in following_to_prune:

            self.prune_following_layer(tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module)
        


        



        LLMs = curr_conv_resource_calc.get_lowest_level_module_tree_ixs()
        # YOU MUSTN'T DO initial_ker_ixs[real_kernel_ix] OR .index(initial_input_slice_ix) HERE, BECAUSE THE PRUNING HAS HAPPENED SO THE LIST IS WRONG ALREADY
        print(10*"-")
        print(f"Pruned conv_tree_ix: {conv_tree_ixs.index(to_prune[0])}\n (LLM, kernel_ix) {LLMs.index(to_prune[0])}. {to_prune[1]} \n {to_prune}")
        print(f"Pruned [(conv_tree_ix, LLM, inp_slice_ix),...] {[(conv_tree_ixs.index(i), LLMs.index(i), j, k) for i,j,k in following_to_prune]} \n {following_to_prune}")
        print(10*"-")
        print(4*"\n")



        py_log.log_locals(passed_logger=MY_LOGGER)
        return


        

