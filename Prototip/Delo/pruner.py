

import logging
import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip")
MY_LOGGER.setLevel(logging.DEBUG)


import torch
import ConvResourceCalc

from model_vizualization import model_graph

from TrainingWrapper import TrainingWrapper






class pruner:



    @py_log.log(passed_logger=MY_LOGGER)
    def __init__(self, FLOPS_min_resource_percentage, weights_min_resource_percentage, initial_conv_resource_calc, input_slice_connection_fn, kernel_connection_fn, conv_tree_ixs, batch_norm_ixs, lowest_level_modules, input_example):
        self.initial_conv_resource_calc = initial_conv_resource_calc
        self.FLOPS_min_resource_percentage_dict = FLOPS_min_resource_percentage.min_resource_percentage_dict
        self.weights_min_resource_percentage_dict = weights_min_resource_percentage.min_resource_percentage_dict
        self.input_slice_connection_fn = input_slice_connection_fn
        self.kernel_connection_fn = kernel_connection_fn
        self.conv_tree_ixs = conv_tree_ixs
        self.lowest_level_modules = lowest_level_modules
        self.input_example = input_example


        self.tree_ix_2_list_of_initial_kernel_ixs = {}
        for tree_ix in conv_tree_ixs + batch_norm_ixs:
            # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
            # 0. dim is the number of kernels of this layer.
            # 0. dim also works for batchnorm, for which this is only needed for display of what kernels have been pruned.
            kernel_num = initial_conv_resource_calc.resource_name_2_resource_dict["weights_dimensions"][tree_ix][0]
            self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix] = list(range(kernel_num))


        self.tree_ix_2_list_of_initial_input_slice_ixs = {}
        for tree_ix in conv_tree_ixs:
            # 1. dim is the number input size of this layer.
            input_slice_num = initial_conv_resource_calc.resource_name_2_resource_dict["weights_dimensions"][tree_ix][1]
            self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix] = list(range(input_slice_num))

        
        self.pruning_logs = {
            "conv" : [],
            "batch_norm" : [],
            "following" : []
        }
        
        
    



    @py_log.log(passed_logger=MY_LOGGER)
    def prune_current_layer(self, tree_ix, real_kernel_ix, wrapper_model, tree_ix_2_module):

        # get the module
        module = tree_ix_2_module[tree_ix]


        if type(module) == torch.nn.BatchNorm2d:
            self.prune_batchnorm(tree_ix, real_kernel_ix, wrapper_model, tree_ix_2_module)
            return


        # This has to happen before pruning, so that the initial_kernel_ix is correct.
        initial_kernel_ix = self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix]
        self.pruning_logs["conv"].append((self.conv_tree_ixs.index(tree_ix), real_kernel_ix, initial_kernel_ix))



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
        print(f"Pruned {tree_ix}, real_kernel_ix: {real_kernel_ix}, initial_kernel_ix: {initial_kernel_ix}")
        
        self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix].pop(real_kernel_ix)





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
    def prune_batchnorm(self, tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module):

        # get the module
        module = tree_ix_2_module[tree_ix]
        

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



    def _get_disallowed_tree_ixs(self, curr_conv_resource_calc: ConvResourceCalc):

        # This would be better if we had conv_resource_calc.module_tree_ix_2_all_children_conv_tree_ixs_list
        # We would only go through all tree_ixs once and record which conv_tree_ixs are disallowed.
        # Now we go through all tree_ixs once, and then for all conv tree ix-s we go to the root. Not as nice.

        # first we find all the tree_ixs which are disallowed directly

        # print(self.FLOPS_min_resource_percentage_dict)

        disallowed_directly = set()
        for tree_ix in self.FLOPS_min_resource_percentage_dict:
            try:
                curr_flops_percentage = curr_conv_resource_calc.resource_name_2_resource_dict["flops_num"][tree_ix] / self.initial_conv_resource_calc.resource_name_2_resource_dict["flops_num"][tree_ix]
            except ZeroDivisionError:
                curr_flops_percentage = 0

            # print(self.FLOPS_min_resource_percentage_dict[tree_ix])
            # print(curr_flops_percentage)
            if curr_flops_percentage < self.FLOPS_min_resource_percentage_dict[tree_ix]:
                disallowed_directly.add(tree_ix)

        for tree_ix in self.weights_min_resource_percentage_dict:
            try:
                curr_weights_percentage = curr_conv_resource_calc.resource_name_2_resource_dict["weights_num"][tree_ix] / self.initial_conv_resource_calc.resource_name_2_resource_dict["weights_num"][tree_ix]
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
        
        return disallowed_directly, disallowed_tree_ixs

    
    def prune(self, num_to_prune, importance_dict, curr_conv_resource_calc: ConvResourceCalc, wrapper_model: TrainingWrapper):

        # Returns True if there are more to prune in the future, False if there are no more to prune.

        
        
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

        
        
        print(10*"-")
        print(f"sortable_list[:5]: {sortable_list[:5]}")





        # This step could be omitted, because .prune_one_layer_recursive() does it.
        # But ist is better to have it, so that if we already know from the current tree_ix, that this can't be pruned,
        #  we don't call .prune_one_layer_recursive() which would have to build this disallowed set and then just return False.
        
        # Just to explain: .prune_one_layer_recursive() doesn't just check if the tree_ix of the current layer is disallowed.
        # It also checks if any of the layers that would recursively be pruned are disallowed.
        # If any of them is, it returns False without pruning anything.

        # This is important for networks like ResNet, where the layers are connected in a way that if you prune one, you have to prune the following ones too.
        # Not just the input slices of the following layers, but actually prune their kernels.

        # So, if the current layer is disallowed before we even started with pruning, no need to run .prune_one_layer_recursive().


        # And this isn't even faster, because we have to make the list of all the kernels that can be pruned.
        
        disallowed_directly, disallowed_tree_ixs = self._get_disallowed_tree_ixs(curr_conv_resource_calc)

        print(f"disallowed_directly: {disallowed_directly}")
        print(f"disallowed_tree_ixs: {disallowed_tree_ixs}")

        # Then we find the least important ones that aren't disallowed.
        to_prune = [(tree_ix, real_kernel_ix) for tree_ix, real_kernel_ix, importance in sortable_list if tree_ix not in disallowed_tree_ixs]
        

        if len(to_prune) == 0:
            print("No more to prune!!!")
            return False
        

        

        # We have to take to_prune to the initial kernel_ixs, because with the current mechanism of pruning num_to_prune at once
        # once we prune a kernel, if we then also go prune a kernel in the same layer, the kernel ix is wrong.
        to_prune = [(tree_ix, self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix]) for tree_ix, real_kernel_ix in to_prune] 




        num_pruned = 0
        for to_prune_elem in to_prune:
            curr_to_prune_elem = (to_prune_elem[0], self.tree_ix_2_list_of_initial_kernel_ixs[to_prune_elem[0]].index(to_prune_elem[1]))
            # this also does curr_conv_resource_calc.calculate_resources(self.input_example)
            succeeded = self.prune_one_layer_recursive(curr_to_prune_elem, curr_conv_resource_calc, wrapper_model)
            if succeeded:
                num_pruned += 1
            if num_pruned >= num_to_prune:
                return True # because there are more to prune
            
        if num_pruned == 0:
            print("No more to prune!!!")
        else:
            print(f"Pruned {num_pruned} layers. No more to prune.")
        
        return False # because there are no more to prune
        

        



    def prune_one_layer_recursive(self, to_prune, curr_conv_resource_calc: ConvResourceCalc, wrapper_model: TrainingWrapper, check_if_disallowed=True):
        
        # check_disallowed is True in usual calls.
        # But when it calls itself recursively, it is False - because it had to be checked in advance anyway.


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # THIS HAS TO HAPPEN BEFORE ANY PRUNING TAKES PLACE.
        # BEFORE IT DIDN'T - AND THIS MESSES UP THE ORDERING OF
        # self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]][to_prune[1]]
        # because one element is alredy popped because of the pruning.
        # And everything is messed up.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        

        # to find the next (tree_ix, real_input_slice_ix) to prune, we have to go through the connection fn
        # however, the fn works on the indexes of the initial unpruned model.
        # So, we have to find out the initial_kernel_ix of the real kernel_ix.
        # Then put that in the fn to get the goal's initial_input_slice_ix.
        # Then change that to the real_input_slice_ix (the ix of where the initial slice is right now).

        # print(self.tree_ix_2_list_of_initial_kernel_ixs)
        # print(to_prune)

        initial_kernel_ix = self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]][to_prune[1]]



        # Checking this enables the num_to_prune parameter in .prune()
        # (otherwise a high num_to_prune could be pruning just one layer and take it far past its minimum kernel amount limit).
        
        # Checking this for ResNet is absolutely necessary.
        # Because in the inextricable_following_to_prune there are also convolutional layers, not just batchnorms (which are generally always okay to prune anyway).
        # So we would violate the limit of the amount of pruned kernels for these layers.
        
        if check_if_disallowed:
            inextricable_following_to_prune = self.kernel_connection_fn(to_prune[0], to_prune[1], self.conv_tree_ixs, self.lowest_level_modules)
            all_that_will_be_recursively_pruned = set(inextricable_following_to_prune)
            all_that_still_need_to_be_added = inextricable_following_to_prune.copy()
            while len(all_that_still_need_to_be_added) > 0:
                curr = all_that_still_need_to_be_added.pop()
                curr_inextricable_following_to_prune = self.kernel_connection_fn(curr[0], curr[1], self.conv_tree_ixs, self.lowest_level_modules)
                for i in curr_inextricable_following_to_prune:
                    if i not in all_that_will_be_recursively_pruned:
                        all_that_will_be_recursively_pruned.add(i)
                        all_that_still_need_to_be_added.append(i)
                    else:
                        print("WATCH FOR POSSIBLE RECURSION IN PRUNING CONNECTIONS!!!")
            

            # Checking if we are even able to do all of this pruning.
            # (following_to_prune can always be pruned tho, because we are pruning input slices.)
            _, disallowed_tree_ixs = self._get_disallowed_tree_ixs(curr_conv_resource_calc)

            if to_prune[0] in disallowed_tree_ixs:
                return False
            
            for i in all_that_will_be_recursively_pruned:
                if i[0] in disallowed_tree_ixs:
                    return False




        following_to_prune = self.input_slice_connection_fn(to_prune[0], initial_kernel_ix, self.conv_tree_ixs, self.lowest_level_modules)

        # print(f"following_to_prune: {following_to_prune}")
        # print(self.input_slice_connection_fn)
        # print(to_prune)
        # print(self.conv_tree_ixs)
        # print(self.lowest_level_modules)


        # Transform the initial_kernel_ixs to real_kernel_ixs:
        # Fix the following_to_prune to be in the form (tree_ix, real_input_slice_ix, inital_input_slice_ix)
        # TODO: could do self.binary search for speed, but it is for later
        following_to_prune = [(i, self.tree_ix_2_list_of_initial_input_slice_ixs[i].index(j), j) for i,j in following_to_prune]
        



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
        


        self.pruning_logs["following"].append([(self.conv_tree_ixs.index(i), j, k) for i,j,k in following_to_prune])
        # on those the method of next to be pruned (its a different pruning method)
        for tree_ix, real_input_slice_ix, _ in following_to_prune:

            self.prune_following_layer(tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module)
        


        



        # LLMs = curr_conv_resource_calc.get_lowest_level_module_tree_ixs()
        # # YOU MUSTN'T DO initial_ker_ixs[real_kernel_ix] OR .index(initial_input_slice_ix) HERE, BECAUSE THE PRUNING HAS HAPPENED SO THE LIST IS WRONG ALREADY
        # print(10*"-")
        # print(f"Pruned conv_tree_ix: {conv_tree_ixs.index(to_prune[0])}\n (LLM, kernel_ix) {LLMs.index(to_prune[0])}. {to_prune[1]} \n {to_prune}")
        # print(f"Pruned [(conv_tree_ix, LLM, inp_slice_ix),...] {[(conv_tree_ixs.index(i), LLMs.index(i), j, k) for i,j,k in following_to_prune]} \n {following_to_prune}")
        # print(10*"-")
        # print(4*"\n")


        # Basically, first we prune the current layer, 
        # then we prune the following layers - which really means just correcting the input dimensions of the following layers.

        # Now we do the recursion step. We find the layers which are "inextricably connected" to the pruned layer.
        # Which really means that their kernels get pruned because of this - so the same pruning happens to them as did to the first layer.
        print(10*"-")

        inextricable_following_to_prune = self.kernel_connection_fn(to_prune[0], to_prune[1], self.conv_tree_ixs, self.lowest_level_modules)        
        for tree_ix, kernel_ix in inextricable_following_to_prune:
            self.prune_one_layer_recursive((tree_ix, kernel_ix), curr_conv_resource_calc, wrapper_model, check_if_disallowed=False)
        
        # When the call of the original pruning call is finished, the dimensions across the network are corrected and they line up correctly.
        # We have to do this recalculation, so that when this function gets called next time, the resources are correct and we correctly know what is allowed to be pruned.
        if check_if_disallowed:
            curr_conv_resource_calc.calculate_resources(self.input_example)
        

        return True

