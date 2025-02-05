

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




import torch

from y_framework.conv_resource_calc import ConvResourceCalc
from y_framework.training_wrapper import TrainingWrapper

from y_helpers.model_vizualization import model_graph







class Pruner:



    @py_log.autolog(passed_logger=MY_LOGGER)
    def __init__(self, pruning_disallowments, initial_conv_resource_calc, input_slice_connection_fn, kernel_connection_fn, conv_tree_ixs, other_zeroth_dim_ixs, lowest_level_modules, input_example):
        try:
            self.initial_conv_resource_calc = initial_conv_resource_calc
            self.pruning_disallowments = pruning_disallowments
            self.input_slice_connection_fn = input_slice_connection_fn
            self.kernel_connection_fn = kernel_connection_fn
            self.conv_tree_ixs = conv_tree_ixs
            self.lowest_level_modules = lowest_level_modules
            self.input_example = input_example


            self.tree_ix_2_list_of_initial_kernel_ixs = {}
            # one example of other_zeroth_dim_ixs is the batchnorms
            for tree_ix in conv_tree_ixs + other_zeroth_dim_ixs:
                # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
                # 0. dim is the number of kernels of this layer.
                # 0. dim also works for batchnorm, for which this is only needed for display of what kernels have been pruned.
                dims = initial_conv_resource_calc.resource_name_2_resource_dict["weights_dimensions"][tree_ix]
                kernel_num = dims[0]
                self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix] = list(range(kernel_num))


            self.tree_ix_2_list_of_initial_input_slice_ixs = {}
            for tree_ix in conv_tree_ixs:
                # 1. dim is the number input size of this layer.
                input_slice_num = initial_conv_resource_calc.resource_name_2_resource_dict["weights_dimensions"][tree_ix][1]
                self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix] = list(range(input_slice_num))

            
            self.pruning_logs = {
                "conv" : [],
                "batch_norm" : [],
                "upconv" : [],
                "following" : []
            }
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e
        
        
    



    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune_current_layer(self, tree_ix, real_kernel_ix, tree_ix_2_module):
        try:

            # get the module
            module = tree_ix_2_module[tree_ix]


            if isinstance(module, torch.nn.BatchNorm2d):
                self.prune_batchnorm(tree_ix, real_kernel_ix, tree_ix_2_module)
                return
            elif isinstance(module, torch.nn.ConvTranspose2d):
                self.prune_upconvolution(tree_ix, real_kernel_ix, tree_ix_2_module)
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

            # py_log.log_locals(passed_logger=MY_LOGGER)
            return
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e


    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune_batchnorm(self, tree_ix, real_input_slice_ix, tree_ix_2_module):

        try:
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


            # py_log.log_locals(passed_logger=MY_LOGGER)
            return

        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e




    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune_upconvolution(self, tree_ix, real_kernel_ix, tree_ix_2_module):
        try:

            # get the module
            module = tree_ix_2_module[tree_ix]


            # This has to happen before pruning, so that the initial_kernel_ix is correct.
            initial_kernel_ix = self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix]
            self.pruning_logs["upconv"].append((tree_ix, real_kernel_ix, initial_kernel_ix))




            # pruning convolutional layers:

            old_weights = module.weight.data

            # print(f"old_weights.shape: {old_weights.shape}")
            # input()

            # weight dimensions: !!!different than Conv2d!!!! [output_channels , input_channels, kernel_height, kernel_width]
            new_weights = torch.cat([old_weights[:real_kernel_ix, :, :, :], old_weights[real_kernel_ix+1:, :, :, :]], dim=0)
            module.weight.data = new_weights # torch.nn.Parameter(new_weights)

            # apparently better to clear out between two forward-backward passes
            # I suspect it would work without this too
            module.weight.grad = None

            # Now we have to update the list of initial kernels
            print(f"Pruned upconv {tree_ix}, input slice ix (in code real_kernel_ix): {real_kernel_ix}, initial_kernel_ix: {initial_kernel_ix}")
            
            self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix].pop(real_kernel_ix)









            # py_log.log_locals(passed_logger=MY_LOGGER)
            return
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e








    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune_following_layer(self, tree_ix, real_input_slice_ix, tree_ix_2_module):

        try:

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

            
            # py_log.log_locals(passed_logger=MY_LOGGER)
            return

        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e










    def _get_disallowed_tree_ixs(self, curr_conv_resource_calc: ConvResourceCalc):
        
        try:
            # This would be better if we had conv_resource_calc.module_tree_ix_2_all_children_conv_tree_ixs_list
            # We would only go through all tree_ixs once and record which conv_tree_ixs are disallowed.
            # Now we go through all tree_ixs once, and then for all conv tree ix-s we go to the root. Not as nice.

            # first we find all the tree_ixs which are disallowed directly

            # print(self.pruning_disallowments["FLOPS"])


            network_flops = curr_conv_resource_calc.get_resource_of_whole_model("flops_num")
            network_weights = curr_conv_resource_calc.get_resource_of_whole_model("weights_num")

            network_flops_initial = self.initial_conv_resource_calc.get_resource_of_whole_model("flops_num")
            network_weights_initial = self.initial_conv_resource_calc.get_resource_of_whole_model("weights_num")

            network_flops_percentage = network_flops / network_flops_initial
            network_weights_percentage = network_weights / network_weights_initial

            print(f"network_flops_percentage: {network_flops_percentage}")
            print(f"network_weights_percentage: {network_weights_percentage}")


            disallowed_directly = set()

            for tree_ix, limit in self.pruning_disallowments["general"].items():
                if limit > 1.0:
                    disallowed_directly.add(tree_ix)


            for tree_ix, limit in self.pruning_disallowments["FLOPS"].items():
                try:
                    curr_flops_percentage = curr_conv_resource_calc.resource_name_2_resource_dict["flops_num"][tree_ix] / self.initial_conv_resource_calc.resource_name_2_resource_dict["flops_num"][tree_ix]
                except ZeroDivisionError:
                    curr_flops_percentage = 0

                relative_disallowment_limit = self.pruning_disallowments["relative_FLOPS"][tree_ix] * network_flops_percentage
                final_limit = min(limit, relative_disallowment_limit)

                # print(self.pruning_disallowments["FLOPS"][tree_ix])
                # print(curr_flops_percentage)
                if curr_flops_percentage <= final_limit:
                    disallowed_directly.add(tree_ix)


            for tree_ix, limit in self.pruning_disallowments["weights"].items():
                try:
                    curr_weights_percentage = curr_conv_resource_calc.resource_name_2_resource_dict["weights_num"][tree_ix] / self.initial_conv_resource_calc.resource_name_2_resource_dict["weights_num"][tree_ix]
                except ZeroDivisionError:
                    curr_weights_percentage = 0

                relative_disallowment_limit = self.pruning_disallowments["relative_weights"][tree_ix] * network_weights_percentage
                final_limit = min(limit, relative_disallowment_limit)
                    
                if curr_weights_percentage <= final_limit:
                    disallowed_directly.add(tree_ix)



            for tree_ix, limit in self.pruning_disallowments["kernel_num"].items():

                curr_kernel_num = curr_conv_resource_calc.resource_name_2_resource_dict["kernels_num"].get(tree_ix, None)
                kernel_num_limit = self.pruning_disallowments["kernel_num"][tree_ix]

                if not curr_kernel_num is None and curr_kernel_num <= kernel_num_limit:
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
            


            disallowed_choices = set()
            for tree_ix, limit in self.pruning_disallowments["choice"].items():
                if limit > 1.0:
                    disallowed_choices.add(tree_ix)
            

            disallowed_choices_tree_ixs = set()
            for tree_ix in self.conv_tree_ixs:
                if tree_ix in disallowed_choices:
                    disallowed_choices_tree_ixs.add(tree_ix)
                else:
                    parents = curr_conv_resource_calc.module_tree_ix_2_all_parents_to_root_tree_ix_list[tree_ix]
                    for parent in parents:
                        if parent in disallowed_choices:
                            disallowed_choices_tree_ixs.add(tree_ix)
                            break

            
            return disallowed_directly, disallowed_tree_ixs, disallowed_choices, disallowed_choices_tree_ixs

        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e
    

    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune(self, num_to_prune, importance_dict, curr_conv_resource_calc: ConvResourceCalc, resource_limitation_dict=None):

        # Returns True if there are more to prune in the future, False if there are no more to prune.


        # Pruning works like this:
        """
        Ot of the not-disallowed kernels, we sort them by their importance,
        and keep pruning the least important ones until we have pruned num_to_prune of them.


        We figure out 2 types of disallowments:
        - general disallowments (like a limit on the amount of kernels)
        - choice disallowments (the ones we can't choose to prune, but they can be pruned as a consequence of another layer being chosen)
        (This happens throug kernel_connection_fn (architectures like ResNet and SegNet))

        Here we take out all the ones that are disallowed in any of these 2 categories.

        In prune_one_layer_recursive() we would then recursively prune all the layers that are inextricably connected to the pruned layer.
        But before we do that, we go and check if any of the ones that would be pruned are disallowed - if any of them shouldn't be pruned, just return False.

        And this is how pruning is done.
        """


        # TODO
        # This would be better if we had conv_resource_calc.module_tree_ix_2_all_children_conv_tree_ixs_list
        # We would only go through all tree_ixs once and record which conv_tree_ixs are disallowed.
        # Now we go through all tree_ixs once, and then for all conv tree ix-s we go to the root. Not as nice.


        try:



            disallowed_directly, disallowed_tree_ixs, _, disallowed_choices_tree_ixs = self._get_disallowed_tree_ixs(curr_conv_resource_calc)

            print(f"disallowed_directly: {disallowed_directly}")
            print(f"disallowed_tree_ixs: {disallowed_tree_ixs}")


            choice_disallowments = disallowed_tree_ixs.union(disallowed_choices_tree_ixs)















            # We plan to sort the kernels by their importance

            # importance_dict is:  {tree_ix: 1d_tensor_of_importances}
            # We have to convert these 1d tensors to
            # lists of tuples of the form (tree_ix, real_kernel_ix, importance).
            # Concat all these lists. Then sort by the importance.

            sortable_list = []
            for tree_ix, importance_tensor in importance_dict.items():
                
                # Here we prevent the disallowed ones from being pruned.
                if tree_ix in choice_disallowments:
                    continue

                for real_kernel_ix, importance in enumerate(importance_tensor):
                    sortable_list.append((tree_ix, real_kernel_ix, float(importance)))
            


        
            sortable_list.sort(key=lambda x: x[2])


            
            
            print(10*"-")
            print(f"sortable_list[:5]: {sortable_list[:5]}")




            

            if len(sortable_list) == 0:
                print("No more to prune!!!")
                return False
            

            

            # ----------TAKE REAL_KERNEL_IXS TO THE INITIAL_KERNEL_IXS----------
            # We have to take to_prune to the initial kernel_ixs, because with the current mechanism of pruning num_to_prune at once
            # once we prune a kernel, if we then also go prune a kernel in the same layer, the kernel ix is wrong.
            # to_prune = [(tree_ix, self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix]) for tree_ix, real_kernel_ix, _ in sortable_list]
            to_prune = []
            for tree_ix, real_kernel_ix, _ in sortable_list:
                try:
                    temp = (tree_ix, self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix])
                    to_prune.append(temp)
                except Exception as e:
                    print(f"tree_ix: {tree_ix}, real_kernel_ix: {real_kernel_ix}")
                    print(f"temp: {temp}")
                    raise e



            num_pruned = 0
            for to_prune_elem in to_prune:

                if resource_limitation_dict is not None:
                    # after every call of prune_one_layer_recursive, the resources are recalculated. So we only need to read them like so:
                    curr_res = curr_conv_resource_calc.get_resource_of_whole_model(resource_limitation_dict["resource_name"])
                    if curr_res < resource_limitation_dict["goal_resource_value"]:
                        return True # we simply conteptually have to say there are more to prune, so we don't break everything.


                curr_to_prune_elem = (to_prune_elem[0], self.tree_ix_2_list_of_initial_kernel_ixs[to_prune_elem[0]].index(to_prune_elem[1]))

                # this also does curr_conv_resource_calc.calculate_resources(self.input_example)
                succeeded = self.prune_one_layer_recursive(curr_to_prune_elem, curr_conv_resource_calc)
                if succeeded:
                    num_pruned += 1
                if num_pruned >= num_to_prune:
                    return True # because there are more to prune
                
            if num_pruned == 0:
                print("No more to prune!!!")
            else:
                print(f"Pruned {num_pruned} layers. No more to prune.")
            
            return False # because there are no more to prune
        


        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e

        


    @py_log.autolog(passed_logger=MY_LOGGER)
    def prune_one_layer_recursive(self, to_prune, curr_conv_resource_calc: ConvResourceCalc, check_if_disallowed=True):
        
        # to_prune is (tree_ix, real_kernel_ix)


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


        try:

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
                _, disallowed_tree_ixs, _, _ = self._get_disallowed_tree_ixs(curr_conv_resource_calc)

                # py_log.log_locals(passed_logger=MY_LOGGER)

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
                self.prune_current_layer(to_prune[0], to_prune[1], tree_ix_2_module)
            except KeyError as e:
                print(f"Pruning {to_prune} failed.")
                # for_log_locals = self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]]
                # py_log.log_locals(passed_logger=MY_LOGGER)
                raise e
            


            self.pruning_logs["following"].append([(self.conv_tree_ixs.index(i), j, k) for i,j,k in following_to_prune])
            # on those the method of next to be pruned (its a different pruning method)
            for tree_ix, real_input_slice_ix, _ in following_to_prune:

                self.prune_following_layer(tree_ix, real_input_slice_ix, tree_ix_2_module)
            


            



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
                self.prune_one_layer_recursive((tree_ix, kernel_ix), curr_conv_resource_calc, check_if_disallowed=False)
            
            # When the call of the original pruning call is finished, the dimensions across the network are corrected and they line up correctly.
            # We have to do this recalculation, so that when this function gets called next time, the resources are correct and we correctly know what is allowed to be pruned.
            if check_if_disallowed:
                curr_conv_resource_calc.calculate_resources(self.input_example)
            

            return True
        
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e

