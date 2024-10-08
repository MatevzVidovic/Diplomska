


import torch
import ConvResourceCalc


import torch.nn.utils.prune as prune


class pruner:



    def __init__(self, FLOPS_min_resource_percentage, weights_min_resource_percentage, initial_conv_resource_calc: ConvResourceCalc, connection_lambda, inextricable_connection_lambda, conv_tree_ixs, batch_norm_ixs, lowest_level_modules):
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
        
        
    

    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1  # Target not found
    




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

            self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix].pop(real_input_slice_ix)
 









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


        self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix].pop(real_input_slice_ix)






    
    def prune(self, importance_dict, curr_conv_resource_calc, wrapper_model):

        

        
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
        disallowed_directly = set()
        for tree_ix in self.FLOPS_min_resource_percentage_dict:
            if curr_conv_resource_calc.module_tree_ix_2_flops_num[tree_ix] <= self.FLOPS_min_resource_percentage_dict[tree_ix]:
                disallowed_directly.add(tree_ix)

        for tree_ix in self.weights_min_resource_percentage_dict:
            if curr_conv_resource_calc.module_tree_ix_2_weights_num[tree_ix] <= self.weights_min_resource_percentage_dict[tree_ix]:
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


        # Then we pick the least important one which isn't disallowed
        to_prune = None
        for tree_ix, real_kernel_ix, importance in sortable_list:
            if tree_ix in disallowed_tree_ixs:
                continue
            to_prune = (tree_ix, real_kernel_ix)
            break

        if to_prune is None:
            print("No more to prune.")
            return
        



        # And we prune it.

        tree_ix_2_module = curr_conv_resource_calc.module_tree_ix_2_module_itself

        # method for pruning current
        self.prune_current_layer(to_prune[0], to_prune[1], wrapper_model, tree_ix_2_module)


        # to find the next (tree_ix, real_input_slice_ix) to prune, we have to go through the connection lambda
        # however, the lambda works on the indexes of the initial unpruned model.
        # So, we have to find out the initial_kernel_ix of the real kernel_ix.
        # Then put that in the lambda to get the goal's initial_input_slice_ix.
        # Then change that to the real_input_slice_ix (the ix of where the initial slice is right now).

        initial_kernel_ix = self.tree_ix_2_list_of_initial_kernel_ixs[to_prune[0]][to_prune[1]]
        following_to_prune = self.connection_lambda(to_prune[0], initial_kernel_ix, self.conv_tree_ixs, self.lowest_level_modules)
        # on those the method of next to be pruned (its a different pruning method)
        for tree_ix, initial_input_slice_ix in following_to_prune:
            real_input_slice_ix = self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix].index(initial_input_slice_ix) # could do self.binary search for speed, but it is for later
            self.prune_following_layer(tree_ix, real_input_slice_ix, wrapper_model, tree_ix_2_module)
        
        print(f"Pruned {to_prune}")
        print(f"Pruned {following_to_prune}")




        

