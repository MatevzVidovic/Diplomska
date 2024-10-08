


import torch
import ConvResourceCalc


import torch.nn.utils.prune as prune


class pruner:



    def __init__(self, FLOPS_min_resource_percentage, weights_min_resource_percentage, initial_conv_resource_calc: ConvResourceCalc, connection_lambda, inextricable_connection_lambda, filter_importance_lambda, conv_tree_ixs, lowest_level_modules):
        self.initial_conv_resource_calc = initial_conv_resource_calc
        self.FLOPS_min_resource_percentage_dict = FLOPS_min_resource_percentage.min_resource_percentage_dict
        self.weights_min_resource_percentage_dict = weights_min_resource_percentage.min_resource_percentage_dict
        self.connection_lambda = connection_lambda
        self.inextricable_connection_lambda = inextricable_connection_lambda
        self.filter_importance_lambda = filter_importance_lambda
        self.conv_tree_ixs = conv_tree_ixs
        self.lowest_level_modules = lowest_level_modules


        self.tree_ix_2_list_of_initial_kernel_ixs = {}
        for tree_ix in conv_tree_ixs:
            kernel_num = initial_conv_resource_calc.module_tree_ix_2_weights_dimensions[tree_ix][1] # 1. dim is the number of filters of this layer
            self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix] = list(range(kernel_num))
        

        self.tree_ix_2_list_of_initial_input_slice_ixs = {}
        for tree_ix in conv_tree_ixs:
            input_slice_num = initial_conv_resource_calc.module_tree_ix_2_weights_dimensions[tree_ix][0] # 0. dim is the number input size of this layer
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
        self.tree_ix_2_list_of_initial_kernel_ixs[tree_ix].remove(real_kernel_ix)


        inextricable_following_to_prune = self.inextricable_connection_lambda(tree_ix, real_kernel_ix, self.conv_tree_ixs, self.lowest_level_modules)
        for tree_ix, real_kernel_ix in inextricable_following_to_prune:
            self.prune_inextricably_following_layer(tree_ix, real_kernel_ix, wrapper_model, tree_ix_2_module)




        # prune.random_unstructured(module, name="weight", amount=0.3)

        # # zadnje poglavje tega si bo treba prebrat:
        # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#pruning-multiple-parameters-in-a-model

        # Pač ta class bo treba razširit in potem njegove metode uporabljat.
        # Veliko bolje, kot da sam delam surgery po utežeh.
        # Samo ta compute_mask() bo treba spremenit. Pa pogledat je treba kaj je ta global, structured, unstructured
        # Pomoje bo treba imet samo pač structured, ker je local, ampak zdajle ne vem zadosti.
        # pass

    def prune_inextricably_following_layer(self, tree_ix, input_slice_ix, wrapper_model, tree_ix_2_module):

        # get the module
        module = tree_ix_2_module[tree_ix]
        
        if type(module) == torch.nn.BatchNorm2d:

            # For the comments of the lines, go look at pruner.prune_current_filter()

            old_weights = module.weight.data
            new_weights = torch.cat([old_weights[:input_slice_ix], old_weights[input_slice_ix+1:]], dim=0)
            module.weight.data = new_weights
            module.weight.grad = None

            old_bias = module.bias.data
            new_bias = torch.cat([old_bias[:input_slice_ix], old_bias[input_slice_ix+1:]], dim=0)
            module.bias.data = new_bias
            module.bias.grad = None

            old_running_mean = module.running_mean.data
            new_running_mean = torch.cat([old_running_mean[:input_slice_ix], old_running_mean[input_slice_ix+1:]], dim=0)
            module.running_mean.data = new_running_mean
            module.running_mean.grad = None

            old_running_var = module.running_var.data
            new_running_var = torch.cat([old_running_var[:input_slice_ix], old_running_var[input_slice_ix+1:]], dim=0)
            module.running_var.data = new_running_var
            module.running_var.grad = None
 









    def prune_following_layer(self, tree_ix, input_slice_ix, wrapper_model, tree_ix_2_module):


        # get the module
        module = tree_ix_2_module[tree_ix]






        # pruning convolutional layers:

        old_weights = module.weight.data

        # print(f"old_weights.shape: {old_weights.shape}")
        # input()

        # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
        # Here input_slice_ix is refering to the input channels (each channel is the result of one kernel. Which one of those are we pruning.)
        new_weights = torch.cat([old_weights[:, :input_slice_ix, :, :], old_weights[:, input_slice_ix+1:, :, :]], dim=1)
        module.weight.data = new_weights # torch.nn.Parameter(new_weights)

        # apparently better to clear out between two forward-backward passes
        # I suspect it would work without this too
        module.weight.grad = None


        self.tree_ix_2_list_of_initial_input_slice_ixs[tree_ix].remove(input_slice_ix)




    """

def disable_filter(device, model, name_index):
    #logger.write('disabling filter in layer {0}'.format(name_index))
    n_parameters_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    name, index = get_parameter_name_and_index_from_activations_dict_key(name_index)
    block_name, _, layer_name = name.rpartition('.')
    block = op.attrgetter(block_name)(model)
    layer = getattr(block, layer_name)

    new_conv = \
        torch.nn.Conv2d(in_channels=layer.in_channels, \
                        out_channels=layer.out_channels - 1,
                        kernel_size=layer.kernel_size, \
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=1,  # conv.groups,
                        bias=True
                        )

    if (layer.groups != 1):
        print('MAYBE THIS IS WRONG GROUPS != 1')
    layer.out_channels -= 1

    old_weights = layer.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[: index, :, :, :] = old_weights[: index, :, :, :]
    new_weights[index:, :, :, :] = old_weights[index + 1:, :, :, :]

    # conv.weight.data = torch.from_numpy(new_weights).to(self.device)
    layer.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(device))
    layer.weight.grad = None

    if layer.bias is not None:
        bias_numpy = layer.bias.data.cpu().numpy()
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:index] = bias_numpy[:index]
        bias[index:] = bias_numpy[index + 1:]
        # conv.bias.data = torch.from_numpy(bias).to(self.device)
        layer.bias = torch.nn.Parameter(torch.from_numpy(bias).to(device))
        layer.bias.grad = None


    # ALSO: change activations sum for this conv layer # todo: i dont update activations (only sum)
    layer_activations_sum = getattr(block, layer_name + '_activations_sum') # vektor dolzine toliko kolikor je filtrov, za vsak filter je ena stevilka
    layer_activations_sum = torch.cat([layer_activations_sum[0:index], layer_activations_sum[index+1:]])
    setattr(block, layer_name + '_activations_sum', torch.nn.Parameter(layer_activations_sum.to(device), requires_grad=False))

    layer_index = _get_layer_index(name, model)
    # prune next bn if nedded
    _prune_next_bn_if_needed(layer_index, index, index, 1, device, model)

    # surgery on chained convolution layers
    next_conv_idx_list = _get_next_conv_id_list_recursive(layer_index, model)
    for next_conv_id in next_conv_idx_list:
        #print(next_conv_id)
        _prune_next_layer(next_conv_id, index, index, 1, device, model)

    n_parameters_after_pruning = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return n_parameters_before - n_parameters_after_pruning


def _prune_next_layer(next_conv_i, filters_begin, filters_end, pruned_filters, device, model):
    logger.write('Additionally pruning (next layer) conv with layer_id ' + str(next_conv_i))
    assert filters_begin == filters_end
    next_conv, block, layer_name = _layer_index_to_conv(next_conv_i, model)

    next_new_conv = \
        torch.nn.Conv2d(in_channels=next_conv.in_channels - pruned_filters, \
                        out_channels=next_conv.out_channels, \
                        kernel_size=next_conv.kernel_size, \
                        stride=next_conv.stride,
                        padding=next_conv.padding,
                        dilation=next_conv.dilation,
                        groups=1,  # next_conv.groups,
                        bias=True
                        )  # next_conv.bias)
    next_conv.in_channels -= pruned_filters

    old_weights = next_conv.weight.data.cpu().numpy()
    new_weights = next_new_conv.weight.data.cpu().numpy()

    new_weights[:, : filters_begin, :, :] = old_weights[:, : filters_begin, :, :]
    new_weights[:, filters_begin:, :, :] = old_weights[:, filters_end + 1:, :, :]

    next_conv.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(device))
    #        next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
    next_conv.weight.grad = None







    # out conv: ne popravljam aktivacij, ker jih nimam za to konvolucijo
    model_name = type(model).__name__.lower()
    if 'densenet' in model_name and next_conv_i == 41 or 'unet' in model_name and next_conv_i == 18:
        return

    index = filters_begin

    # ALSO: change activations sum for this conv layer # todo: i dont update activations
    layer_activations_sum = getattr(block,
                                    layer_name + '_activations_sum')  # vektor dolzine toliko kolikor je filtrov, za vsak filter je ena stevilka
    layer_activations_sum = torch.cat([layer_activations_sum[0:index], layer_activations_sum[index + 1:]])
    setattr(block, layer_name + '_activations_sum',
            torch.nn.Parameter(layer_activations_sum.to(device), requires_grad=False))

    """




    
    def prune(self, activations, curr_conv_resource_calc, wrapper_model):

        # Prepare the activations - concatenate the batches of repetitions

        # First we evaluate all the filters
        importance_dict = self.filter_importance_lambda(activations, self.conv_tree_ixs)

        # print(20*"\n")
        # print(activations[self.conv_tree_ixs[0]])
        # print(5*"\n")
        # print(importance_dict[self.conv_tree_ixs[0]])
        # print(importance_dict[self.conv_tree_ixs[0]].shape)
        # print(len(activations[self.conv_tree_ixs[0]]))
        # print(activations[self.conv_tree_ixs[0]][0].shape)

        
        # Then we sort them

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
            if curr_conv_resource_calc.module_tree_ixs_2_flops_dict[tree_ix] <= self.FLOPS_min_resource_percentage_dict[tree_ix]:
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
                parents = curr_conv_resource_calc.module_tree_ixs_2_all_parents_to_root_tree_ix_list[tree_ix]
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

        tree_ix_2_module = curr_conv_resource_calc.module_tree_ixs_2_modules_themselves

        # method for pruning current
        self.prune_current_layer(to_prune[0], to_prune[1], wrapper_model, tree_ix_2_module)


        # to find the next (tree_ix, input_slice_ix) to prune, we have to go through the connection lambda
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



        

