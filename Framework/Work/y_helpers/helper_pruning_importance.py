



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

from model_wrapper import ModelWrapper













# When each batch is processed, the averaging_objects function is called.
# Here you define how you would like to create your averaging objects through one epoch of training.
# This function shows how we would like to update our average of the activations (outputs)
# for the convolutional layers (because in the background this is only set for convolutional layers).
# At each iteration the mean is corrects so far. So at the end the mean is also correct.
# It is better to train with larger batch sizes so numerical errors of the iterative mean calculation are smaller.

# Proof:
# The first mean is correct so far. It is avg_0 = \sum x_i / n_0 where n_0 is the number of elements of the 0-th iteration.
# by the same logic, avg_1 is also correct (the average of just the next batch).
# The second mean avg_{1,2} is (n_0 * avg _0 + n_1 * avg_1) / (n_0 + n_1) = 
# (n_0 * (\sum x_i / n_0) + n_1 * (\sum x_j / n_1)) / (n_0 + n_1) =
# ( \sum x_i + \sum x_j ) / (n_0 + n_1)
# # Which is the correct mean of all the elements. By induction, the same logic applies to all iterations.  

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IF USING INPUT OR MODULE WEIGHTS, YOU HAVE TO DETACH THEM!!!!!
# Also, input is a tuple, so you have to figure out what it really is first - I haven't looked into it.
# The output has already been detached, so we don't need to worry about backpropagation.
# You can do .detach() again, which won't change anything, it's idempotent.
# If they weren't detached, they remain in the computational graph and keep being in the gradient calculation during loss.backward().
# Because of pruning, this shows an error like so:
#  File "/home/matevzvidovic/Desktop/Diplomska/Prototip/Delo/TrainingWrapper.py", line 424, in train
#     loss.backward()
#   File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/_tensor.py", line 522, in backward
#     torch.autograd.backward(
#   File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/autograd/__init__.py", line 266, in backward
#     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# RuntimeError: Function ConvolutionBackward0 returned an invalid gradient at index 1 - got [128, 255, 3, 3] but expected shape compatible with [128, 256, 3, 3]

# We would like to also use weights in our importance calculation.
# The easiest and conceptually best place to put them is in the averaging function (outside of making their own function).
# It doesn't make sense to average them, so we would just save them when the first average is made.


INITIAL_AVG_OBJECT = (0, None, None)
def averaging_function(module, input, output, prev_avg_object):
    
    batch_size = output.shape[0]
    batch_mean = output.mean(dim=(0))

    if prev_avg_object[1] is None:
        new_avg_object = (batch_size, batch_mean, module.weight.data.detach().clone())
        return new_avg_object

    new_avg_object = (prev_avg_object[0] + batch_size, 
                      (prev_avg_object[0] * prev_avg_object[1] + batch_size * batch_mean) / (prev_avg_object[0] + batch_size),
                        prev_avg_object[2])

    return new_avg_object 


# averaging_mechanism = {
#     "initial_averaging_object" : INITIAL_AVG_OBJECT,
#     "averaging_function" : averaging_function
# }



# An additional function could be applied in between the averaging function and the importance function.
# If we were, for example, interested in a specific interaction between the active outputs (not the averaged ones)
# with our averaging object. For example, to calculate the correlation between the output and our average activations.
# Then the averaging function would be applied in the first pass through the network and we would make our averaging objects.
# Then this middle function would be used and we would calculate our batch_importances (correlations) for each batch.
# Then the final importance function we see below us would only be used to combine these batch_importances.
# For example to average them or to sum them.
# But this is not currently implemented.





def IPAD_kernel_importance_fn_generator(L1_ADC_weight):
    assert L1_ADC_weight >= 0 and L1_ADC_weight <= 1, "L1_ADC_weight must be between 0 and 1."
    
    
    def IPAD_kernel_importance_fn(averaging_objects: dict, conv_tree_ixs):
        # Returns dict tree_ix_2_list_of_kernel_importances
        # The ix-th importance is for the kernel currently on the ix-th place.
        # To convert this ix to the initial unpruned models kernel ix, use the pruner's
        # state of active kernels.

        tree_ix_2_kernels_importances = {}
        for tree_ix in conv_tree_ixs:

            kernels_average_activation = averaging_objects[tree_ix][1]
            # print(kernels_average_activation.shape)
            # print(kernels_average_activation)
            overall_average_activation = kernels_average_activation.mean(dim=(0))
            # print(overall_average_activation)
            # print(overall_average_activation.shape)
            # print(overall_average_activation)
            h = kernels_average_activation.shape[1]
            w = kernels_average_activation.shape[2]
            diff = kernels_average_activation - overall_average_activation
            L1_ADC = torch.abs(diff).sum(dim=(1,2)) / (h*w)
            L2_ADC = (diff).pow(2).sum(dim=(1,2)).sqrt() / (h*w)
            kernels_importances = L1_ADC_weight * L1_ADC + (1 - L1_ADC_weight) * L2_ADC
            # print(f"L1_ADC: {L1_ADC}")
            # print(f"L2_ADC: {L2_ADC}")
            # print(kernels_importances.shape)
            # print(kernels_importances)

            tree_ix_2_kernels_importances[tree_ix] = kernels_importances
        
        
        return tree_ix_2_kernels_importances
        
    
    return IPAD_kernel_importance_fn




def weights_importance_fn_generator(L1_over_L2_alpha):
    assert L1_over_L2_alpha >= 0 and L1_over_L2_alpha <= 1, "L1_over_L2_alpha must be between 0 and 1."
    
    def weights_importance_fn(averaging_objects: dict, conv_tree_ixs):
        # Returns dict tree_ix_2_list_of_kernel_importances
        # The ix-th importance is for the kernel currently on the ix-th place.
        # To convert this ix to the initial unpruned models kernel ix, use the pruner's
        # state of active kernels.

        tree_ix_2_kernels_importances = {}
        for tree_ix in conv_tree_ixs:
            
            # [num_of_kernels, depth, h, w]
            kernels_weights = averaging_objects[tree_ix][2]
            overall_weights = kernels_weights.mean(dim=(0))
            d = kernels_weights.shape[1]
            h = kernels_weights.shape[2]
            w = kernels_weights.shape[3]
            L1 = torch.abs(kernels_weights - overall_weights).sum(dim=(1,2,3)) / (d*h*w)
            L2 = (kernels_weights - overall_weights).pow(2).sum(dim=(1,2,3)).sqrt() / (d*h*w)
            kernels_importances = L1_over_L2_alpha * L1 + (1 - L1_over_L2_alpha) * L2

            tree_ix_2_kernels_importances[tree_ix] = kernels_importances
        
        
        return tree_ix_2_kernels_importances
        
    
    return weights_importance_fn



def IPAD_and_weights(IPAD_over_weights_alpha, IPAD_L1_ADC_weight, weights_L1_over_L2_alpha):
    assert IPAD_over_weights_alpha >= 0 and IPAD_over_weights_alpha <= 1, "IPAD_over_weights_alpha must be between 0 and 1."

    IPAD_fn = IPAD_kernel_importance_fn_generator(IPAD_L1_ADC_weight)
    weights_fn = weights_importance_fn_generator(weights_L1_over_L2_alpha)

    def joined_imporance_fn(averaging_objects: dict, conv_tree_ixs):
        IPAD_importances = IPAD_fn(averaging_objects, conv_tree_ixs)
        weights_importances = weights_fn(averaging_objects, conv_tree_ixs)

        joined_importances = {}
        for tree_ix in conv_tree_ixs:
            joined_importances[tree_ix] = IPAD_over_weights_alpha * IPAD_importances[tree_ix] + (1 - IPAD_over_weights_alpha) * weights_importances[tree_ix]

        return joined_importances

    return joined_imporance_fn



def IPAD_and_weights_granular(IPAD1_weight, IPAD2_weight, L1_weight, L2_weight):
    
    eps = 1e-6
    weights_sum = IPAD1_weight + IPAD2_weight + L1_weight + L2_weight
    assert weights_sum > (1 - eps) and weights_sum < (1 + eps), "The weights must sum to 1."
    

    IPAD1_fn = IPAD_kernel_importance_fn_generator(1)
    IPAD2_fn = IPAD_kernel_importance_fn_generator(0)
    L1_fn = weights_importance_fn_generator(1)
    L2_fn = weights_importance_fn_generator(0)

    def joined_imporance_fn(averaging_objects: dict, conv_tree_ixs):
        IPAD1_importances = IPAD1_fn(averaging_objects, conv_tree_ixs)
        IPAD2_importances = IPAD2_fn(averaging_objects, conv_tree_ixs)
        L1_importances = L1_fn(averaging_objects, conv_tree_ixs)
        L2_importances = L2_fn(averaging_objects, conv_tree_ixs)

        joined_importances = {}
        for tree_ix in conv_tree_ixs:
            joined_importances[tree_ix] = IPAD1_weight * IPAD1_importances[tree_ix] + IPAD2_weight * IPAD2_importances[tree_ix] \
            + L1_weight * L1_importances[tree_ix] + L2_weight * L2_importances[tree_ix]

        return joined_importances

    return joined_imporance_fn


def random_pruning_importance_fn(averaging_objects: dict, conv_tree_ixs):
    tree_ix_2_kernel_importances = {}
    for tree_ix in conv_tree_ixs:
        num_of_kernels = averaging_objects[tree_ix][1].shape[0]
        kernel_importance = torch.rand(num_of_kernels)
        tree_ix_2_kernel_importances[tree_ix] = kernel_importance

    return tree_ix_2_kernel_importances



# Da imamo najmanjše importance v layerju, čigar curr_conv_ix (ix v conv_tree_ixs) je enak oziroma njabližje CURR_PRUNING_IX.
# Znotraj layerja pa imajo kernels v V shapeu - da se vedno na sredini prunea (saj uniform pruning bi bil, da vedno 0-tega prunaš.- Ampak mi ni všeč, da se vedno the edge one prunea. Raje da vedno the middle one.)
# Za posamezen layer določimo oddaljenost od trenutnega pruninga:
# curr_dist = abs(curr_conv_ix - CURR_PRUNING_IX)
# Naredi torej recimo, da kernel importances iz sredine proti robu rastejo med:
# curr_dist in curr_dist+1.

CURRENT_PRUNING_IX = 0
def uniform_random_pruning_importance_fn(averaging_objects: dict, conv_tree_ixs):

    global CURRENT_PRUNING_IX

    tree_ix_2_kernel_importances = {}
    for ix, tree_ix in enumerate(conv_tree_ixs):
        
        num_of_kernels = averaging_objects[tree_ix][1].shape[0]
        curr_dist = abs(ix - CURRENT_PRUNING_IX)

        middle_kernel_ix = num_of_kernels // 2
        ixs = torch.arange(num_of_kernels)
        kernel_distances = torch.abs(ixs - middle_kernel_ix)
        
        # should look sth like: [1.0, 0.97,...,0.0, 0.02, ... 1.0]
        base_importances = kernel_distances.float() / kernel_distances.max().float()
        # and now we put them in the right bracket based on distance of the layer from the current pruning ix
        final_importances = base_importances + curr_dist
        
        tree_ix_2_kernel_importances[tree_ix] = final_importances
    
    CURRENT_PRUNING_IX += 1
    if CURRENT_PRUNING_IX >= len(conv_tree_ixs):
        CURRENT_PRUNING_IX = 0


    return tree_ix_2_kernel_importances


























def set_averaging_objects_hooks(model_wrapper, initial_averaging_object, averaging_function, averaging_objects: dict, resource_calc, tree_ixs: list):
        
    
    def get_activation(tree_ix):
        
        def hook(module, input, output):
            
            detached_output = output.detach()

            if tree_ix not in averaging_objects:
                averaging_objects[tree_ix] = initial_averaging_object

            averaging_objects[tree_ix] = averaging_function(module, input, detached_output, averaging_objects[tree_ix])

        return hook

    tree_ix_2_hook_handle = {}
    for tree_ix in tree_ixs:
        module = resource_calc.module_tree_ix_2_module_itself[tree_ix]
        tree_ix_2_hook_handle[tree_ix] = module.register_forward_hook(get_activation(tree_ix))
    
    model_wrapper.tree_ix_2_hook_handle = tree_ix_2_hook_handle
    



def remove_hooks(model_wrapper):
    
    if model_wrapper.tree_ix_2_hook_handle is None:
        raise ValueError("In remove_hooks: model_wrapper.tree_ix_2_hook_handle is already None")
    
    for hook_handle in model_wrapper.tree_ix_2_hook_handle.values():
        hook_handle.remove()
    
    model_wrapper.tree_ix_2_hook_handle = None


