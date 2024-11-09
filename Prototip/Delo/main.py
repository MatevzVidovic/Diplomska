

import os
import logging
import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)




import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse

from unet import UNet

# from dataset import IrisDataset, transform
from my_dataset import IrisDataset, transform

from min_resource_percentage import min_resource_percentage
from ModelWrapper import ModelWrapper

from training_support import *




# save_path = os.path.join(os.path.dirname(__file__), "UNet")
save_path = os.path.join(".", "UNet")

main_save_path = os.path.join(save_path, "saved_main")




learning_parameters = {
    "learning_rate" : 1e-3,
    "loss_fn" : nn.CrossEntropyLoss(),
    "optimizer_class" : torch.optim.SGD
}

dataloading_args = {


    "testrun" : True,
   

    # Image resize setting - don't know what it should be.
    "width" : 128,
    "height" : 128,
    
    # iris dataset params
    "path_to_sclera_data" : "./sclera_data",
    "transform" : transform,
    "n_classes" : 2,

    # DataLoader params
    # Could have separate "train_batch_size" and "eval_batch_size" (for val and test)
    #  since val and test use torch.no_grad() and therefore use less memory. 
    "batch_size" : 16,
    "shuffle" : False, # TODO shuffle??
    "num_workers" : 1,
}


def get_data_loaders(**dataloading_args):
    
    data_path = dataloading_args["path_to_sclera_data"]
    # n_classes = 4 if 'sip' in args.dataset.lower() else 2

    print('path to file: ' + str(data_path))

    train_dataset = IrisDataset(filepath=data_path, split='train', **dataloading_args)
    valid_dataset = IrisDataset(filepath=data_path, split='val', **dataloading_args)
    test_dataset = IrisDataset(filepath=data_path, split='test', **dataloading_args)

    trainloader = DataLoader(train_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    validloader = DataLoader(valid_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"])
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # I'm not sure why we're dropping last, but okay.

    # Actually, no. Let's not drop last.
    # it makes no sense. i think this might have been done because the IPAD fn was done only on the last batch, and so that
    # batch needed to be big.


    print('train dataset len: ' + str(train_dataset.__len__()))
    print('val dataset len: ' + str(valid_dataset.__len__()))
    print('test dataset len: ' + str(test_dataset.__len__()))

    print('train dataloader num of batches: ' + str(trainloader.__len__()))
    print('val dataloader num of batches: ' + str(validloader.__len__()))
    print('test dataloader num of batches: ' + str(testloader.__len__()))

    
    return trainloader, validloader, testloader






train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(**dataloading_args)# 

dataloader_dict = {
    "train" : train_dataloader,
    "validation" : valid_dataloader,
    "test" : test_dataloader,
}



model_parameters = {
    # layer sizes
    "n_channels" : 3,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }

INPUT_EXAMPLE = torch.randn(1, 3, 128, 128)








# Go see model graph to help you construct these connection functions.
# model_wrapper.model_graph()




def unet_tree_ix_2_skip_connection_start(tree_ix, conv_tree_ixs):
    #    tree_ix -> skip_conn_starting_index

    # It could be done programatically, however:
    # Assuming the layers that have skip connections have only one source of them,
    # we could calculate how many inputs come from the previous layer.
    # That is then the starting ix of skip connections.

    # To make this function, go look in the drawn matplotlib graph.
    # On the upstream, just look at the convolution's weight dimensions.
    # They are: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
    # (output_dimensions - input_dimensions) is the ix of the first skip connection



    # Oh, I see. This is easily programmable.
    # Just use "initial_conv_resource_calc.pkl" and use 
    # (output_dimensions - input_dimensions) where output_dimensions > input_dimensions.
    # And that's it haha.

    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)

        if conv_ix == 16:
            
            return 64
        elif conv_ix == 14:
            
            return 128
        elif conv_ix == 12:
            
            return 256
        elif conv_ix == 10:
            
            return 512


    else:
        
        return None
    





"""
THIS HERE IS THE START OF BUILDING A CONNECTION fn
based on the _get_next_conv_id_list_recursive()
It is very early stage.
"""



def unet_input_slice_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, initial_kernel_ix) -> [(goal_tree_ix_1, goal_initial_input_slice_ix_1), (goal_tree_ix_2, goal_initial_input_slice_ix_2),...]

    # TL;DR -  for skip connections, where the input channels need to be pruned, because the output channels of this layer were pruned
    
    # This functions takes the tree_ix and the ix of where the kernel we are concerned with was in the model initially (before pruning).
    # And it returns a list of tuples giving the following modules tree_ixs and the input_slice_ix
    # (where the effect of the above-mentioned kernel is in the input tensor) in the initial model (before pruning).


    conn_destinations = []

    # we kind of only care about convolutional modules.
    # We just need to prune there (and possibly something with the batch norm layer)
    # So it would make sense to transform the tree_ix to the ordinal number of 
    # the convolutional module, and work with that ix instead.

    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)
        conn_destinations.append((conv_tree_ixs[conv_ix+1], kernel_ix))

    # We made it so that for conv layers who receive as input the previous layer and a skip connection
    # the first inpute slices are of the previous layer. This makes the line above as elegant as it is.
    # We will, however, have to deal with more trouble with skip connections. 

    
    # (however, we included in a different way, because it is more elegant and makes more sense that way) 
    # For the more general option (e.g. to include pruning of some other affected layers)
    # we can instead work with "lowest_level_modules" indexes.
    # These are modules that appear the lowest in the tree, and are the ones that actually 
    # do the work. Data passes through them. They arent just composites of less complex modules.
    # They are the actual building blocks.

    LLM_ix = None
    if tree_ix in lowest_level_modules:
        LLM_ix = lowest_level_modules.index(tree_ix)




    # We already handled the regular connections for convolutional networks.
    # Now, here come skip connections.
    # For explanation, look at the graphic in the original U-net paper.
    
    # We have to know where the skip connections start.
    # What real index is the zeroth index of the skip connections for the goal layer?
    # In this way we can then use the tree_ix to get the base ix.

    # For this, we will for now create a second function where we hardcode this.
    # It could be done programatically, however:
    # Assuming the layers that have skip connections have only one source of them,
    # we could calculate how many inputs come from the previous layer.
    # That is then the starting ix of skip connections.

    # To do this, we look at the code where the skip connections of the model are defined:
    # def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits
    
    # We then look at the graphic of our network. We see that the inc block and first three down blocks create skip connections.
    # Therefore the last (second) convolution in those blocks will be senging the skip connection forward.
    # This is how we identify the particular convolutional modules (LLMs) that are involved in skip connections.
    

    # if conv_ix in [1, 3, 5, 7]:
    
    goal_conv_ix = None
    if conv_ix == 1:
        goal_conv_ix = 16
    elif conv_ix == 3:
        goal_conv_ix = 14
    elif conv_ix == 5:
        goal_conv_ix = 12
    elif conv_ix == 7:
        goal_conv_ix = 10
    
    if goal_conv_ix is not None:
        goal_input_slice_ix = kernel_ix + unet_tree_ix_2_skip_connection_start(conv_tree_ixs[goal_conv_ix], conv_tree_ixs)
        conn_destinations.append((conv_tree_ixs[goal_conv_ix], goal_input_slice_ix))

    # outc has no next convolution
    if conv_ix == 18:
        conn_destinations = []
    
    
    return conn_destinations


    

    # if idx == 6:
    #     next_idxs_list.append



    # # output is: [(goal_tree_ix_1, goal_input_slice_ix_1), (goal_tree_ix_2, goal_input_slice_ix_2),...] 
    #         # Output of conv2 in each down block also goes to conv1 in corresponding up block
    # if layer_index == 1:
    #     next_conv_idx = [2, 16]
    # elif layer_index == 3:
    #     next_conv_idx = [4, 14]
    # elif layer_index == 5:
    #     next_conv_idx = [6, 12]
    # elif layer_index == 7:
    #     next_conv_idx = [8, 10]
    # # outc has no next convolution
    # elif layer_index >= 18:
    #     next_conv_idx = []
    # # Every other convolution output just goes to the next one
    # else:
    #     next_conv_idx = [layer_index + 1]






def unet_kernel_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, real_kernel_ix) -> [(goal_tree_ix_1, goal_real_kernel_ix_1), (goal_tree_ix_2, goal_real_kernel_ix_2),...]
    
    # This functions takes the tree_ix and the ix of where the kernel we are concerned with was in the model RIGHT NOW, NOT INITIALLY.
    # And it returns a list of tuples giving the tree_ixs and "kernel_ixs" in the model RIGHT NOW, NOT INITIALLY.
    # for layers which are inextricably linked with the convolutional layer.

    # Meant for batchnorm and special cases.

    # Inextricably linked are in direct connection with the conv's current (not intitial) kernel_ix, so they don't need the more complex fn.
    # We could have treated them in the regular way (through initial ixs), but this way is better,
    # because, in the pruner, we don't need to keep track of the initial ixs (although we do anyways for accounting reasons).
    # Also it's simpler and conceptually makes more sense - which is the main reason.

    # The batchnorm is such a layer - for it, the "kernel_ix" isn't really a kernel ix.
    # It is, however, the position we need to affect due to pruning the kernel_ix in the convolutional layer.
    # There are possibly more such layers and more types of such layers, so we made this function more general.
    
    conn_destinations = []

    LLM_ix = None
    if tree_ix in lowest_level_modules:
        LLM_ix = lowest_level_modules.index(tree_ix)


    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)
    
        # out.conv doesn't have a batchnorm after it.
        if conv_ix < 18:
            conn_destinations.append((lowest_level_modules[LLM_ix+1], kernel_ix))


    # for batchnorm, conn_destinations is simply empty
    

    
    return conn_destinations




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


INITIAL_AVG_OBJECT = (0, None)
def averaging_function(module, input, output, prev_avg_object):
    
    batch_size = output.shape[0]
    batch_mean = output.mean(dim=(0))

    if prev_avg_object[1] is None:
        new_avg_object = (batch_size, batch_mean)
        return new_avg_object

    new_avg_object = (prev_avg_object[0] + batch_size, 
                      (prev_avg_object[0] * prev_avg_object[1] + batch_size * batch_mean) / (prev_avg_object[0] + batch_size))

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
    assert L1_ADC_weight > 0 and L1_ADC_weight < 1, "L1_ADC_weight must be between 0 and 1."
    
    
    def IPAD_kernel_importance_fn(averaging_objects: dict, conv_tree_ixs):
        # Returns dict tree_ix_2_list_of_kernel_importances
        # The ix-th importance is for the kernel currently on the ix-th place.
        # To convert this ix to the initial unpruned models kernel ix, use the pruner's
        # state of active kernels.

        tree_ix_2_kernel_importances = {}
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
            L1_ADC = torch.abs(kernels_average_activation - overall_average_activation).sum(dim=(1,2)) / (h*w)
            L2_ADC = (kernels_average_activation - overall_average_activation).pow(2).sum(dim=(1,2)).sqrt() / (h*w)
            kernel_importance = L1_ADC_weight * L1_ADC + (1 - L1_ADC_weight) * L2_ADC
            # print(f"L1_ADC: {L1_ADC}")
            # print(f"L2_ADC: {L2_ADC}")
            # print(kernel_importance.shape)
            # print(kernel_importance)

            tree_ix_2_kernel_importances[tree_ix] = kernel_importance
        
        
        return tree_ix_2_kernel_importances
        
    
    return IPAD_kernel_importance_fn
        

IMPORTANCE_FN = IPAD_kernel_importance_fn_generator(0.5)



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


def get_importance_dict(model_wrapper: ModelWrapper):

    model_wrapper.averaging_objects = {}
    set_averaging_objects_hooks(model_wrapper, INITIAL_AVG_OBJECT, averaging_function, model_wrapper.averaging_objects, model_wrapper.resource_calc, model_wrapper.conv_tree_ixs)

    model_wrapper.epoch_pass()

    # pruner needs the current state of model resources to know which modules shouldn't be pruned anymore
    model_wrapper.resource_calc.calculate_resources(model_wrapper.input_example)

    importance_dict = IMPORTANCE_FN(model_wrapper.averaging_objects, model_wrapper.conv_tree_ixs)
    remove_hooks(model_wrapper)
    model_wrapper.averaging_objects = {}

    return importance_dict




if __name__ == "__main__":

    
    model_wrapper = ModelWrapper(UNet, model_parameters, dataloader_dict, learning_parameters, INPUT_EXAMPLE, save_path)





    tree_ix_2_name = model_wrapper.get_tree_ix_2_name()

    # Go see model graph to help you choose the right layers to prune.
    # model_wrapper.model_graph()

    # If you change FLOPS_min_res_percents and weights_min_res_percents between runnings of main, 
    # the new onew will be used. So you can have an effect on your training by doing this.

    FLOPS_min_res_percents = min_resource_percentage(tree_ix_2_name)
    FLOPS_min_res_percents.set_by_name("Conv2d", 0.5)

    tree_ix_2_percentage_dict = {
        (0,) : 0.2    # This will obviously have no effect, since all convolutional layers are capped at 0.5. It is simply to show an example.
    }
    FLOPS_min_res_percents.set_by_tree_ix_dict(tree_ix_2_percentage_dict)


    weights_min_res_percents = min_resource_percentage(tree_ix_2_name)
    weights_min_res_percents.set_by_name("Conv2d", 0.2)

    model_wrapper.initialize_pruning(get_importance_dict, unet_input_slice_connection_fn, unet_kernel_connection_fn, FLOPS_min_res_percents, weights_min_res_percents)



    model_wrapper.training_wrapper.test_showcase()








    def validation_stop(val_errors):
        # returns True when you should stop

        if len(val_errors) >= 2:
            return True
        
        if len(val_errors) < 3:
            return False
        
        if len(val_errors) >= 25:
            return True
        
        returner = val_errors[-1] > val_errors[-2] and val_errors[-1] > val_errors[-3]

        # if previous metric doesn't say we should return, we also go check another metric:
        # if the current validation error is higher than either of the 4. and 5. back
        # we should stop. Because it means we are not improving.
        if not returner and len(val_errors) >= 5:
            returner = val_errors[-1] > val_errors[-4] or val_errors[-1] > val_errors[-5]
            
        return returner




    parser = argparse.ArgumentParser(description="Process an optional positional argument.")

    # Add the optional positional arguments
    parser.add_argument('iter_possible_stop', nargs='?', type=int, default=1e9,
                        help='An optional positional argument with a default value of 1e9')
    
    # These store True if the flag is present and False otherwise.
    # Watch out with argparse and bool fields - they are always True if you give the arg a nonempty string.
    # So --pbop False would still give True to the pbop field.
    # This is why they are implemented this way now.
    parser.add_argument('-v', '--validation_phase', action='store_true',
                        help='If present, enables validation (automatic pruning) phase')
    parser.add_argument('--pbop', action='store_true',
                        help='Prune by original percent, otherwise by number of filters')
    

    # Add the optional arguments
    # setting error_ix: ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)
    parser.add_argument('--e_ix', type=int, default=3,
                        help='ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)')
    parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations')
    parser.add_argument('--map', type=int, default=1e9, help='Max auto prunings')
    parser.add_argument('--nept', type=int, default=1,
                        help='Number of epochs per training iteration')
    parser.add_argument('--nftp', type=int, default=1,
                        help='Number of filters to prune in one pruning')
    parser.add_argument('--rn', type=str, default="flops_num", help='Resource name to prune by')
    parser.add_argument('--ptp', type=float, default=0.1, help='Percent to prune.')

    args = parser.parse_args()

    is_val_ph = args.validation_phase
    iter_possible_stop = args.iter_possible_stop

    err_ix = args.e_ix
    max_train_iters = args.mti
    max_auto_prunings = args.map
    num_ep_per_iter = args.nept

    prune_by_original_percent = args.pbop
    num_to_prune = args.nftp
    resource_name = args.rn
    percent_to_prune = args.ptp

    pruning_kwargs = {
        "prune_by_original_percent": prune_by_original_percent,
        "num_of_prunes": num_to_prune,
        "resource_name": resource_name,
        "original_percent_to_prune": percent_to_prune
    }

    print(f"Validation phase: {is_val_ph}")
    print(args)
    
    train_automatically(model_wrapper, main_save_path, validation_stop, max_training_iters=max_train_iters, max_auto_prunings=max_auto_prunings, train_iter_possible_stop=iter_possible_stop, validation_phase=is_val_ph, error_ix=err_ix,
                         num_of_epochs_per_training=num_ep_per_iter, pruning_kwargs_dict=pruning_kwargs)








