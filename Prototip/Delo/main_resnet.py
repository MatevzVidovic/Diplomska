

import os
import logging
import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)



import argparse
from training_support import *


# save_path = os.path.join(os.path.dirname(__file__), "UNet")
save_path = os.path.join(".", "ResNet")

main_save_path = os.path.join(save_path, "saved_main")




import torch
from torch import nn
from torch.utils.data import DataLoader

from min_resource_percentage import min_resource_percentage


import pandas as pd
import pickle


from ResNet import ResNet, Bottleneck

from dataset import IrisDataset, transform

from ModelWrapper import ModelWrapper




# Logging preparation:

import os




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
    "batch_size" : 2,
    "shuffle" : False, # TODO shuffle??
    "num_workers" : 4,
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
    # it makes no sense. i think this might have been done because the IPAD lambda was done only on the last batch, and so that
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
    "ResBlock" : Bottleneck,
    "layer_list" : [3,4,6,3],
    "num_classes" : 2,
    "num_channels" : 1
  }









# Go see model graph to help you construct these connection functions.
# model_wrapper.model_graph()




# def unet_tree_ix_2_skip_connection_start(tree_ix, conv_tree_ixs):
#     #    tree_ix -> skip_conn_starting_index

#     # It could be done programatically, however:
#     # Assuming the layers that have skip connections have only one source of them,
#     # we could calculate how many inputs come from the previous layer.
#     # That is then the starting ix of skip connections.

#     # To make this function, go look in the drawn matplotlib graph.
#     # On the upstream, just look at the convolution's weight dimensions.
#     # They are: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
#     # (output_dimensions - input_dimensions) is the ix of the first skip connection



#     # Oh, I see. This is easily programmable.
#     # Just use "initial_conv_resource_calc.pkl" and use 
#     # (output_dimensions - input_dimensions) where output_dimensions > input_dimensions.
#     # And that's it haha.

#     conv_ix = None
#     if tree_ix in conv_tree_ixs:
#         conv_ix = conv_tree_ixs.index(tree_ix)

#         if conv_ix == 16:
            
#             return 64
#         elif conv_ix == 14:
            
#             return 128
#         elif conv_ix == 12:
            
#             return 256
#         elif conv_ix == 10:
            
#             return 512


#     else:
        
#         return None
    





"""
THIS HERE IS THE START OF BUILDING A CONNECTION LAMBDA
based on the _get_next_conv_id_list_recursive()
It is very early stage.
"""




# Now we sometimes need to add a convolutional layer to this pruning,
# because in a ResNet bottleneck block, the input into the block is added to the output of the block.
# So they need to be the same dimensions.
# So if the convolution we originally pruned is right before a bottleneck block, the input into the bottleneck layer changes,
# so we need to prune the kernel_ix of the last convolution of the bottleneck block as well, 
# so that its output will have the same dimensions as the input of the bottlencek layer.

# There is a possible sequential layer at the tail end of a bottleneck layer. It is supposed to (up/down) sample the input of the bottleneck layer to the same dimensions as the so-far output of the bottleneck layer.
# Since the bottleneck layer changed the dimensions compared to the input.
# So in this situation this convolutional layer of this sequential block will have to have it's kernel_ix pruned.




# So it really is always the last convolution in the bottleneck block that has to be pruned - just in different ways.

# Just that sometimes that means kernel-pruning the last convolution in the ordinary flow of the network, so that the bottleneck layer output matches the input of the bottleneck layer. (and the input gets added after that)
# And sometimes that means input-pruning the sequential (up/down) sampling layer, so that its input matches the input of the bottleneck layer
# (so that it can transform the input into the dimensions of the output of the regular flow of the bottleneck layer - as is the job of this sequential layer all along).


# This also means that if the last convolution of the regular flow of the bottleneck layer is kernel pruned,
# the the (up-dpwn) sampling sequential layer has to be kernel-pruned - so that its output dimensions will match the above-mentioned layer's output and the addition will be able to happen.
# And the first layer of the next bottleneck block has to be input-pruned - the regular flow of the network needs to match..


# So, basically:
# - batchnorm is always inextricably pruned
# - the next layer in the regular flow is allways input-pruned
# - the up/down sampling layer should never be chosen to be pruned (set the FLOPS_min_res_percents to 1.1 so it never is)
# - if the layer right before the bottleneck layer in the regular flow of the network is kernel-pruned, we need to either
#       - (if that bottleneck has no up/down sampling layer) kernel-prune the last convolution in the bottleneck layer, 
#       - (if the bottleneck layer has one) input-prune the up/down sampling layer
# - if the last convolution in the bottleneck layer is kernel-pruned, we need to (besides the regular flow input-pruing) either:
#       - (if there is an upsampling layer) kernel-prune the up/down sampling layer
#       - (if there is no upsampling layer), nothing special happens. BUT IMPORTANTLY!!!!!!!! 
# !!!!!!!!!!!But this kernel-pruning should only occur as a result of the pruning of the layer right before this bottleneck.
#         Because if we were to start the pruning cycle by directly pruning this layer, 
#            this would mismatch the dimensions of the input of the bottleneck layer (since the layer right before this bottleneck wouldnt be pruned).

# So:
# - we have to know all  right-before-regular-ending-bottleneck layers, right-before-(up/down)sampling-ending-bottleneck layers,
#  all regular-ending-bottleneck layers, and all (up/down)sampling-ending-bottleneck layers.
# 
# - For regular-ending-bottleneck layers, we have to know what layer is the ending one, and disallow it from pruning.
# 
# - For (up/down)sampling-ending-bottleneck layers, we have to know what layer is the regular-flow ending one, and what layer is the (up/down) sampler.
#       - for regular-flow ending ones, the upsampler needs to be in inextricably_pruned().
#           - THIS ALSO MEANS, that for the upsampler, WE NEED TO MIND THE INEX_TO_PRUNE AND FOLLOWING_TO_PRUNE(): the following to prune() has to return [] (since it is not a part of the regular flow) and inextricably_prune() has to return only the batchnorm.
#           - But the disallowing through the FLOPS_min_res_percents is already nicely bypassed, since that is only used in the kernel decision part of pruning.
#      - the (up/down) sampler, is disallowed from deciding to prune it.
# 
# - For right-before-regular-ending-bottleneck layers, the ending-layer needs to be added to inextricably-pruned. (this also nicely bypases the FLOPS_min_res_percents)
#  
# - For right-before-(up/down)sampling-ending-bottleneck layers, the upsampling layer needs to be added to following_to_prune()


# So:
#
# - set diasslowed to FLOPs_min_res_percents (upsamplers and last regular flow layer in bottlenecks without upsamplers)
#  
# - inextricable-connection lambda:
#       - add batchnorm
#       - if right-before-regular-ending-bottleneck:
#           - if no upsampler, prune last regular flow layer
#       - if last regular flow layer is pruned (only allowed to be directly chosen when there is an upsampler):
#             - if upsampler, prune upsampler
# 
# - connection lambda:
#      - if this is upsampler, return empty list
#      - add next regular flow layer
#     - if right-before-regular-ending-bottleneck:
#        - if has upsampler, add upsampler
#   








# [(right_before_conv_ix, bottleneck_last_regular_conv_ix, possibly_bottlenek_upsampler_conv_ix)]
BOTTLENECK_LIST = [
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12, 13),
    (12, 16),
    (16, 19),
    (19, 22),
    (22, 25, 26),
    (25, 29),
    (29, 32),
    (32, 35),
    (35, 38),
    (38, 41),
    (41, 44, 45),
    (44, 48),
    (48, 51)
]

class BottleneckHelper():

    def __init__(self) -> None:
        self.bottleneck_list = BOTTLENECK_LIST

        self.right_before_bottleneck = [bottleneck[0] for bottleneck in self.bottleneck_list]
        self.regular_flow_botleneck_ends = [bottleneck[1] for bottleneck in self.bottleneck_list]
        self.upsamplers = [bottleneck[2] for bottleneck in self.bottleneck_list if len(bottleneck) == 3]
    

    def is_right_before_bottleneck(self, conv_ix):
        return conv_ix in self.right_before_bottleneck
    
    def is_regular_flow_bottleneck_end(self, conv_ix):
        return conv_ix in self.regular_flow_botleneck_ends
    
    def is_upsampler(self, conv_ix):
        return conv_ix in self.upsamplers
    
    
    def get_bottleneck_ix_from_right_before(self, right_before_conv_ix):
        bottleneck_ix = self.right_before_bottleneck.index(right_before_conv_ix)
        return bottleneck_ix
    
    def get_bottleneck_ix_from_regular_flow_end(self, regular_flow_end_conv_ix):
        bottleneck_ix = self.regular_flow_botleneck_ends.index(regular_flow_end_conv_ix)
        return bottleneck_ix

    def bottleneck_has_upsampler(self, bottleneck_ix):
        return len(self.bottleneck_list[bottleneck_ix]) == 3
    
    def get_disallowed_conv_ixs(self):
        disallowed_conv_ixs = []
        for bottleneck in self.bottleneck_list:
            if len(bottleneck) == 3:
                disallowed_conv_ixs.append(bottleneck[2])
            else:
                disallowed_conv_ixs.append(bottleneck[1])

        return disallowed_conv_ixs
    

BH = BottleneckHelper()

REGULAR_FLOW_CONV_IXS = [i for i in range(53) if i not in BH.upsamplers]

DISALLOWED_CONV_IXS = BH.get_disallowed_conv_ixs()






def resnet_input_slice_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, initial_kernel_ix) -> [(goal_tree_ix_1, goal_initial_input_slice_ix_1), (goal_tree_ix_2, goal_initial_input_slice_ix_2),...]
    
    # This functions takes the tree_ix and the ix of where the kernel we are concerned with was in the model initially (before pruning).
    # And it returns a list of tuples giving the following modules tree_ixs and the input_slice_ix
    # (where the effect of the above-mentioned kernel is in the input tensor) in the initial model (before pruning).


    # - connection lambda:
    #      - if this is upsampler, return empty list
    #      - add next regular flow layer
    #     - if right-before-regular-ending-bottleneck:
    #        - if has upsampler, add upsampler

    bh = BH

    conn_destinations = []

    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)

    
    # For batchnorms, this is empty.
    if conv_ix is None:
        return conn_destinations
    

    # If this is upsampler, return empty list
    if bh.is_upsampler(conv_ix):
        return conn_destinations
    

    # outconv has no next convolution, and is not before a bottleneck
    if conv_ix == 52:
        return conn_destinations


    if conv_ix in REGULAR_FLOW_CONV_IXS:
        regular_flow_ix = REGULAR_FLOW_CONV_IXS.index(conv_ix)
        target_conv_ix = REGULAR_FLOW_CONV_IXS[regular_flow_ix+1]
        target_tree_ix = conv_tree_ixs[target_conv_ix]
        conn_destinations.append((target_tree_ix, kernel_ix))

    if bh.is_right_before_bottleneck(conv_ix):
        bottleneck_ix = bh.get_bottleneck_ix_from_right_before(conv_ix)
        if bh.bottleneck_has_upsampler(bottleneck_ix):
            target_conv_ix = bh.bottleneck_list[bottleneck_ix][2]
            target_tree_ix = conv_tree_ixs[target_conv_ix]
            conn_destinations.append((target_tree_ix, kernel_ix))

    return conn_destinations
    
    # We have no skip connections here.



def resnet_kernel_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, real_kernel_ix) -> [(goal_tree_ix_1, goal_real_kernel_ix_1), (goal_tree_ix_2, goal_real_kernel_ix_2),...]
    
    # This functions takes the tree_ix and the ix of where the kernel we are concerned with was in the model RIGHT NOW, NOT INITIALLY.
    # And it returns a list of tuples giving the tree_ixs and "kernel_ixs" in the model RIGHT NOW, NOT INITIALLY.
    # for layers which are inextricably linked with the convolutional layer.

    # Inextricably linked are in direct connection with the conv's kernel_ix, so they don't need the more complex lambda.
    # We could have treated them with the regular lambda, but this way is better,
    # because, in the pruner, we don't need to keep the output_slice_ix.
    # Also it's simpler and conceptually makes more sense.

    # The batchnorm is such a layer - for it, the "kernel_ix" isn't really a kernel ix.
    # It is, however, the position we need to affect due to pruning the kernel_ix in the convolutional layer.
    # There are possibly more such layers and more types of such layers, so we made this function more general.

    bh = BH


    # - inextricable-connection lambda:
    #       - add batchnorm
    #       - if right-before-regular-ending-bottleneck:
    #           - if no upsampler, prune last regular flow layer
    #       - if last regular flow layer is pruned (only allowed to be directly chosen when there is an upsampler):
    #             - if upsampler, prune upsampler


    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)

    LLM_ix = None
    if tree_ix in lowest_level_modules:
        LLM_ix = lowest_level_modules.index(tree_ix)
    
    

    conn_destinations = []


    # For the batchnorms, this should simply be an empty list.
    if conv_ix is None:
        return conn_destinations



    # Adding the batchnorm.
    # out.conv doesn't have a batchnorm after it.
    if conv_ix < 52:
        conn_destinations.append((lowest_level_modules[LLM_ix+1], kernel_ix))

    # Look at comment block above!
    if bh.is_right_before_bottleneck(conv_ix):
        bottleneck_ix = bh.get_bottleneck_ix_from_right_before(conv_ix)
        if not bh.bottleneck_has_upsampler(bottleneck_ix):
            target_conv_ix = bh.bottleneck_list[bottleneck_ix][1]
            target_tree_ix = conv_tree_ixs[target_conv_ix]
            conn_destinations.append((target_tree_ix, kernel_ix))
    
    # Look at comment block above!
    if bh.is_regular_flow_bottleneck_end(conv_ix):
        bottleneck_ix = bh.get_bottleneck_ix_from_regular_flow_end(conv_ix)

        if bh.bottleneck_has_upsampler(bottleneck_ix):
            target_conv_ix = bh.bottleneck_list[bottleneck_ix][2]
            target_tree_ix = conv_tree_ixs[target_conv_ix]
            conn_destinations.append((target_tree_ix, kernel_ix))



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
def averaging_objects(module, input, output, prev_avg_object):
    
    batch_size = output.shape[0]
    batch_mean = output.mean(dim=(0))

    if prev_avg_object[1] is None:
        new_avg_object = (batch_size, batch_mean)
        return new_avg_object

    new_avg_object = (prev_avg_object[0] + batch_size, 
                      (prev_avg_object[0] * prev_avg_object[1] + batch_size * batch_mean) / (prev_avg_object[0] + batch_size))

    return new_avg_object 


averaging_mechanism = {
    "initial_averaging_object" : INITIAL_AVG_OBJECT,
    "averaging_function" : averaging_objects
}



# An additional function could be applied in between the averaging function and the importance function.
# If we were, for example, interested in a specific interaction between the active outputs (not the averaged ones)
# with our averaging object. For example, to calculate the correlation between the output and our average activations.
# Then the averaging function would be applied in the first pass through the network and we would make our averaging objects.
# Then this middle function would be used and we would calculate our batch_importances (correlations) for each batch.
# Then the final importance function we see below us would only be used to combine these batch_importances.
# For example to average them or to sum them.
# But this is not currently implemented.



def IPAD_kernel_importance_lambda_generator(L1_ADC_weight):
    assert L1_ADC_weight > 0 and L1_ADC_weight < 1, "L1_ADC_weight must be between 0 and 1."
    
    
    def IPAD_kernel_importance_lambda(averaging_objects: dict, conv_tree_ixs):
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
        
    
    return IPAD_kernel_importance_lambda
        




if __name__ == "__main__":

    importance_lambda = IPAD_kernel_importance_lambda_generator(0.5)





    input_example = torch.randn(1, 1, 128, 128)

    model_wrapper = ModelWrapper(ResNet, model_parameters, dataloader_dict, learning_parameters, input_example, save_path)






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



    # We have to do this due to this schema above:
    # So:
    #
    # - set diasslowed to FLOPs_min_res_percents (upsamplers and last regular flow layer in bottlenecks without upsamplers)
    #  
    # - inextricable-connection lambda:
    #       - add batchnorm
    #       - if right-before-regular-ending-bottleneck:
    #           - if no upsampler, prune last regular flow layer
    #       - if last regular flow layer is pruned (only allowed to be directly chosen when there is an upsampler):
    #             - if upsampler, prune upsampler
    # 
    # - connection lambda:
    #      - if this is upsampler, return empty list
    #      - add next regular flow layer
    #     - if right-before-regular-ending-bottleneck:
    #        - if has upsampler, add upsampler
    #
    conv_tree_ixs = FLOPS_min_res_percents.get_ordered_list_of_tree_ixs_for_layer_name("Conv2d")
    disallowed_dict = {}
    for tree_ix in DISALLOWED_CONV_IXS:
        disallowed_dict[conv_tree_ixs[tree_ix]] = 1.1
    FLOPS_min_res_percents.set_by_tree_ix_dict(disallowed_dict)
    



    weights_min_res_percents = min_resource_percentage(tree_ix_2_name)
    weights_min_res_percents.set_by_name("Conv2d", 0.2)

    model_wrapper.initialize_pruning(importance_lambda, averaging_mechanism, resnet_input_slice_connection_fn, resnet_kernel_connection_fn, FLOPS_min_res_percents, weights_min_res_percents)



    # model_wrapper.model_graph()
    # input()












    def validation_stop(val_errors):
        # returns True when you should stop

        if len(val_errors) >= 2:
            return True
        
        if len(val_errors) < 3:
            return False
        
        
        returner = val_errors[-1] > val_errors[-2] and val_errors[-1] > val_errors[-3]

        # if previous metric doesn't say we should return, we also go check another metric:
        # if the current validation error is higher than either of the 4. and 5. back
        # we should stop. Because it means we are not improving.
        if not returner and len(val_errors) >= 5:
            returner = val_errors[-1] > val_errors[-4] or val_errors[-1] > val_errors[-5]
            
        return returner




    parser = argparse.ArgumentParser(description="Process an optional positional argument.")

    # Add the optional positional arguments
    parser.add_argument('validation_phase', nargs='?', type=bool, default=False,
                        help='Boolean flag for validation phase')
    parser.add_argument('iter_possible_stop', nargs='?', type=int, default=1e9,
                        help='An optional positional argument with a default value of 1e9')
    
    # Add the optional arguments
    # setting error_ix: ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)
    parser.add_argument('--e_ix', type=int, default=3,
                        help='ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)')
    parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations')
    parser.add_argument('--map', type=int, default=1e9, help='Max auto prunings')
    parser.add_argument('--nept', type=int, default=1,
                        help='Number of epochs per training iteration')
    
    parser.add_argument('--pbop', type=bool, default=False,
                        help='Prune by original percent, otherwise by number of filters')
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

    
    
    train_automatically(model_wrapper, main_save_path, validation_stop, max_training_iters=max_train_iters, max_auto_prunings=max_auto_prunings, train_iter_possible_stop=iter_possible_stop, validation_phase=is_val_ph, error_ix=err_ix,
                         num_of_epochs_per_training=num_ep_per_iter, pruning_kwargs_dict=pruning_kwargs)








