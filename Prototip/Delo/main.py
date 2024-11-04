

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

import matplotlib.pyplot as plt
import pandas as pd
import pickle
import argparse

from unet import UNet
from dataset import IrisDataset, transform

from min_resource_percentage import min_resource_percentage
from ModelWrapper import ModelWrapper









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
    "n_channels" : 1,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }









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
def averaging_objects(module, input, output, prev_avg_object):
    
    batch_size = output.shape[0]
    batch_mean = output.mean(dim=(0))

    if prev_avg_object[1] is None:
        new_avg_object = (batch_size, batch_mean)
        return new_avg_object

    new_avg_object = (prev_avg_object[0] + batch_size, 
                      (prev_avg_object[0] * prev_avg_object[1] + batch_size * batch_mean) / (prev_avg_object[0] + batch_size))

    return new_avg_object 

if __name__ == "__main__":
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
        




if __name__ == "__main__":

    importance_fn = IPAD_kernel_importance_fn_generator(0.5)





    input_example = torch.randn(1, 1, 128, 128)

    model_wrapper = ModelWrapper(UNet, model_parameters, dataloader_dict, learning_parameters, input_example)





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

    model_wrapper.initialize_pruning(importance_fn, averaging_mechanism, unet_input_slice_connection_fn, unet_kernel_connection_fn, FLOPS_min_res_percents, weights_min_res_percents)
























def validation_stop(val_errors):
    # returns True when you should stop

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




def is_previous_model(model_path, model_wrapper, get_last_model_path = False):

    prev_model_details = pd.read_csv(os.path.join(model_wrapper.save_path, "previous_model_details.csv"))
    prev_model_path = prev_model_details["previous_model_path"][0]

    returner = model_path == prev_model_path

    if get_last_model_path:
        returner = (returner, prev_model_path)

    return returner


def delete_old_model(model_path, model_wrapper):
   
    is_prev, prev_model_path = is_previous_model(model_path, model_wrapper, get_last_model_path=True)
    
    if is_prev:
        print("The model you are trying to delete is the last model that was saved. You can't delete it.")
        return False
    
    os.remove(prev_model_path)
    return True





class TrainingLogs:

    def __init__(self, number_of_epochs_per_training, last_train_iter=None, deleted_models_errors=[]) -> None:
        
        self.number_of_epochs_per_training = number_of_epochs_per_training
        self.last_train_iter = last_train_iter
        self.deleted_models_errors = deleted_models_errors

        # of the form (val_error, test_error, train_iter, model_path)
        self.errors = []

    def add_error(self, val_error, test_error, train_iter, model_path):
        self.errors.append((val_error, test_error, train_iter, model_path))
        self.last_train_iter = train_iter

    
    def __str__(self):
        returner = ""
        returner += f"Number of epochs per training: {self.number_of_epochs_per_training}\n"
        returner += f"Last train iteration: {self.last_train_iter}\n"
        returner += f"Errors: {self.errors}\n"
        returner += f"Deleted models errors: {self.deleted_models_errors}\n"
        return returner
    
# with the exception of keeping (k+1) models when one of the worse models is the last model we have 
# (we have to keep it to continue training)
def delete_all_but_best_k_models(k: int, training_logs: TrainingLogs, model_wrapper: ModelWrapper):

    # sort by validation error
    sorted_errors = sorted(training_logs.errors, key = lambda x: x[0], reverse=True)

    to_delete = []

    while len(sorted_errors) > 0 and (len(training_logs.errors) - len(to_delete)) > k:

        error = sorted_errors.pop(0)
        model_path = error[3]

        if is_previous_model(model_path, model_wrapper):
            continue

        to_delete.append(error)
    


    for error in to_delete:
        model_path = error[3]
        os.remove(model_path)
    

    to_keep = [error for error in training_logs.errors if error not in to_delete]
    new_training_logs = TrainingLogs(training_logs.number_of_epochs_per_training, training_logs.last_train_iter, training_logs.deleted_models_errors)
    for error in to_keep:
        new_training_logs.add_error(*error)

    new_training_logs.deleted_models_errors.extend(to_delete)
    
    return new_training_logs

 


# What we pruned is already saved in pruner_istance.pruning_logs
# We just need to keep track of the corresponding train_iter to be able to know when which pruning happened.
# That's what this function is for.
def log_pruning_train_iter(train_iter):
    
    save_path = os.path.join(os.path.dirname(__file__), "saved_main")
    os.makedirs(save_path, exist_ok=True)

    pruning_train_iter_path = os.path.join(save_path, "pruning_train_iter.pkl")


    if os.path.exists(pruning_train_iter_path):
        pruning_train_iters = pickle.load(open(pruning_train_iter_path, "rb"))
    else:
        pruning_train_iters = []
    
    pruning_train_iters.append((train_iter, False))

    with open(pruning_train_iter_path, "wb") as f:
        pickle.dump(pruning_train_iters, f)


# When we prune, we save the training iter of that pruning.
# But what if we stop the training before that pruned model is actually saved?
# This function sets the flag for the train_iter actually being confirmed.
def confirm_last_pruning_train_iter():
    
    save_path = os.path.join(os.path.dirname(__file__), "saved_main")
    os.makedirs(save_path, exist_ok=True)

    pruning_train_iter_path = os.path.join(save_path, "pruning_train_iter.pkl")


    if os.path.exists(pruning_train_iter_path):
        pruning_train_iters = pickle.load(open(pruning_train_iter_path, "rb"))
    else:
        return
    
    if len(pruning_train_iters) == 0:
        return
    
    pruning_train_iters[-1] = (pruning_train_iters[-1][0], True)

    with open(pruning_train_iter_path, "wb") as f:
        pickle.dump(pruning_train_iters, f)


# If we stop the training before the pruned model is saved, 
# the last train iter would have turned to true in the next saving iteration,
# despite the fact it was never saved and has no effect.
# That's why we have to clean it up before training.
def clean_up_pruning_train_iters():

    save_path = os.path.join(os.path.dirname(__file__), "saved_main")
    os.makedirs(save_path, exist_ok=True)

    pruning_train_iter_path = os.path.join(save_path, "pruning_train_iter.pkl")


    if os.path.exists(pruning_train_iter_path):
        pruning_train_iters = pickle.load(open(pruning_train_iter_path, "rb"))
    else:
        return

    if len(pruning_train_iters) == 0:
        return
    
    if pruning_train_iters[-1][1] == False:
        pruning_train_iters = pruning_train_iters[:-1]

    with open(pruning_train_iter_path, "wb") as f:
        pickle.dump(pruning_train_iters, f)





        








def train_automatically(train_iter_possible_stop=5, validation_phase=False, error_ix=1, num_of_epochs_per_training=1, num_of_filters_to_prune=1):





    save_path = os.path.join(os.path.dirname(__file__), "saved_main")
    os.makedirs(save_path, exist_ok=True)

    clean_up_pruning_train_iters()

    previous_training_phase_details_path = os.path.join(save_path, "previous_training_phase_details.csv")

    if os.path.exists(previous_training_phase_details_path):
        prev_training_phase_details = pd.read_csv(previous_training_phase_details_path)
        prev_training_phase_serial_num = prev_training_phase_details["previous_serial_num"][0]
        prev_training_logs_path = prev_training_phase_details["previous_training_logs_path"][0]
    else:
        prev_training_phase_serial_num = None
        prev_training_logs_path = None
    

    if prev_training_phase_serial_num is None:
        curr_training_phase_serial_num = 0
    else:
        curr_training_phase_serial_num = prev_training_phase_serial_num + 1


    if prev_training_logs_path is None:
        training_logs = TrainingLogs(num_of_epochs_per_training)
    else:
        training_logs = pickle.load(open(prev_training_logs_path, "rb"))
        











    train_iter = training_logs.last_train_iter
    
    if train_iter is None:
        train_iter = 0
    else:
        train_iter += 1 # because we are starting a new training phase

    initial_train_iter = train_iter


    validation_errors = []

    while True:


        
        model_wrapper.train(num_of_epochs_per_training)

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        val_error = model_wrapper.validation()[error_ix]

        if error_ix in [1, 2, 3]:
            val_error = 1 - val_error

        validation_errors.append(val_error)

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")



        test_error = model_wrapper.test()[error_ix]

        if error_ix in [1, 2, 3]:
            test_error = 1 - test_error

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        # print(f"Hooks: {model_wrapper.tree_ix_2_hook_handle}")

        # print(model_wrapper)






        new_model_path, _ = model_wrapper.save(str(train_iter))
        confirm_last_pruning_train_iter()

        training_logs.add_error(val_error, test_error, train_iter, new_model_path)

        # This has to be done before saving the training logs.
        # Otherwise we wil load a training_logs that will still have something in its errors that it has actually deleted.
        # e.g. Errors: [(0.6350785493850708, 0.6345304846763611, 0, 6), (0.6335894465446472, 0.6331750154495239, 1, 7), (0.6319190859794617, 0.6316145658493042, 2, 8), (0.630038321018219, 0.6299036741256714, 3, 9)]
        # But model 6 has actually been already deleted: [(0.6350785493850708, 0.6345304846763611, 0, 6)]
        
        # The conceptual lifetime of training logs is created/loaded -> added to -> model_deletion -> saved
        # And then the process can repeat. Deletion can't be after saved, it makes no sense. Just think of doing just one iteration of it.
        training_logs = delete_all_but_best_k_models(3, training_logs, model_wrapper)

        

        new_training_logs_path = os.path.join(save_path, f"training_logs_{curr_training_phase_serial_num}_{train_iter}.pkl")
        with open(new_training_logs_path, "wb") as f:
            pickle.dump(training_logs, f)
        
        new_df = pd.DataFrame({"previous_serial_num": [curr_training_phase_serial_num],
                                "previous_training_logs_path": new_training_logs_path})
        new_df.to_csv(os.path.join(save_path, "previous_training_phase_details.csv"))











        train_iter += 1

        # Implement the stopping by hand. We need this for debugging.
        
        if (train_iter - initial_train_iter) % train_iter_possible_stop == 0:
            
            inp = input(f"""{train_iter_possible_stop} trainings have been done error stopping.
                        Best k models are kept. (possibly (k+1) models are kept if one of the worse models is the last model we have).
                        
                        Enter g to show the graph of the model and re-ask for input.
                        Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            if inp == "g":
                model_wrapper.model_graph()
                inp = input("""Enter r to trigger show_results() and re-ask for input.
                        Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            if inp == "r":
                show_results()
                inp = input("""Enter a number to reset in how many trainings we ask you this again, and re-ask for input.
                        Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            
            
            try:
                train_iter_possible_stop = int(inp)
                print(f"New trainings before stopping: {train_iter_possible_stop}")
                inp = input("""Enter p to prune anyways (in production code, that is commented out, so the program will simply stop).
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")
            except ValueError:
                pass


            if inp == "p":
                
                model_wrapper.prune(num_of_filters_to_prune)

                # This will ensure I have the best k models from every pruning phase.
                model_wrapper.create_safety_copy_of_existing_models(str(train_iter))
                log_pruning_train_iter(train_iter)
                validation_errors = []

                inp = input("""Enter g to show the graph of the model and re-ask for input.
                        Press Enter to continue training.
                        Enter any other key to stop.\n""")

            if inp == "g":
                model_wrapper.model_graph()
                inp = input("""Press Enter to continue training.
                        Enter any other key to stop.\n""")

            if inp != "":
                break


        
        if validation_phase and validation_stop(validation_errors):
            model_wrapper.prune(num_of_filters_to_prune)

            # This will ensure I have the best k models from every pruning phase.
            model_wrapper.create_safety_copy_of_existing_models(str(train_iter))
            log_pruning_train_iter(train_iter)
            validation_errors = []



def show_results():

    try:
        inp = input("Enter what comes between pruner_ and .pkl: ")

        # pruner_instance = pickle.load(open("./saved/pruner_" + inp + ".pkl", "rb"))
        # pruning_logs_dict = pruner_instance.pruning_logs

        pruning_train_iter = pickle.load(open("./saved_main/pruning_train_iter.pkl", "rb"))
        pruning_moments = [i[0] for i in pruning_train_iter]


        training_logs = pickle.load(open("./saved_main/training_logs_" + inp + ".pkl", "rb"))

        model_errors = training_logs.errors + training_logs.deleted_models_errors
        model_errors.sort(key = lambda x: x[2])

        val_errors = [error[0] for error in model_errors]
        test_errors = [error[1] for error in model_errors]


        # make a plot of the val errors and test errors over time
        # make vertical lines for when pruning happened (in pruning moments)

        plt.plot(val_errors, label="Validation errors")
        plt.plot(test_errors, label="Test errors")
        plt.ylabel("1 - IoU")
        plt.xlabel("training iterations")

        for moment in pruning_moments:
            plt.axvline(x=moment, color="blue", linestyle="--")
            




        from matplotlib.patches import Patch

        # Create the initial legend
        plt.legend()

        # Get current handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dummy handle for the arbitrary string
        extra = Patch(facecolor='none', edgecolor='none', label='Arbitrary String')

        # Append the new handle and label
        # handles.append(extra)
        # labels.append("Green line: 10 filters pruned")
        handles.append(extra)
        labels.append("Blue line: 1 filter pruned")

        # Update the legend with the new entries
        plt.legend(handles=handles, labels=labels)

        plt.show()

    except Exception as e:
        # Print the exception message
        print("An exception occurred in show_results():", e)
        # Print the type of the exception
        print("Exception type:", type(e).__name__)
        print("Continuin operation.")



if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Process an optional positional argument.")

    # Add the optional positional arguments
    parser.add_argument('iter_possible_stop', nargs='?', type=int, default=1e9,
                        help='An optional positional argument with a default value of 1e9')
    parser.add_argument('validation_phase', nargs='?', type=bool, default=False,
                        help='Boolean flag for validation phase')
    
    # Add the optional arguments
    # setting error_ix: ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)
    parser.add_argument('--e_ix', type=int, default=3,
                        help='ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)')
    parser.add_argument('--nept', type=int, default=1,
                        help='Number of epochs per training iteration')
    parser.add_argument('--nftp', type=int, default=1,
                        help='Number of filters to prune in one pruning')

    args = parser.parse_args()

    iter_possible_stop = args.iter_possible_stop
    is_val_ph = args.validation_phase
    err_ix = args.e_ix
    num_ep_per_iter = args.nept
    num_to_prune = args.nftp


    
    
    train_automatically(train_iter_possible_stop=iter_possible_stop, validation_phase=is_val_ph, error_ix=err_ix,
                         num_of_epochs_per_training=num_ep_per_iter, num_of_filters_to_prune=num_to_prune)








