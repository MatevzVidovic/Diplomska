
import operator as op

import torch
from torch import nn
from torch.utils.data import DataLoader

import pickle



from model_vizualization import model_graph



from unet import UNet

from dataset import IrisDataset, transform

from ModelWrapper import ModelWrapper




# Logging preparation:

import logging
import sys
import os

# Assuming the submodule is located at 'python_logger'
submodule_path = os.path.join(os.path.dirname(__file__), 'python_logger')
sys.path.insert(0, submodule_path)

import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string instead of __name__. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)
# def file_handler_setup(logger, path_to_python_logger_folder, add_stdout_stream: bool = False)





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
    "valid" : valid_dataloader,
    "test" : test_dataloader,
}



model_parameters = {
    # layer sizes
    "n_channels" : 1,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }













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
THIS HERE IS THE START OF BUILDING A CONNECTION LAMBDA
based on the _get_next_conv_id_list_recursive()
It is very early stage.
"""



def unet_connection_lambda(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, initial_kernel_ix) -> [(goal_tree_ix_1, goal_initial_input_slice_ix_1), (goal_tree_ix_2, goal_initial_input_slice_ix_2),...]
    
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






def unet_inextricable_connection_lambda(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
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
    


    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)

    LLM_ix = None
    if tree_ix in lowest_level_modules:
        LLM_ix = lowest_level_modules.index(tree_ix)
    
    

    conn_destinations = []
    
    # out.conv doesn't have a batchnorm after it.
    if conv_ix < 18:
        conn_destinations.append((lowest_level_modules[LLM_ix+1], kernel_ix))
    

    
    return conn_destinations










def IPAD_kernel_importance_lambda_generator(L1_ADC_weight):
    assert L1_ADC_weight > 0 and L1_ADC_weight < 1, "L1_ADC_weight must be between 0 and 1."
    
    
    def IPAD_kernel_importance_lambda(activations, conv_tree_ixs):
        # Returns dict tree_ix_2_list_of_kernel_importances
        # The ix-th importance is for the kernel currently on the ix-th place.
        # To convert this ix to the initial unpruned models kernel ix, use the pruner's
        # state of active kernels.

        tree_ix_2_kernel_importances = {}
        for tree_ix in conv_tree_ixs:

            curr_batch_outputs = activations[tree_ix]
            # print("len(curr_batch_outputs):")
            # print(len(curr_batch_outputs))
            # print("curr_batch_outputs[0].shape:")
            # print(curr_batch_outputs[0].shape)
            curr_batch_outputs = torch.cat(curr_batch_outputs, dim=(0))
            # print(curr_batch_outputs.shape)
            # print(type(curr_batch_outputs))
            # print(curr_batch_outputs)
            kernels_average_activation = curr_batch_outputs.mean(dim=(0))
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
        



importance_lambda = IPAD_kernel_importance_lambda_generator(0.5)


input_example = torch.randn(1, 1, 128, 128)

model_wrapper = ModelWrapper(UNet, model_parameters, dataloader_dict, learning_parameters, input_example, importance_lambda, unet_connection_lambda, unet_inextricable_connection_lambda)






inp = ""
while inp == "" or inp == "g":
    

    inp = input("""Press enter to continue, any text to stop, g to continue and show graph, s to save and stop.\n Enter a number to train and prune automatically for that number of times before asking for input again.\n""")

    if inp == "g":
        model_wrapper.model_graph()
        input("Press enter to continue.")

    if inp == "s":
        model_wrapper.save()
        break


    try:
        repetitions = int(inp)
        inp = ""
    except ValueError:
        repetitions = 1

    if inp not in ["", "g", "s"]:
        break
    
    for _ in range(repetitions):
        model_wrapper.train(1)
        model_wrapper.prune()




    










