
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from unet import UNet

from dataset import IrisDataset, transform

from ModelWrapper import ModelWrapper

from ConvResourceCalc import ConvResourceCalc

import pickle

from pruner import pruner

from min_resource_percentage import min_resource_percentage



test_purposes = True

batch_size = 16
learning_rate = 1e-3








dataloading_args = {


    "testrun" : test_purposes,
   

    # Image resize setting - don't know what it should be.
    "width" : 128,
    "height" : 128,
    
    # iris dataset params
    "path_to_sclera_data" : "./sclera_data",
    "transform" : transform,
    "n_classes" : 2,

    # DataLoader params
    "batch_size" : batch_size,
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

    trainloader = DataLoader(train_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"])
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # I'm not sure why we're dropping last, but okay.

    print('train dataset len: ' + str(train_dataset.__len__()))
    print('val dataset len: ' + str(valid_dataset.__len__()))
    print('test dataset len: ' + str(test_dataset.__len__()))

    return trainloader, validloader, testloader





model_parameters = {
    # layer sizes
    "n_channels" : 1,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }

if __name__ == "__main__":


    model = UNet(**model_parameters)



    train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(**dataloading_args)

    dataloader_dict = {
        "train" : train_dataloader,
        "valid" : valid_dataloader,
        "test" : test_dataloader,
    }


    for X, y in test_dataloader:
        # print(X)
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    
    # https://pytorch.org/docs/stable/optim.html
    # SGD - stochastic gradient descent
    # imajo tudi Adam, pa sparse adam, pa take.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss() # nn.MSELoss()

    learning_dict = {
        "optimizer" : optimizer,
        "loss_fn" : loss_fn,
    }


    wrap_model = ModelWrapper(model, dataloader_dict, learning_dict, model_name="UNet")

    wrap_model.loop_train_test(epochs=1)

    wrap_model.save()

    # wrap_model.test_showcase()


    resource_calc = ConvResourceCalc(wrap_model)
    resource_calc.calculate_resources(torch.randn(1, 1, 128, 128))
    FLOPs = resource_calc.cur_flops
    resource_dict = resource_calc.module_tree_ixs_2_flops_dict

    print(f"FLOPs: {FLOPs}")
    print(f"Resource dict: {resource_dict}")


    # pickle resource_dict
    with open("initial_resource_dict.pkl", "wb") as f:
        pickle.dump(resource_dict, f)
    














    """
    This is proof of concept of how to get activations through hooks.
    It kind of already works, which is awesome.
    """

    print(wrap_model.model)

    # print(resource_calc.module_tree_ixs_2_modules_themselves)

    print(resource_calc.module_tree_ixs_2_name)


    conv_modules_tree_ixs = []
    for key, value in resource_calc.module_tree_ixs_2_name.items():
        if value == "Conv2d":
            conv_modules_tree_ixs.append(key)
    
    print(conv_modules_tree_ixs)


    activations = {}
    def get_activation(tree_ix):
        def hook(model, input, output):
            activations[tree_ix] = output.detach()
        return hook

    tree_ix_2_hook_handle = {}
    for tree_ix in conv_modules_tree_ixs:
        module = resource_calc.module_tree_ixs_2_modules_themselves[tree_ix]
        tree_ix_2_hook_handle[tree_ix] = module.register_forward_hook(get_activation(tree_ix))
    
    input_tensor = torch.randn(1, 1, 128, 128)
    wrap_model.model.eval()
    with torch.no_grad():
        model(input_tensor)
    
    print(activations)


    # This shows how to remove hooks when they are no longer needed.
    # Tiis can save memory.
    for tree_ix, hook_handle in tree_ix_2_hook_handle.items():
        hook_handle.remove()

    




    min_res_percents = min_resource_percentage(resource_calc.module_tree_ixs_2_name)
    min_res_percents.set_by_name("Conv2d", 0.5)

    tree_ix_2_percentage_dict = {
        (0,) : 0.2,
        ((0,), 0) : 0.2,
    }
    min_res_percents.set_by_tree_ix_dict(tree_ix_2_percentage_dict)

    print(min_res_percents.min_resource_percentage_dict)





    """
    Kind of an attempt at getting module names - could help with specifying connections.
    """

    def generator_to_list(generator):
        return [name for name, module in generator]

    def first_element(generator):
        name, module = next(generator)
        # return name
        return module
    
    def second_element(generator):
        name, module = next(generator)
        name, module = next(generator)
        return name
        # return module

    print(generator_to_list(resource_calc.module_tree_ixs_2_modules_themselves[(0,)].named_modules(remove_duplicate=False)))
    # print(first_element(resource_calc.module_tree_ixs_2_modules_themselves[((None, 0), 0)].named_modules(remove_duplicate=False)))
    # print(second_element(resource_calc.module_tree_ixs_2_modules_themselves[((None, 0), 0)].named_modules(remove_duplicate=False)))

    print(generator_to_list(resource_calc.module_tree_ixs_2_modules_themselves[((0,), 0)].named_modules(remove_duplicate=False)))



    """
    NEW IDEA:
    I will make an ordered list where there are only tree_ixs of the lowest level modules.
    These are the onew we are actually concerned with when writing the conections lambda.
    The user can then check the list and check the corresponding names.
    This should make working with the tree_ixs easier.

    If it doesn't work, the user can always still type the tree_ixs manually.
    It isn't hard - they would mostly be copying and pasting them and only changing a numebr or two.
    """

    lowest_level_modules_tree_ixs = []
    for tree_ix, children_list in resource_calc.module_tree_ixs_2_children_tree_ix_lists.items():
        if len(children_list) == 0:
            lowest_level_modules_tree_ixs.append(tree_ix)
    
    print(lowest_level_modules_tree_ixs)

    def denest_tuple(tup):
        returning_list = []
        for item in tup:
            if isinstance(item, tuple):
                returning_list.extend(denest_tuple(item))
            elif isinstance(item, int):
                returning_list.append(item)
        return returning_list
    
    def renest_tuple(lst):
        curr_tuple = (lst[0],)
        for item in lst[1:]:
            curr_tuple = (curr_tuple, item)
        return curr_tuple
        
    lowest_level_modules_denested_tree_ixs = [denest_tuple(tree_ix) for tree_ix in lowest_level_modules_tree_ixs]
    print(lowest_level_modules_denested_tree_ixs)
    
    lowest_level_modules_denested_tree_ixs = sorted(lowest_level_modules_denested_tree_ixs)
    print(lowest_level_modules_denested_tree_ixs)



    sorted_lowest_level_modules_tree_ixs = [renest_tuple(lst) for lst in lowest_level_modules_denested_tree_ixs]
    print(sorted_lowest_level_modules_tree_ixs)

    lowest_level_modules_names = [resource_calc.module_tree_ixs_2_name[tree_ix] for tree_ix in sorted_lowest_level_modules_tree_ixs]
    print(lowest_level_modules_names)

    lowest_level_modules = sorted_lowest_level_modules_tree_ixs


    """
    I HAVE TO THINK MORE ABOUT HOW TO MAKE THIS EASIER.
    LOWEST LEVEL IDEA IS NOT THE BEST.
    MAYBE GO BY ORTING ALL MODULES OF A CERTAIN NAME AND CREATE THAT LIST.
    THIS WAY YOU CAN ACCESS THE TREE_IXS BY SIMPLY CONNECTING THE TYPE INDEX.

    WITH PARALLEL CONNECTIONS ONE WILL SIMPLY HAVE TO LOOK UP WHICH ONE CAME FIRST IN THIS LIST.
    IT REQUIRES SOME MORE WORK FOR THE USER BUT THERE IS NO WAY TO DO IT PROGRAMATICALLY.
    """



    def unet_resource_lambda(tree_ix, filter_ix):

        low_level_module_ix = lowest_level_modules.index(tree_ix)
        idx = low_level_module_ix

        next_idxs_list = [idx+1]

        # if idx == 6:
        #     next_idxs_list.append



        # output is: [(goal_tree_ix_1, goal_filter_ix_1), (goal_tree_ix_2, goal_filter_ix_2),...] 
                # Output of conv2 in each down block also goes to conv1 in corresponding up block
        if layer_index == 1:
            next_conv_idx = [2, 16]
        elif layer_index == 3:
            next_conv_idx = [4, 14]
        elif layer_index == 5:
            next_conv_idx = [6, 12]
        elif layer_index == 7:
            next_conv_idx = [8, 10]
        # outc has no next convolution
        elif layer_index >= 18:
            next_conv_idx = []
        # Every other convolution output just goes to the next one
        else:
            next_conv_idx = [layer_index + 1]


