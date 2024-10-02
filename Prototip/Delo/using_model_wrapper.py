
import os
import operator as op

import torch
from torch import nn
from torch.utils.data import DataLoader

import pickle





from unet import UNet

from dataset import IrisDataset, transform

from ModelWrapper import ModelWrapper

from ConvResourceCalc import ConvResourceCalc

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
    FLOPs = resource_calc.all_flops_num
    resource_dict = resource_calc.module_tree_ixs_2_flops_dict

    print(f"FLOPs: {FLOPs}")
    print(f"Resource dict: {resource_dict}")
    print(f"Weight dimensions: {resource_calc.module_tree_ix_2_weights_dimensions}")
    print(f"Weight numbers: {resource_calc.module_tree_ix_2_weights_num}")



    # pickle resource_dict
    with open("initial_conv_resource_calc.pkl", "wb") as f:
        to_pickle = resource_calc.get_copy_for_pickle()
        pickle.dump(to_pickle, f)
    










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
            if tree_ix not in activations:
                activations[tree_ix] = []
            activations[tree_ix].append(output.detach())
        return hook

    tree_ix_2_hook_handle = {}
    for tree_ix in conv_modules_tree_ixs:
        module = resource_calc.module_tree_ixs_2_modules_themselves[tree_ix]
        tree_ix_2_hook_handle[tree_ix] = module.register_forward_hook(get_activation(tree_ix))
    
    print(activations)

    wrap_model.model.eval()
    with torch.no_grad():
        for i in range(10):
            input_tensor = torch.randn(1, 1, 128, 128)
            model(input_tensor)
    
    # print(activations)

    print(f"Number of activations for [((((0,), 0), 0), 0)]: {len([((((0,), 0), 0), 0)])}")
    print(activations[((((0,), 0), 0), 0)][0].shape)


    # input_tensor = torch.randn(1, 1, 128, 128)
    # wrap_model.model.eval()
    # with torch.no_grad():
    #     model(input_tensor)
    

    # print(activations[((((0,), 0), 0), 0)].shape)




    # This shows how to remove hooks when they are no longer needed.
    # This can save memory.
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
    It works - you can access the modules by their names.
    The problem is .named_modules() has its own order, and we don't know if it is the same as the order of the tree_ixs.
    We will probably just use the tree_ixs directly.
    But it's interestting that the previous IPAD code does it by name (train_with_pruning_combined.py/_layer_index_to_conv)
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

    module_full_names = generator_to_list(resource_calc.module_tree_ixs_2_modules_themselves[(0,)].named_modules(remove_duplicate=False))
    print(module_full_names)
    # print(first_element(resource_calc.module_tree_ixs_2_modules_themselves[((None, 0), 0)].named_modules(remove_duplicate=False)))
    # print(second_element(resource_calc.module_tree_ixs_2_modules_themselves[((None, 0), 0)].named_modules(remove_duplicate=False)))

    print(generator_to_list(resource_calc.module_tree_ixs_2_modules_themselves[((0,), 0)].named_modules(remove_duplicate=False)))

    accessing_first_module = getattr(wrap_model.model, module_full_names[1]) # this works
    print(accessing_first_module)
    # accessing_first_module = getattr(wrap_model.model, module_full_names[2]) # this doesn't work
    # print(accessing_first_module)
    accessing_first_module = op.attrgetter(module_full_names[2])(wrap_model.model) # but this works
    print(accessing_first_module)
    # So it makes sense to just always use op.attrgetter() instead of getattr().

    # input("Press enter to continue.")











    """
    NEW IDEA:
    I will make an ordered list where there are only tree_ixs of the lowest level modules - these are the onew who actually get pruned.
    These are the ones we are actually concerned with when writing the conections lambda.
    The user can then check the list and check the corresponding names.
    This should make working with the tree_ixs easier.

    If it doesn't work, the user can always still type the tree_ixs manually.
    It isn't hard - they would mostly be copying and pasting them and only changing a numebr or two.

    However, it is kind of hard to know what is the ordinal number of the lowest level module.
    This is why lower we make a better method.
    """



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
    

    def sort_tree_ixs(list_of_tree_ixs):
        
        denested_list = [denest_tuple(tree_ix) for tree_ix in list_of_tree_ixs]
        
        sorted_denested_list = sorted(denested_list)

        sorted_list = [renest_tuple(lst) for lst in sorted_denested_list]

        # print(denested_list)
        # print(sorted_denested_list)
        return sorted_list


    lowest_level_modules_tree_ixs = []
    for tree_ix, children_list in resource_calc.module_tree_ixs_2_children_tree_ix_lists.items():
        if len(children_list) == 0:
            lowest_level_modules_tree_ixs.append(tree_ix)
    
    print(lowest_level_modules_tree_ixs)


    sorted_lowest_level_modules_tree_ixs = sort_tree_ixs(lowest_level_modules_tree_ixs)
    print(sorted_lowest_level_modules_tree_ixs)

    lowest_level_modules_names = [resource_calc.module_tree_ixs_2_name[tree_ix] for tree_ix in sorted_lowest_level_modules_tree_ixs]
    print(lowest_level_modules_names)

    lowest_level_modules = sorted_lowest_level_modules_tree_ixs


    """
    I HAVE TO THINK MORE ABOUT HOW TO MAKE THIS EASIER.
    LOWEST LEVEL IDEA IS NOT THE BEST.
    MAYBE GO BY SORTING ALL MODULES OF A CERTAIN NAME AND CREATE THAT LIST.
    THIS WAY YOU CAN ACCESS THE TREE_IXS BY SIMPLY CONNECTING THE TYPE INDEX.

    WITH PARALLEL CONNECTIONS ONE WILL SIMPLY HAVE TO LOOK UP WHICH ONE CAME FIRST IN THIS LIST.
    IT REQUIRES SOME MORE WORK FOR THE USER BUT THERE IS NO WAY TO DO IT PROGRAMATICALLY.
    """


    """
    I guess:
    - find all tree_ixs of a certain name
    - sort them with denesting and renesting as above
    - voila. This is the list you want - you can access the tree_ix as the ordinal number of this type of layer.
    Just make a method where you pass a name and you get the sorted list.
    This should make things a lot easier.
    """

    tree_ix_2_layer_name = resource_calc.module_tree_ixs_2_name

    def get_ordered_list_of_tree_ixs_for_layer_name(module_tree_ixs_2_name, layer_name):
        
        applicable_tree_ixs = []
        for tree_ix, module_name in module_tree_ixs_2_name.items():
            if module_name == layer_name:
                applicable_tree_ixs.append(tree_ix)
        
        assert len(applicable_tree_ixs) > 0, f"No module with name {layer_name} found."

        sorted_applicable_tree_ixs = sort_tree_ixs(applicable_tree_ixs)

        return sorted_applicable_tree_ixs
    
    conv_tree_ixs = get_ordered_list_of_tree_ixs_for_layer_name(tree_ix_2_layer_name, "Conv2d")
    print(5*"\n" + "Example of all Conv2d layers list:")
    print(conv_modules_tree_ixs)
    names_of_conv_tree_ixs = [tree_ix_2_layer_name[tree_ix] for tree_ix in conv_tree_ixs]
    print(names_of_conv_tree_ixs)

    
    


    """
    Great idea to visualize these tree_ixs and their names:
    

    how to do this in python?
    I have a dictionary: tree_ix_2_layer_name
    Tree ixs are tuples like so: ((0,), 1), 0), 2) - which is the path down a virtual tree (zeroth node (root), then first child, then zeroth child, then second child. I have such tree_ixs for both leaves and inside nodes.

    How to accomplish this in python:

    - naredit tree ix visualiser. Naj se izriše pravokotnik čez cel ekran. Zgornja 2 centimetra piše Unet in zraven njegov tree_ix. Kar je ostalo spodaj se razseka po širini na toliko delov, iz kolikor je sestavljen Unet. Potem  Za vsak ta del storimo isto naprej. Na najnižji vrstici lahko sledimo bottom level layerjem ki so actual building vlocks po katerih input potuje dalje.
    Tako se bo actually very simply dalo pogledat, kater tree_ix ima nek blok, in kater zapovrsti tip bloka je to.
    """


    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # def denest_tuple_to_tuple(tup):
    #     new_tup = tuple(denest_tuple(tup))
    #     return new_tup

    def draw_tree(ix, layer_name, ax, x, y, width, height, max_depth):
        # Draw the rectangle and label it

        # which of its' name is it?
        this_name_tree_ixs = get_ordered_list_of_tree_ixs_for_layer_name(tree_ix_2_layer_name, layer_name)
        # print(this_name_tree_ixs)
        # print(ix)
        ordered_ix = this_name_tree_ixs.index(ix)


        # ordered_ix = -1        
        # for curr_ix, tree_ix in enumerate(this_name_tree_ixs):
        #     if curr_ix == ix:
        #         ordered_ix = curr_ix
        #         break
        # if ordered_ix == -1:
        #     raise ValueError(f"tree_ix not found: {ix}")

        display_string = f'{ordered_ix}. {layer_name}\n{ix}'


        # The reason for this is, that when writing the connection lambda,
        # which is used for pruning, we generally only care about the connections
        # to lowest level modules.
        # These are modules that appear the lowest in the tree, and are the ones that actually 
        # do the work. Data passes through them. They arent just composites of less complex modules.
        # They are the actual building blocks.
        if ix in lowest_level_modules:
            lowest_level_modules_index = lowest_level_modules.index(ix)
            display_string += f"\n{lowest_level_modules_index}. LLM"
        

        if ix in resource_calc.module_tree_ix_2_weights_dimensions:
            display_string += f"\n{resource_calc.module_tree_ix_2_weights_dimensions[ix]}"

        

        

        ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor='black', facecolor='none'))
        ax.text(x + width/2, y + height/2, display_string, ha='center', va='center')

        # Find children of the current index
        children = [key for key in tree_ix_2_layer_name if key[0] == ix]
        if children:
            child_width = width / len(children)
            for i, child in enumerate(sort_tree_ixs(children)):
                child_name = tree_ix_2_layer_name[child]
                draw_tree(child, child_name, ax, x + i * child_width, y - height, child_width, height, max_depth - 1)

    def visualize_tree(tree, ax, width=1, height=0.1):
        max_depth = max(len(denest_tuple(k)) for k in tree.keys())
        total_height = max_depth * height
        root_ix = (0,)
        root_name = tree[root_ix]
        draw_tree(root_ix, root_name, ax, 0, total_height, width, height, max_depth)

    fig, ax = plt.subplots()
    visualize_tree(tree_ix_2_layer_name, ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()













    def unet_tree_ix_2_skip_connection_start(tree_ix):

        # It could be done programatically, however:
        # Assuming the layers that have skip connections have only one source of them,
        # we could calculate how many inputs come from the previous layer.
        # That is then the starting ix of skip connections.

        # To make this function, go look in the drawn matplotlib graph.
        # On the upstream, just look at thhe convolution's weight dimensions.
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


    def unet_resource_lambda(tree_ix, filter_ix):
        # f(tree_ix, filter_ix) -> [(goal_tree_ix_1, goal_filter_ix_1), (goal_tree_ix_2, goal_filter_ix_2),...]

        conn_destinations = []

        # we kind of only care about convolutional modules.
        # We just need to prune there (and possibly something with the batch norm layer)
        # So it would make sense to transform the tree_ix to the ordinal number of 
        # the convolutional module, and work with that ix instead.

        conv_ix = None
        if tree_ix in conv_tree_ixs:
            conv_ix = conv_tree_ixs.index(tree_ix)
            conn_destinations.append((conv_ix+1, filter_ix))

        # We made it so that for conv layers who receive as input the previous layer and a skip connection
        # the first filters are of the previous layer. This makes the line above as elegant as it is.
        # We will, however, have to deal with more trouble with skip connections. 

        
        # For the more general option (e.g. to include a possible batchnorm pruning) 
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
        
        goal_tree_ix = None
        if conv_ix == 1:
            goal_tree_ix = 16
        elif conv_ix == 3:
            goal_tree_ix = 14
        elif conv_ix == 5:
            goal_tree_ix = 12
        elif conv_ix == 7:
            goal_tree_ix = 10
        
        if goal_tree_ix is not None:
            goal_filter_ix = filter_ix + unet_tree_ix_2_skip_connection_start(conv_tree_ixs[goal_tree_ix])
            conn_destinations.append((goal_tree_ix, goal_filter_ix))

        # outc has no next convolution
        if conv_ix == 18:
            conn_destinations = []
        
        return conn_destinations


        

        # if idx == 6:
        #     next_idxs_list.append



        # # output is: [(goal_tree_ix_1, goal_filter_ix_1), (goal_tree_ix_2, goal_filter_ix_2),...] 
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


    def IPAD_filter_importance_lambda_generator(L1_ADC_weight):
        assert L1_ADC_weight > 0 and L1_ADC_weight < 1, "L1_ADC_weight must be between 0 and 1."
        
        def IPAD_filter_importance_lambda(activations, conv_tree_ixs):
            # Returns dict tree_ix_2_list_of_filter_importances
            # The ix-th importance is for the filter currently on the ix-th place.
            # To convert this ix to the initial unpruned models filter ix, use the pruner's
            # state of active filters.

            tree_ix_2_filter_importances = {}
            for tree_ix in conv_tree_ixs:

                curr_batch_outputs = activations[tree_ix]
                print("len(curr_batch_outputs):")
                print(len(curr_batch_outputs))
                print("curr_batch_outputs[0].shape:")
                print(curr_batch_outputs[0].shape)
                curr_batch_outputs = torch.stack(tuple(curr_batch_outputs), dim=(0))
                # print(curr_batch_outputs.shape)
                # print(type(curr_batch_outputs))
                # print(curr_batch_outputs)
                filters_average_activation = curr_batch_outputs.mean(dim=(0))
                # print(filters_average_activation.shape)
                # print(filters_average_activation)
                overall_average_activation = filters_average_activation.mean(dim=(0))
                print(overall_average_activation)
                # print(overall_average_activation.shape)
                # print(overall_average_activation)
                h = filters_average_activation.shape[1]
                w = filters_average_activation.shape[2]
                L1_ADC = torch.abs(filters_average_activation - overall_average_activation).sum(dim=(1,2)) / (h*w)
                L2_ADC = (filters_average_activation - overall_average_activation).pow(2).sum(dim=(1,2)).sqrt() / (h*w)
                filter_importance = L1_ADC_weight * L1_ADC + (1 - L1_ADC_weight) * L2_ADC
                print(f"L1_ADC: {L1_ADC}")
                print(f"L2_ADC: {L2_ADC}")
                print(filter_importance.shape)
                print(filter_importance)

                tree_ix_2_filter_importances[tree_ix] = filter_importance
            
            return tree_ix_2_filter_importances
            
        return IPAD_filter_importance_lambda
            
        

    def test_matrix_creator(number):
        return torch.Tensor([[number, number, number], [number, number, number], [number, number, number]])
    
    def test_tensor_creator():
        return torch.stack([test_matrix_creator(i) for i in range(4)], dim=(0))

    test_conv_tree_ixs = [0]

    test_activations = {0 : [test_tensor_creator() for _ in range(16)]}

    # for 16 batches, 4 filters, 3x3 activation map, this needs to be a list of 16 tensors of shape (4, 3, 3)
    # average filter for it should have 1.5 everywhere
    # L1_ADC should be [1.5, 0.5, 0.5, 1.5], since it is just the mean of the abses of the differences
    # L2_ADC should be, for the first index: sqrt(3*3* 1.5**2) / 3*3
    # equals [sqrt(3*3) * sqrt(1.5**2) / 3*3,... ]
    # equals 3 * sqrt(1.5**2) / 9 = 1.5 / 3 = 0.5
    # So for the 0th element, the IPAD should be 0.5 * 1.5 + 0.5 * 0.5 = 0.75 + 0.25 = 1

    # and for the second index: sqrt(3*3* 0.5**2) / 3*3 = 0.5 / 3 = 0.16666666666666666
    # so IPAD should be 0.5 * 0.5 + 0.5 * 0.16666666666666666 = 0.25 + 0.08333333333333333 = 0.3333333333333333

    # the test passes

    test_IPAD_func = IPAD_filter_importance_lambda_generator(0.5)

    test_final_importances = test_IPAD_func(test_activations, test_conv_tree_ixs)
    print(test_final_importances) 

    input("Press enter to continue.")




    with open("initial_conv_resource_calc.pkl", "rb") as f:
        initial_resource_calc = pickle.load(f)

    pruner_instance = pruner(min_res_percents, initial_resource_calc, unet_resource_lambda, IPAD_filter_importance_lambda_generator(0.5), conv_tree_ixs)

    print(pruner_instance.initial_conv_resource_calc.module_tree_ix_2_weights_dimensions)

    pruner_instance.prune(activations)















"""
WHAT IS THE SETUP FOR THE PRUNING PROCESS IN THE PREVIOUS CODE:

It all happens in remove_filters_and_retrain_model()
- in their step # 1:
    - they simply run the model. This fills activations (the literal Unet code has saving activations in each block's forward method)
    - they calculate which filters are below the treshold and shuldn't be pruned
- in their step # 2:
    - they calculate the importance of each filter (_get_sorted_filter_importance_dict_for_model())
    - they get the least important filter, where they take into account the unprunable filters from step #1 (get_filter_with_minimal_importance()) 
"""


_original_sizes = {}
def _get_sorted_filter_importance_dict_for_model(model, omega, args):
    filter_weights_norms = {}  # for each filter: ||W_lk||_p
    #filter_activations_l2sq = {}  # for each filter: ||F_lk||^2
    filter_activation_diffs = {}  # for each filter: avg(|F_lk - F_lavg|)
    sizes = {}  # for each layer: kernel_width, kernel_height
    layer_n = {}  # for each layer: input_channels * kernel_width * kernel_height (= filter size)
    layer_m = {}  # for each layer: output_channels (= number of filters in layer)
    #layer_a = {}  # for each layer: activation_width * activation_height
    #layer_weights_l2sq_sum_normed = {}  # for each layer: 1 / (n_filters_l * filter_size_l) * sum_k(||W_lk||^2)

    for name, _ in model.named_parameters():  # always weight and bias
        if name.endswith('_activations_sum'):
            block_name, _, activation_sum_name = name.rpartition('.')
            block = op.attrgetter(block_name)(model)  # Need to use attrgetter since block_name can include dots in UNet
            layer = getattr(block, activation_sum_name.replace('_activations_sum', ''))
            activations = getattr(block, activation_sum_name.replace('_sum', '')).detach().cpu().numpy()

            weights = layer.weight.detach()
            if args.interpolate:
                weights = F.interpolate(weights, size=(3, 3), mode='bilinear', align_corners=False)
            if args.norm % 2:  # In odd norms we need to use absolute values
                weights = weights.abs()
            if args.norm == 1:
                layer_weights_norm_tensor = weights.sum(dim=[1, 2, 3])
            elif args.norm == 2:
                layer_weights_norm_tensor = weights.pow(2).sum(dim=[1, 2, 3]).sqrt()
            else:
                layer_weights_norm_tensor = weights.pow(args.norm).sum(dim=[1, 2, 3]).pow(1 / args.norm)

            #layer_activations_l2sq_tensor = getattr(block, activations_name)
            #assert layer_weights_l2sq_tensor.shape == layer_activations_l2sq_tensor.shape

            name_without_activations_sum_suffix = name.replace('_activations_sum', '')  # down_block1.conv1
            sizes[name_without_activations_sum_suffix] = layer.kernel_size
            layer_n[name_without_activations_sum_suffix] = layer.in_channels if args.interpolate else np.prod((layer.in_channels, *layer.kernel_size))
            layer_m[name_without_activations_sum_suffix] = layer.out_channels
            #layer_a[name_without_activations_sum_suffix] = block.activation_height * block.activation_width
            #mn = layer_n[name_without_activations_sum_suffix] * layer_m[name_without_activations_sum_suffix]
            #layer_weights_l2sq_sum_normed[name_without_activations_sum_suffix] = layer_weights_l2sq_tensor.sum() / mn
            #layer_activations_l2sq_sum_normed[name_without_activations_sum_suffix] = layer_activations_l2sq_tensor.sum() / mn

            # My activation criterion
            avg_activation = activations.mean(axis=(0,1))  # Mean activation over the batch: 2D matrix of activation pixels
            filter_activations = activations.mean(axis=0)  # Mean per-filter activations over the batch: 3D matrix - 1st dimension is filters, 2nd and 3rd are activation pixels
            diff_matrices = filter_activations - avg_activation  # Differential matrix D_k for each filter: 3D matrix - 1st dimension is filters, 2nd and 3rd are difference pixels
            if args.norm % 2:  # In odd norms we need to use absolute values
                diff_matrices = np.abs(diff_matrices)
            if args.norm == 1:
                diff_means = diff_matrices.mean(axis=(1,2))
            elif args.norm == 2:
                diff_means = np.sqrt(np.square(diff_matrices).sum(axis=(1,2))) / (diff_matrices.shape[1] * diff_matrices.shape[2])
            else:
                diff_means = np.power(np.power(diff_matrices, args.norm).sum(axis=(1,2)), 1 / args.norm) / (diff_matrices.shape[1] * diff_matrices.shape[2])

            #for k, (weights_norm, activations_l2sq) in enumerate(zip(layer_weights_norm_tensor, layer_activations_l2sq_tensor)):
            for k, (weights_norm, diff_mean) in enumerate(zip(layer_weights_norm_tensor, diff_means)):
                dict_key = '{0}-{1}'.format(name_without_activations_sum_suffix, k)  # down_block1.conv1-15
                filter_weights_norms[dict_key] = weights_norm
                #filter_activations_l2sq[dict_key] = activations_l2sq
                filter_activation_diffs[dict_key] = diff_mean

    #total_weights_sum_normed = torch.cat(layer_weights_l2sq_sum_normed.values()).sum()
    #total_activations_sum_normed = torch.cat(layer_activations_l2sq_sum_normed.values()).sum()

    # Print ranges to help determine the equalizer (I care mainly about the min values, since that's closer to what determines the pruned filters)
    #weight_importance_array = np.array([filter_weights_l2sq[k].item() / layer_n[k.rsplit('-', 1)[0]] for k in filter_weights_l2sq])
    #activation_importance_array = np.array([filter_activations_l2sq[k].item() / layer_a[k.rsplit('-', 1)[0]] for k in filter_activations_l2sq])  # Old
    #activation_importance_array = np.array([filter_activation_diffs[k].item() for k in filter_activation_diffs])  # New
    #print(f"Weights: {sorted(np.partition(weight_importance_array, 5)[:5])}–{np.median(weight_importance_array)}–{weight_importance_array.max()}")
    #print(f"Activations: {sorted(np.partition(activation_importance_array, 5)[:5])}–{np.median(activation_importance_array)}–{activation_importance_array.max()}")

    # Compute correlation between the rankings of the two importance criteria (requires weight_importance_array and activation_importance_array from above)
    #weight_ranking = weight_importance_array.argsort()
    #activation_ranking = activation_importance_array.argsort()
    #print(f"Weight ranking:\n{np.array(list(filter_weights_l2sq.keys()))[weight_ranking][:200]}")
    #print(f"Activation ranking:\n{np.array(list(filter_weights_l2sq.keys()))[activation_ranking][:200]}")
    #print(f"Spearman's rank-order correlation: {scipy.stats.spearmanr(weight_ranking, activation_ranking)}")

    activation_equalizer = 10  # experimentally determined ratio to bring the activation criterion roughly to the same order of magnitude as the weight criterion
    all_importances_dict = {}
    for k in filter_weights_norms:
        l = k.rsplit('-', 1)[0]  # k = down_block1.conv1-15, l = down_block1.conv1
        if args.random:
            all_importances_dict[k] = torch.rand(1, device=filter_weights_norms[k].device)  # For random pruning use random filter importance
        elif args.uniform:
            if l not in _original_sizes:  # This is a hacky way of saving the original sizes on the first run of this function (before any pruning is done)
                _original_sizes[l] = layer_m[l]
            # For uniform pruning use the proportion of pruned filterss (so less pruned layers get pruned first) with a small random perturbation (to prune a random filter inside the chosen layer)
            all_importances_dict[k] = 1 - (layer_m[l] / _original_sizes[l]) + torch.distributions.Uniform(-1e3, 1e3).sample().to(filter_weights_norms[k].device)
        else:
            all_importances_dict[k] = (
                omega / layer_n[l] * filter_weights_norms[k] +  # Weight criterion
                #(1 - omega) * activation_equalizer / layer_a[l] * filter_activations_l2sq[k] #+  # Old activation criterion
                (1 - omega) * activation_equalizer * filter_activation_diffs[k] #+  # New activation criterion
            ) if not args.channelsUseWeightsOnly or sizes[l] != (1, 1) else filter_weights_norms[k] / layer_n[l]

    return dict(sorted(all_importances_dict.items(), key=op.itemgetter(1)))

def get_filter_with_minimal_importance(device, all_importances_dict_sorted, layers_with_exceeded_limit, blocks_with_exceeded_limit):
    # we must return filter with nonzero activation, because zeroed filters are not used in network
    # ALSO set selected filter's activation to zero!
    # and skip layers in layers_with_exceeded_limit or blocks in blocks_with_exceeded_limit
    for key, value in all_importances_dict_sorted.items():
        zero_tensor = torch.tensor(0, dtype=torch.float32).to(device)
        if torch.equal(value, zero_tensor):
            continue

        layer_name, _ = get_parameter_name_and_index_from_activations_dict_key(key)

        if layer_name in layers_with_exceeded_limit:
            #logger.write('layer {0} skipped because it has exceeded percent limit'.format(layer_name))
            continue
        block_name, _, _ = layer_name.rpartition('.')
        if block_name in blocks_with_exceeded_limit:
            #logger.write('layer {0} skipped because block {1} has exceeded percent limit'.format(layer_name, block_name))
            continue

        all_importances_dict_sorted[key] = zero_tensor
        return key, value#, all_importances_dict_sorted



"""
HOW IS PRUNING DONE IN THE PREVIOUS CODE:

They get the layer through the name as I did above.
They do it in two steps, actually:
disable_filter(device, model, name_index):
- get the full lower level block name block name (like down1.conv.conv1, up4.conv.conv1, etc.)
- do surgery on it's weights
- prune the next batch norm if needed (_prune_next_bn_if_needed())
- from the name, get what layer_index it is (_get_layer_index())
- from the layer_index, get a list of layer indexes of the next convolutions (_get_next_conv_id_list_recursive())
- go through the list and prune based on these layer:indexes (_prune_next_layer())
     - this calls _layer_index_to_conv, which returns the pointer to layer to prune.
     It literally gets the layer by the name like this: (calculates names of blocks based on the index)
     layer_index -= 2
            du = 'up' if layer_index // 8 else 'down'
            block = layer_index % 8 // 2 + 1
            layer = layer_index % 2 + 1
            block = op.attrgetter(f'{du}{block}.conv')(model)

!!!! i get it now - (but in this last step the batch norm isn't pruned, which I don't understand)
The next layers only have their input size changed, not the output.
The layer is:   filter_num x width x height x input_channels
The initial pruned layer gets_ filter_num -= 1
The subsequent layers get their input_channels -= 1, so their output is actually the same.
It's a different kind of pruning.
"""



def _get_layer_index(name, model):
    model = type(model).__name__.lower()
    if 'densenet' in model:
        if name == 'down_block1.conv1':
            return 0
        elif name == 'down_block1.conv21':
            return 1
        elif name == 'down_block1.conv22':
            return 2
        elif name == 'down_block1.conv31':
            return 3
        elif name == 'down_block1.conv32':
            return 4
        elif name == 'down_block2.conv1':
            return 5
        elif name == 'down_block2.conv21':
            return 6
        elif name == 'down_block2.conv22':
            return 7
        elif name == 'down_block2.conv31':
            return 8
        elif name == 'down_block2.conv32':
            return 9
        elif name == 'down_block3.conv1':
            return 10
        elif name == 'down_block3.conv21':
            return 11
        elif name == 'down_block3.conv22':
            return 12
        elif name == 'down_block3.conv31':
            return 13
        elif name == 'down_block3.conv32':
            return 14
        elif name == 'down_block4.conv1':
            return 15
        elif name == 'down_block4.conv21':
            return 16
        elif name == 'down_block4.conv22':
            return 17
        elif name == 'down_block4.conv31':
            return 18
        elif name == 'down_block4.conv32':
            return 19
        elif name == 'down_block5.conv1':
            return 20
        elif name == 'down_block5.conv21':
            return 21
        elif name == 'down_block5.conv22':
            return 22
        elif name == 'down_block5.conv31':
            return 23
        elif name == 'down_block5.conv32':
            return 24

        elif name == 'up_block1.conv11':
            return 25
        elif name == 'up_block1.conv12':
            return 26
        elif name == 'up_block1.conv21':
            return 27
        elif name == 'up_block1.conv22':
            return 28
        elif name == 'up_block2.conv11':
            return 29
        elif name == 'up_block2.conv12':
            return 30
        elif name == 'up_block2.conv21':
            return 31
        elif name == 'up_block2.conv22':
            return 32
        elif name == 'up_block3.conv11':
            return 33
        elif name == 'up_block3.conv12':
            return 34
        elif name == 'up_block3.conv21':
            return 35
        elif name == 'up_block3.conv22':
            return 36
        elif name == 'up_block4.conv11':
            return 37
        elif name == 'up_block4.conv12':
            return 38
        elif name == 'up_block4.conv21':
            return 39
        elif name == 'up_block4.conv22':
            return 40
        elif name.startswith('out_conv1'):
            return 41
        else:
            raise Exception('Neki je narobe pri layer index')

    elif 'unet' in model:
        return [
            'inc.conv1', 'inc.conv2',                # 0 1
            'down1.conv.conv1', 'down1.conv.conv2',  # 2 3
            'down2.conv.conv1', 'down2.conv.conv2',  # 4 5
            'down3.conv.conv1', 'down3.conv.conv2',  # 6 7
            'down4.conv.conv1', 'down4.conv.conv2',  # 8 9
            'up1.conv.conv1', 'up1.conv.conv2',     # 10 11
            'up2.conv.conv1', 'up2.conv.conv2',     # 12 13
            'up3.conv.conv1', 'up3.conv.conv2',     # 14 15
            'up4.conv.conv1', 'up4.conv.conv2',     # 16 17
            'outc.conv'                             # 18
        ].index(name)

    else:
        raise ValueError(f"Unknown model {model}")


def _get_next_conv_id_list_recursive(layer_index, model):
    model = type(model).__name__.lower()
    if 'densenet' in model:
        if layer_index == 0:
            next_conv_idx = [1, 3]
        elif layer_index == 1:
            next_conv_idx = [2]
        elif layer_index == 2:
            next_conv_idx = [3]
        elif layer_index == 3:
            next_conv_idx = [4]
        elif layer_index == 4:
            next_conv_idx = [5, 6, 8, 37, 39]
        elif layer_index == 5:
            next_conv_idx = [6, 8]
        elif layer_index == 6:
            next_conv_idx = [7]
        elif layer_index == 7:
            next_conv_idx = [8]
        elif layer_index == 8:
            next_conv_idx = [9]
        elif layer_index == 9:
            next_conv_idx = [10, 11, 13, 33, 35]
        elif layer_index == 10:
            next_conv_idx = [11, 13]
        elif layer_index == 11:
            next_conv_idx = [12]
        elif layer_index == 12:
            next_conv_idx = [13]
        elif layer_index == 13:
            next_conv_idx = [14]
        elif layer_index == 14:
            next_conv_idx = [15, 16, 18, 29, 31]
        elif layer_index == 15:
            next_conv_idx = [16, 18]
        elif layer_index == 16:
            next_conv_idx = [17]
        elif layer_index == 17:
            next_conv_idx = [18]
        elif layer_index == 18:
            next_conv_idx = [19]
        elif layer_index == 19:
            next_conv_idx = [20, 21, 23, 25, 27]
        elif layer_index == 20:
            next_conv_idx = [21, 23]
        elif layer_index == 21:
            next_conv_idx = [22]
        elif layer_index == 22:
            next_conv_idx = [23]
        elif layer_index == 23:
            next_conv_idx = [24]
        elif layer_index == 24:
            next_conv_idx = [25, 27]
        # UP BLOCKS:
        elif layer_index == 25:
            next_conv_idx = [26]
        elif layer_index == 26:
            next_conv_idx = [27]
        elif layer_index == 27:
            next_conv_idx = [28]
        elif layer_index == 28:
            next_conv_idx = [29, 31]
        elif layer_index == 29:
            next_conv_idx = [30]
        elif layer_index == 30:
            next_conv_idx = [31]
        elif layer_index == 31:
            next_conv_idx = [32]
        elif layer_index == 32:
            next_conv_idx = [33, 35]
        elif layer_index == 33:
            next_conv_idx = [34]
        elif layer_index == 34:
            next_conv_idx = [35]
        elif layer_index == 35:
            next_conv_idx = [36]
        elif layer_index == 36:
            next_conv_idx = [37, 39]
        elif layer_index == 37:
            next_conv_idx = [38]
        elif layer_index == 38:
            next_conv_idx = [39]
        elif layer_index == 39:
            next_conv_idx = [40]
        elif layer_index == 40:
            next_conv_idx = [41]
        elif layer_index == 41:
            next_conv_idx = []
        else:
            raise Exception("Error occured")

    elif 'unet' in model:
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

    else:
        raise ValueError(f"Unknown model {model}")

    # recursion call
    result_list = next_conv_idx.copy()
    # for id in next_conv_idx:
    #    result_list = result_list + self._get_next_conv_id_list_recursive(id)

    return result_list





def _layer_index_to_conv(layer_index, model):
    model_name = type(model).__name__.lower()
    if 'densenet' in model_name:
        if layer_index == 0:
            block_name = 'down_block1'
            conv_name = 'conv1'
        elif layer_index == 1:
            block_name = 'down_block1'
            conv_name = 'conv21'
        elif layer_index == 2:
            block_name = 'down_block1'
            conv_name = 'conv22'
        elif layer_index == 3:
            block_name = 'down_block1'
            conv_name = 'conv31'
        elif layer_index == 4:
            block_name = 'down_block1'
            conv_name = 'conv32'
        elif layer_index == 5:
            block_name = 'down_block2'
            conv_name = 'conv1'
        elif layer_index == 6:
            block_name = 'down_block2'
            conv_name = 'conv21'
        elif layer_index == 7:
            block_name = 'down_block2'
            conv_name = 'conv22'
        elif layer_index == 8:
            block_name = 'down_block2'
            conv_name = 'conv31'
        elif layer_index == 9:
            block_name = 'down_block2'
            conv_name = 'conv32'
        elif layer_index == 10:
            block_name = 'down_block3'
            conv_name = 'conv1'
        elif layer_index == 11:
            block_name = 'down_block3'
            conv_name = 'conv21'
        elif layer_index == 12:
            block_name = 'down_block3'
            conv_name = 'conv22'
        elif layer_index == 13:
            block_name = 'down_block3'
            conv_name = 'conv31'
        elif layer_index == 14:
            block_name = 'down_block3'
            conv_name = 'conv32'
        elif layer_index == 15:
            block_name = 'down_block4'
            conv_name = 'conv1'
        elif layer_index == 16:
            block_name = 'down_block4'
            conv_name = 'conv21'
        elif layer_index == 17:
            block_name = 'down_block4'
            conv_name = 'conv22'
        elif layer_index == 18:
            block_name = 'down_block4'
            conv_name = 'conv31'
        elif layer_index == 19:
            block_name = 'down_block4'
            conv_name = 'conv32'
        elif layer_index == 20:
            block_name = 'down_block5'
            conv_name = 'conv1'
        elif layer_index == 21:
            block_name = 'down_block5'
            conv_name = 'conv21'
        elif layer_index == 22:
            block_name = 'down_block5'
            conv_name = 'conv22'
        elif layer_index == 23:
            block_name = 'down_block5'
            conv_name = 'conv31'
        elif layer_index == 24:
            block_name = 'down_block5'
            conv_name = 'conv32'

        elif layer_index == 25:
            block_name = 'up_block1'
            conv_name = 'conv11'
        elif layer_index == 26:
            block_name = 'up_block1'
            conv_name = 'conv12'
        elif layer_index == 27:
            block_name = 'up_block1'
            conv_name = 'conv21'
        elif layer_index == 28:
            block_name = 'up_block1'
            conv_name = 'conv22'
        elif layer_index == 29:
            block_name = 'up_block2'
            conv_name = 'conv11'
        elif layer_index == 30:
            block_name = 'up_block2'
            conv_name = 'conv12'
        elif layer_index == 31:
            block_name = 'up_block2'
            conv_name = 'conv21'
        elif layer_index == 32:
            block_name = 'up_block2'
            conv_name = 'conv22'
        elif layer_index == 33:
            block_name = 'up_block3'
            conv_name = 'conv11'
        elif layer_index == 34:
            block_name = 'up_block3'
            conv_name = 'conv12'
        elif layer_index == 35:
            block_name = 'up_block3'
            conv_name = 'conv21'
        elif layer_index == 36:
            block_name = 'up_block3'
            conv_name = 'conv22'
        elif layer_index == 37:
            block_name = 'up_block4'
            conv_name = 'conv11'
        elif layer_index == 38:
            block_name = 'up_block4'
            conv_name = 'conv12'
        elif layer_index == 39:
            block_name = 'up_block4'
            conv_name = 'conv21'
        elif layer_index == 40:
            block_name = 'up_block4'
            conv_name = 'conv22'
        elif layer_index == 41:
            return getattr(model, 'out_conv1'), None, 'out_conv1'
        else:
            raise Exception('neki je narobe pri pridobivanju conv iz layer indexa')

        block = getattr(model, block_name)

    elif 'unet' in model_name:
        if layer_index == 18:
            return model.outc.conv, None, 'conv'
        if layer_index < 2:
            layer = layer_index + 1
            block = model.inc
        else:
            layer_index -= 2
            du = 'up' if layer_index // 8 else 'down'
            block = layer_index % 8 // 2 + 1
            layer = layer_index % 2 + 1
            block = op.attrgetter(f'{du}{block}.conv')(model)
        conv_name = f'conv{layer}'

    else:
        raise ValueError(f"Unknown model {model_name}")

    return getattr(block, conv_name), block, conv_name


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



def _prune_next_bn_if_needed(layer_index, filters_begin, filters_end, pruned_filters, device, model):
    model_name = type(model).__name__.lower()
    if 'densenet' in model_name:
        next_bn_index = None
        if layer_index == 4:  # layer_index == 0 or layer_index == 1 or layer_index == 2 or layer_index == 3 or layer_index == 4:
            next_bn_index = 4  # 4, 9, 14, 19, 24
        elif layer_index == 9:  # layer_index == 5 or layer_index == 6 or layer_index == 7 or layer_index == 8 or layer_index == 9:
            next_bn_index = 9  # 4, 9, 14, 19, 24
        elif layer_index == 14:  # layer_index == 10 or layer_index == 11 or layer_index == 12 or layer_index == 13 or layer_index == 14:
            next_bn_index = 14  # 4, 9, 14, 19, 24
        elif layer_index == 19:  # layer_index == 15 or layer_index == 16 or layer_index == 17 or layer_index == 18 or layer_index == 19:
            next_bn_index = 19  # 4, 9, 14, 19, 24
        elif layer_index == 24:  # layer_index == 20 or layer_index == 21 or layer_index == 22 or layer_index == 23 or layer_index == 24:
            next_bn_index = 24  # 4, 9, 14, 19, 24
        else:
            next_bn = None
    elif 'unet' in model_name:
        next_bn_index = layer_index if layer_index < 18 else None  # outc doesn't have a BN layer
    else:
        raise ValueError(f"Unknown model {model_name}")


    if next_bn_index is not None:
        next_bn = _get_bn_by_prev_conv_index(next_bn_index, model)

    # Surgery on next batchnorm layer
    if next_bn is not None:
        logger.write('additionally pruning batch norm with index {0}'.format(next_bn_index))
        logger.write('n features compressed from {0} to {1} '.format(next_bn.num_features, next_bn.num_features - pruned_filters))
        next_new_bn = \
            torch.nn.BatchNorm2d(num_features=next_bn.num_features - pruned_filters, \
                                 eps=next_bn.eps, \
                                 momentum=next_bn.momentum, \
                                 affine=next_bn.affine,
                                 track_running_stats=next_bn.track_running_stats)
        next_bn.num_features -= pruned_filters

        old_weights = next_bn.weight.data.cpu().numpy()
        new_weights = next_new_bn.weight.data.cpu().numpy()
        old_bias = next_bn.bias.data.cpu().numpy()
        new_bias = next_new_bn.bias.data.cpu().numpy()
        old_running_mean = next_bn.running_mean.data.cpu().numpy()
        new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
        old_running_var = next_bn.running_var.data.cpu().numpy()
        new_running_var = next_new_bn.running_var.data.cpu().numpy()

        new_weights[: filters_begin] = old_weights[: filters_begin]
        new_weights[filters_begin:] = old_weights[filters_end + 1:]
        #next_bn.weight.data = torch.from_numpy(new_weights).to(device)
        next_bn.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(device))
        next_bn.weight.grad = None

        new_bias[: filters_begin] = old_bias[: filters_begin]
        new_bias[filters_begin:] = old_bias[filters_end + 1:]
        #next_bn.bias.data = torch.from_numpy(new_bias).to(device)
        next_bn.bias = torch.nn.Parameter(torch.from_numpy(new_bias).to(device))
        next_bn.bias.grad = None

        new_running_mean[: filters_begin] = old_running_mean[: filters_begin]
        new_running_mean[filters_begin:] = old_running_mean[filters_end + 1:]
        next_bn.running_mean.data = torch.from_numpy(new_running_mean).to(device)
        #next_bn.running_mean = torch.nn.Parameter(torch.from_numpy(new_running_mean).to(device))
        next_bn.running_mean.grad = None

        new_running_var[: filters_begin] = old_running_var[: filters_begin]
        new_running_var[filters_begin:] = old_running_var[filters_end + 1:]
        next_bn.running_var.data = torch.from_numpy(new_running_var).to(device)
        #next_bn.running_var = torch.nn.Parameter(torch.from_numpy(new_running_var).to(device))
        next_bn.running_var.grad = None
