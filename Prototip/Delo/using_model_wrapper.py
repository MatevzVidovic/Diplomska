
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
    input_example = torch.randn(1, 1, 128, 128)
    resource_calc.calculate_resources(input_example)
    FLOPs = resource_calc.all_flops_num
    resource_dict = resource_calc.module_tree_ix_2_flops_num

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

    # print(resource_calc.module_tree_ix_2_module_itself)

    print(resource_calc.module_tree_ix_2_name)


    conv_modules_tree_ixs = []
    for key, value in resource_calc.module_tree_ix_2_name.items():
        if value == "Conv2d":
            conv_modules_tree_ixs.append(key)
    

    print(conv_modules_tree_ixs)


    activations = {}

    def set_activations_hooks(activations: dict, tree_ixs, resource_calc: ConvResourceCalc):
    
        def get_activation(tree_ix):
            def hook(model, input, output):
                if tree_ix not in activations:
                    activations[tree_ix] = []
                activations[tree_ix].append(output.detach())
            return hook

        tree_ix_2_hook_handle = {}
        for tree_ix in tree_ixs:
            module = resource_calc.module_tree_ix_2_module_itself[tree_ix]
            tree_ix_2_hook_handle[tree_ix] = module.register_forward_hook(get_activation(tree_ix))
    
    set_activations_hooks(activations, conv_modules_tree_ixs, resource_calc)

    
    # print(activations)

    wrap_model.model.eval()
    with torch.no_grad():
        for i in range(10):
            input_tensor = torch.randn(1, 1, 128, 128)
            model(input_tensor)
    
    # print(activations)

    activations.clear()

    wrap_model.model.eval()
    with torch.no_grad():
        for i in range(10):
            input_tensor = torch.randn(1, 1, 128, 128)
            model(input_tensor)

    # print(f"Number of activations for [((((0,), 0), 0), 0)]: {len([((((0,), 0), 0), 0)])}")
    # print(activations[((((0,), 0), 0), 0)][0].shape)
    # input()


    # input_tensor = torch.randn(1, 1, 128, 128)
    # wrap_model.model.eval()
    # with torch.no_grad():
    #     model(input_tensor)
    

    # print(activations[((((0,), 0), 0), 0)].shape)




    # # This shows how to remove hooks when they are no longer needed.
    # # This can save memory.
    # for tree_ix, hook_handle in tree_ix_2_hook_handle.items():
    #     hook_handle.remove()

    




    FLOPS_min_res_percents = min_resource_percentage(resource_calc.module_tree_ix_2_name)
    FLOPS_min_res_percents.set_by_name("Conv2d", 0.5)

    tree_ix_2_percentage_dict = {
        (0,) : 0.2,
        ((0,), 0) : 0.2,
    }
    FLOPS_min_res_percents.set_by_tree_ix_dict(tree_ix_2_percentage_dict)

    # print(FLOPS_min_res_percents.min_resource_percentage_dict)
    # input()

    weights_min_res_percents = min_resource_percentage(resource_calc.module_tree_ix_2_name)
    weights_min_res_percents.set_by_name("Conv2d", 0.2)





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

    module_full_names = generator_to_list(resource_calc.module_tree_ix_2_module_itself[(0,)].named_modules(remove_duplicate=False))
    print(module_full_names)
    # print(first_element(resource_calc.module_tree_ix_2_module_itself[((None, 0), 0)].named_modules(remove_duplicate=False)))
    # print(second_element(resource_calc.module_tree_ix_2_module_itself[((None, 0), 0)].named_modules(remove_duplicate=False)))

    print(generator_to_list(resource_calc.module_tree_ix_2_module_itself[((0,), 0)].named_modules(remove_duplicate=False)))

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
    for tree_ix, children_list in resource_calc.module_tree_ix_2_children_tree_ix_list.items():
        if len(children_list) == 0:
            lowest_level_modules_tree_ixs.append(tree_ix)
    
    print(lowest_level_modules_tree_ixs)


    sorted_lowest_level_modules_tree_ixs = sort_tree_ixs(lowest_level_modules_tree_ixs)
    print(sorted_lowest_level_modules_tree_ixs)

    lowest_level_modules_names = [resource_calc.module_tree_ix_2_name[tree_ix] for tree_ix in sorted_lowest_level_modules_tree_ixs]
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

    tree_ix_2_layer_name = resource_calc.module_tree_ix_2_name

    def get_ordered_list_of_tree_ixs_for_layer_name(module_tree_ix_2_name, layer_name):
        
        applicable_tree_ixs = []
        for tree_ix, module_name in module_tree_ix_2_name.items():
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

    def string_of_pruned(list_of_initial_ixs, initial_dim_size):
        ix_is_in_initial_ixs_list = []
        for i in range(initial_dim_size):
            ix_is_in_initial_ixs_list.append(False)
        
        for ix in list_of_initial_ixs:
            try:
                ix_is_in_initial_ixs_list[ix] = True
            except:
                print(f"ix: {ix}")
                print(f"initial_dim_size: {initial_dim_size}")
                print(f"list_of_initial_ixs: {list_of_initial_ixs}")
                raise ValueError("Index out of range.")
        
        # When you come into a territory of False, you start a new section.
        # When you leave it, you end the section.
        # If we are in false territory at the end, we end the section.
        string = ""
        ix_in_False_territory = -1
        for i, is_in_initial_ixs in enumerate(ix_is_in_initial_ixs_list):
            if not is_in_initial_ixs:
                if ix_in_False_territory == -1:
                    string += f"{i}"
                ix_in_False_territory += 1
            else:
                if ix_in_False_territory >= 1:
                    string += f"-{i-1}, "
                elif ix_in_False_territory == 0:
                    string += f", "
                ix_in_False_territory = -1
        if ix_in_False_territory > 0:
            string += f"-{initial_dim_size}"

        return string
                



    def draw_tree(ix, layer_name, ax, x, y, width, height, max_depth, resource_calc: ConvResourceCalc, initial_resource_calc: ConvResourceCalc, pruner: pruner):
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
        
        # To mi ni všeč, ker gre prek resource calc in ne direktno. In torej če se zgodi, da ga ne posodobiš,
        # ker ti tu recimo reference-a star resource_calc objekt, bo to nasty bug.
        # Pa tudi sedaj itak rabim initial resource calc v nadaljevanju, in je bolje preprosto skozi njega vse delat.
        # if ix in resource_calc.module_tree_ix_2_weights_dimensions:
        #     display_string += f"\n{resource_calc.module_tree_ix_2_weights_dimensions[ix]}"

        # To je bolje, ker ta tu bo vsaka verzija resource calc-a bila pravilna. 
        layer = resource_calc.module_tree_ix_2_module_itself[ix]
        if type(layer) == nn.Conv2d:
            display_string += f"\n{list(layer.weight.shape)}"
        
        elif type(layer) == nn.BatchNorm2d:
            
            shapes = [list(layer.weight.shape), list(layer.bias.shape), 
                      list(layer.running_mean.shape), list(layer.running_var.shape)]
            
            # are all shapes the same?
            if all(shape == shapes[0] for shape in shapes):
                display_string += f"\n{shapes[0]}"
            else:
                for shape in shapes:
                    display_string += f"\n{shape}"
        

        # The display of what we have pruned:
        if pruner is not None and initial_resource_calc is not None:

            # print(display_string)

            if ix in pruner.tree_ix_2_list_of_initial_kernel_ixs.keys():
                list_of_active_initial_kernel_ixs = pruner.tree_ix_2_list_of_initial_kernel_ixs[ix]
                # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
                initial_dim_size = initial_resource_calc.module_tree_ix_2_weights_dimensions[ix][0]
                display_string += f"\nKernels pruned: [{string_of_pruned(list_of_active_initial_kernel_ixs, initial_dim_size)}]"
            
            if ix in pruner.tree_ix_2_list_of_initial_input_slice_ixs.keys():
                list_of_active_initial_input_slice_ixs = pruner.tree_ix_2_list_of_initial_input_slice_ixs[ix]
                initial_dim_size = initial_resource_calc.module_tree_ix_2_weights_dimensions[ix][1]
                display_string += f"\nInput slices pruned: [{string_of_pruned(list_of_active_initial_input_slice_ixs, initial_dim_size)}]"


        

        

        ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor='black', facecolor='none'))
        ax.text(x + width/2, y + height/2, display_string, ha='center', va='center')

        # Find children of the current index
        children = [key for key in tree_ix_2_layer_name if key[0] == ix]
        if children:
            child_width = width / len(children)
            for i, child in enumerate(sort_tree_ixs(children)):
                child_name = tree_ix_2_layer_name[child]
                draw_tree(child, child_name, ax, x + i * child_width, y - height, child_width, height, max_depth - 1, resource_calc, initial_resource_calc, pruner)

    def visualize_tree(ax, resource_calc: ConvResourceCalc, initial_resource_calc, pruner, width=1, height=0.1):
        tree = resource_calc.module_tree_ix_2_name
        max_depth = max(len(denest_tuple(k)) for k in tree.keys())
        total_height = max_depth * height
        root_ix = (0,)
        root_name = tree[root_ix]
        draw_tree(root_ix, root_name, ax, 0, total_height, width, height, max_depth, resource_calc, initial_resource_calc, pruner)



    def model_graph(resource_calc, initial_resource_calc=None, pruner=None):
        fig, ax = plt.subplots()
        visualize_tree(ax, resource_calc, initial_resource_calc, pruner)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.show(block=False)

    model_graph(resource_calc)

    # fig, ax = plt.subplots()
    # visualize_tree(tree_ix_2_layer_name, ax)
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.axis('off')
    # plt.show(block=False)













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
            
    







    # Testing the IPAD_kernel_importance_lambda_generator

    def test_one_kernel_creator(number):
        # make the batch_size 2, so the test is more real-life
        intermeds = torch.Tensor([[number, number, number], [number, number, number], [number, number, number]])
        result = torch.stack([intermeds for _ in range(2)], dim=(0))
        return result
    
    def test_kernel_combinator():
        return torch.stack([test_one_kernel_creator(i) for i in range(4)], dim=(1))

    test_conv_tree_ixs = [0]

    test_activations = {0 : [test_kernel_combinator() for _ in range(8)]}

    # for 16 batch_size, 4 kernels, 3x3 activation map, this needs to be a list of 16 tensors of shape (4, 3, 3)
    # overall average kernel for it should have 1.5 everywhere
    # L1_ADC should be [1.5, 0.5, 0.5, 1.5], since it is just the mean of the abs-es of the differences
    # L2_ADC should be, for the first index: sqrt(3*3* 1.5**2) / 3*3
    # equals [sqrt(3*3) * sqrt(1.5**2) / 3*3,... ]
    # equals 3 * sqrt(1.5**2) / 9 = 1.5 / 3 = 0.5
    # So for the 0th element, the IPAD should be 0.5 * 1.5 + 0.5 * 0.5 = 0.75 + 0.25 = 1

    # and for the second index: sqrt(3*3* 0.5**2) / 3*3 = 0.5 / 3 = 0.16666666666666666
    # so IPAD should be 0.5 * 0.5 + 0.5 * 0.16666666666666666 = 0.25 + 0.08333333333333333 = 0.3333333333333333

    # the test passes

    test_IPAD_func = IPAD_kernel_importance_lambda_generator(0.5)

    test_final_importances = test_IPAD_func(test_activations, test_conv_tree_ixs)

    # print(test_final_importances)

    # print(test_final_importances[test_conv_tree_ixs[0]])

    # print(test_final_importances[test_conv_tree_ixs[0]].shape)
    # print(len(test_activations[test_conv_tree_ixs[0]]))
    # print(test_activations[test_conv_tree_ixs[0]][0].shape)

    # input("Press enter to continue.")








    def load_initial_conv_resource_calc() -> ConvResourceCalc:
        with open("initial_conv_resource_calc.pkl", "rb") as f:
            initial_resource_calc = pickle.load(f)
        return initial_resource_calc
    
    initial_resource_calc = load_initial_conv_resource_calc()

    batch_norm_ixs = get_ordered_list_of_tree_ixs_for_layer_name(initial_resource_calc.module_tree_ix_2_name, "BatchNorm2d")
    pruner_instance = pruner(FLOPS_min_res_percents, weights_min_res_percents, initial_resource_calc, unet_connection_lambda, unet_inextricable_connection_lambda, conv_tree_ixs, batch_norm_ixs, lowest_level_modules)

    # print(pruner_instance.initial_conv_resource_calc.module_tree_ix_2_weights_dimensions)

    importance_lambda = IPAD_kernel_importance_lambda_generator(0.5)







    inp = ""
    while inp == "" or inp == "g":
    
        # print(activations)

        activations.clear()

        # set_activations_hooks(activations, conv_modules_tree_ixs, resource_calc)

        wrap_model.model.eval()
        with torch.no_grad():
            for i in range(10):
                input_tensor = torch.randn(1, 1, 128, 128)
                model(input_tensor)
        
        # print(activations)
        # print(f"Number of activations for [((((0,), 0), 0), 0)]: {len([((((0,), 0), 0), 0)])}")
        # print(activations[((((0,), 0), 0), 0)][0].shape)
        # input()
    
    
        # pruner needs the current state of model resources to know which modules shouldn't be pruned anymore
        resource_calc.calculate_resources(input_example)
        importance_dict = importance_lambda(activations, conv_tree_ixs)
        pruner_instance.prune(importance_dict, resource_calc, wrap_model)

        resource_calc.calculate_resources(input_example)
    
        if inp == "g":
            model_graph(resource_calc, initial_resource_calc, pruner_instance)
    
        inp = input("Press enter to continue, any text to stop, g to continue and show graph.\n")












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

