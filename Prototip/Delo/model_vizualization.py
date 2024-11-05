




import logging
import python_logger.log_helper as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)





from torch import nn
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from model_sorting import sort_tree_ixs, denest_tuple
from ConvResourceCalc import ConvResourceCalc





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
            




def draw_tree(ix, layer_name, ax, x, y, width, height, max_depth, resource_calc: Union[None, ConvResourceCalc], initial_resource_calc: Union[None, ConvResourceCalc], pruner, lowest_level_modules):
    # Draw the rectangle and label it


    this_name_tree_ixs = resource_calc.get_ordered_list_of_tree_ixs_for_layer_name(layer_name)
    # which of its' name is it?
    ordered_ix = this_name_tree_ixs.index(ix)



    display_string = f'{ordered_ix}. {layer_name}\n{ix}'


    
            
    if ix in lowest_level_modules:
        lowest_level_modules_index = lowest_level_modules.index(ix)
        display_string += f"\n{lowest_level_modules_index}. LLM"
    


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
            initial_dim_size = initial_resource_calc.resource_name_2_resource_dict["weights_dimensions"][ix][0]
            display_string += f"\nKernels pruned: [{string_of_pruned(list_of_active_initial_kernel_ixs, initial_dim_size)}]"
        
        if ix in pruner.tree_ix_2_list_of_initial_input_slice_ixs.keys():
            list_of_active_initial_input_slice_ixs = pruner.tree_ix_2_list_of_initial_input_slice_ixs[ix]
            initial_dim_size = initial_resource_calc.resource_name_2_resource_dict["weights_dimensions"][ix][1]
            display_string += f"\nInput slices pruned: [{string_of_pruned(list_of_active_initial_input_slice_ixs, initial_dim_size)}]"


    

    

    ax.add_patch(patches.Rectangle((x, y), width, height, edgecolor='black', facecolor='none'))
    ax.text(x + width/2, y + height/2, display_string, ha='center', va='center')

    # Find children of the current index
    children = [key for key in resource_calc.module_tree_ix_2_name if key[0] == ix]
    if children:
        child_width = width / len(children)
        for i, child in enumerate(sort_tree_ixs(children)):
            child_name = resource_calc.module_tree_ix_2_name[child]
            draw_tree(child, child_name, ax, x + i * child_width, y - height, child_width, height, max_depth - 1, resource_calc, initial_resource_calc, pruner, lowest_level_modules)





def model_graph(resource_calc, initial_resource_calc=None, pruner=None, width=1, height=0.1):

    fig, ax = plt.subplots()

    lowest_level_modules = []
    for tree_ix, children_list in resource_calc.module_tree_ix_2_children_tree_ix_list.items():
        if len(children_list) == 0:
            lowest_level_modules.append(tree_ix)
    
    tree = resource_calc.module_tree_ix_2_name
    max_depth = max(len(denest_tuple(k)) for k in tree.keys())
    total_height = max_depth * height
    root_ix = (0,)
    root_name = tree[root_ix]
    draw_tree(root_ix, root_name, ax, 0, total_height, width, height, max_depth, resource_calc, initial_resource_calc, pruner, lowest_level_modules)


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show(block=False)









