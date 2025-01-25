




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





from torch import nn
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from y_helpers.helper_model_sorting import sort_tree_ixs, denest_tuple
from conv_resource_calc import ConvResourceCalc





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
            



def get_string_of_pruned(tree_ix, initial_resource_calc, pruner):
    
    ix = tree_ix

    display_string = ""

    # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
    initial_weights_shape = initial_resource_calc.resource_name_2_resource_dict["weights_dimensions"][ix]
    
    if ix in pruner.tree_ix_2_list_of_initial_kernel_ixs.keys():
        list_of_active_initial_kernel_ixs = pruner.tree_ix_2_list_of_initial_kernel_ixs[ix]
        initial_dim_size = initial_weights_shape[0]
        display_string += f"\nKernels pruned: [{string_of_pruned(list_of_active_initial_kernel_ixs, initial_dim_size)}]"
    
    if ix in pruner.tree_ix_2_list_of_initial_input_slice_ixs.keys():
        list_of_active_initial_input_slice_ixs = pruner.tree_ix_2_list_of_initial_input_slice_ixs[ix]
        initial_dim_size = initial_weights_shape[1]
        display_string += f"\nInput slices pruned: [{string_of_pruned(list_of_active_initial_input_slice_ixs, initial_dim_size)}]"
    
    return display_string


# Only used if dynamically resizing text:
def rescale_text_to_fit(ax, rect, text):
    # Get the bounding box of the rectangle and text
    renderer = ax.figure.canvas.get_renderer()
    rect_extent = rect.get_window_extent(renderer=renderer)
    text_extent = text.get_window_extent(renderer=renderer)

    # Calculate scaling factors for width and height
    scale_width = rect_extent.width / text_extent.width
    scale_height = rect_extent.height / text_extent.height

    # Use the smaller scale to ensure the text fits within both dimensions
    scale = min(scale_width, scale_height)

    # Set the new font size
    current_fontsize = text.get_fontsize()
    text.set_fontsize(current_fontsize * scale)

# Only used if dynamically resizing text:
RECTS = []
TEXTS = []
def on_draw(event):
    ax = event.canvas.figure.axes[0]
    for rect, text in zip(RECTS, TEXTS):
        rescale_text_to_fit(ax, rect, text)


def draw_tree(ix, layer_name, fig, ax, x, y, width, height, max_depth, resource_calc: Union[None, ConvResourceCalc], initial_resource_calc: Union[None, ConvResourceCalc], pruner, lowest_level_modules):
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
    if pruner is not None and initial_resource_calc is not None and ix in pruner.tree_ix_2_list_of_initial_kernel_ixs.keys():

        # print(display_string)
        display_string += get_string_of_pruned(ix, initial_resource_calc, pruner)

        # just take one real_kernel_ix we are sure exists
        real_kernel_ix = 0
        initial_kernel_ix = pruner.tree_ix_2_list_of_initial_kernel_ixs[ix][real_kernel_ix]

        try:
            

            inextricable_following_to_prune = pruner.kernel_connection_fn(ix, real_kernel_ix, pruner.conv_tree_ixs, pruner.lowest_level_modules)        

            following_to_prune = pruner.input_slice_connection_fn(ix, initial_kernel_ix, pruner.conv_tree_ixs, pruner.lowest_level_modules)

            display_string += 3*"\n" + 10*"=" + "\n"


            if len(inextricable_following_to_prune) > 0:
                display_string += 3*"\n"
                display_string += f"\nInextricable pruned connections:"

            for following_ix, _ in inextricable_following_to_prune:
                to_add = get_string_of_pruned(following_ix, initial_resource_calc, pruner)
                display_string += f"\n{following_ix}: {to_add}"
                display_string += "\n"



            
            if len(following_to_prune) > 0:
                display_string += 3*"\n"
                display_string += f"\nInput slice pruned connections:"

            for following_ix, _ in following_to_prune:
                to_add = get_string_of_pruned(following_ix, initial_resource_calc, pruner)
                display_string += f"\n{following_ix}: {to_add}"
                display_string += "\n"
        except:
            pass
        
        # now for all that this connects to:
        



    

    
    rect = patches.Rectangle((x, y), width, height, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    # text = ax.text(x + width/2, y + height/2, display_string, ha='center', va='center')
    text = ax.text(x + width/2, y + 15/16* height, display_string, ha='center', va='top')
    

    # If you want dynamic resizing of the text:
    if False:
        global RECTS, TEXTS
        RECTS.append(rect)
        TEXTS.append(text)

        # Connect the draw event
        fig.canvas.mpl_connect('draw_event', on_draw)


    # Find children of the current index
    children = [key for key in resource_calc.module_tree_ix_2_name if key[0] == ix]
    if children:
        child_width = width / len(children)
        for i, child in enumerate(sort_tree_ixs(children)):
            child_name = resource_calc.module_tree_ix_2_name[child]
            draw_tree(child, child_name, fig, ax, x + i * child_width, y - height, child_width, height, max_depth - 1, resource_calc, initial_resource_calc, pruner, lowest_level_modules)





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
    draw_tree(root_ix, root_name, fig, ax, 0, total_height, width, height, max_depth, resource_calc, initial_resource_calc, pruner, lowest_level_modules)


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show(block=False)

    return fig, ax









