




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

import random

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from y_helpers.model_sorting import sort_tree_ixs, denest_tuple
from y_framework.conv_resource_calc import ConvResourceCalc


cm = 1/2.54  # centimeters in inches


def string_of_pruned(list_of_initial_ixs, initial_dim_size):
    """
    list_of_initial_ixs: list of indices that are not pruned.
    We have to then invert this in this function, because the pruned ones interest us.

    When trying to display the kernel ixs that are pruned, we want it to be a bit clearer.
    Instead of   [3, 6, 7, 8, 12, 13, 15, 17],
    we want it to be  [3, 6-8, 12-13, 15, 17]
    """


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
            



def break_up_string_of_pruned(string_of_pruned, limit_chars_per_line):
    list_of_elements = string_of_pruned.split(", ")
    
    # remove empty strings
    for ix in range(len(list_of_elements)-1, -1, -1):
        if list_of_elements[ix] == "":
            del list_of_elements[ix]
    
    final_string = ""
    curr_line = ""
    for element in list_of_elements[:-1]:
        
        if len(curr_line) + len(element) + 2 > limit_chars_per_line:
            final_string += curr_line + "\n"
            curr_line = ""

        curr_line += element + ", "


    if len(list_of_elements) > 0:
        curr_line += list_of_elements[-1]
    
    final_string += curr_line

    return final_string




def get_string_of_pruned(tree_ix, initial_resource_calc, pruner, limit_chars_per_line):
    
    ix = tree_ix

    display_string = ""

    # weight dimensions: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
    initial_weights_shape = initial_resource_calc.resource_name_2_resource_dict["weights_dimensions"][ix]
    
    if ix in pruner.tree_ix_2_list_of_initial_kernel_ixs.keys():
        list_of_active_initial_kernel_ixs = pruner.tree_ix_2_list_of_initial_kernel_ixs[ix]
        initial_dim_size = initial_weights_shape[0]
        curr_string_of_pruned = string_of_pruned(list_of_active_initial_kernel_ixs, initial_dim_size)
        broken_up_string = break_up_string_of_pruned(curr_string_of_pruned, limit_chars_per_line)
        broken_up_string = f"\n{broken_up_string}" if broken_up_string != "" else broken_up_string
        display_string += f"\nKernels pruned: [{broken_up_string}]"
    
    if ix in pruner.tree_ix_2_list_of_initial_input_slice_ixs.keys():
        list_of_active_initial_input_slice_ixs = pruner.tree_ix_2_list_of_initial_input_slice_ixs[ix]
        initial_dim_size = initial_weights_shape[1]
        curr_string_of_pruned = string_of_pruned(list_of_active_initial_input_slice_ixs, initial_dim_size)
        broken_up_string = break_up_string_of_pruned(curr_string_of_pruned, limit_chars_per_line)
        broken_up_string = f"\n{broken_up_string}" if broken_up_string != "" else broken_up_string
        display_string += f"\nInput slices pruned: [{broken_up_string}]"
    
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


def draw_rect(tree_ix, ll_y, lu_x, rect_width, rect_height, new_root_id, dict_of_unchanging):
    """
    Draw the rectangle and label it.
    Then draw the children.
    """

    """
    With matplotlib, the origin is at the bottom left corner, so there y = 0.0 and x=0.0.
    
    By default the axis coordinates are from 0.0-1.0. We use these coordinates in draw_tree (use of transform=ax.transAxes).
    That is different from the data coordinates, which i suspect would here be the actual physical width and height? But I don't know.

    When specifying rect or text position, you are specifying the lower left corner of the text or rect.

    

    For ll_y (left upper y) we will start at 1.0, and then go down by rect_height for each level.

    For lu_x: it will be calculated dynamically in draw_tree - we start with 0.0, and then divide the width by 
    the number of children of the current node to know the x of the children.

    The initial rect_width is 1.0, because the first level spans the whole width, because it is just onw module - the name of the network architecture.

    rect_height is always the same: 1/max_depth, 
    because we know how many levels there are, so we can precompute it.
    """
    is_for_img = dict_of_unchanging["is_for_img"]
    limit_chars_in_line = dict_of_unchanging["limit_chars_in_line"]
    physical_width = dict_of_unchanging["physical_width"]
    physical_height = dict_of_unchanging["physical_height"]
    
    fig = dict_of_unchanging["fig"]
    ax = dict_of_unchanging["ax"]
    resource_calc : Union[None, ConvResourceCalc] = dict_of_unchanging["resource_calc"]
    initial_resource_calc : Union[None, ConvResourceCalc] = dict_of_unchanging["initial_resource_calc"]
    pruner = dict_of_unchanging["pruner"]
    lowest_level_modules = dict_of_unchanging["lowest_level_modules"]















    # -------------------- GETTING THE TEXT FOR DISPLAY --------------------


    layer_name = resource_calc.module_tree_ix_2_name[tree_ix]

    this_name_tree_ixs = resource_calc.get_ordered_list_of_tree_ixs_for_layer_name(layer_name)
    # which of its' name is it?
    ordered_ix = this_name_tree_ixs.index(tree_ix)


    display_string = ""
    if new_root_id is not None:
        display_string += f"new root: {new_root_id}: \n"

    display_string += f'{ordered_ix}. {layer_name}\n{tree_ix}'


    
            
    if tree_ix in lowest_level_modules:
        lowest_level_modules_index = lowest_level_modules.index(tree_ix)
        display_string += f"\n{lowest_level_modules_index}. LLM"
    


    layer = resource_calc.module_tree_ix_2_module_itself[tree_ix]
    if type(layer) == nn.Conv2d:
        display_string += f"\nW.shape: {list(layer.weight.shape)}"
    
    elif type(layer) == nn.BatchNorm2d:
        
        shapes = [list(layer.weight.shape), list(layer.bias.shape), 
                    list(layer.running_mean.shape), list(layer.running_var.shape)]
        
        # are all shapes the same?
        if all(shape == shapes[0] for shape in shapes):
            display_string += f"\nW.shape: {shapes[0]}"
        else:
            for shape in shapes:
                display_string += f"\nW,B,RM,RV shapes: {shape}"
    

    # The display of what we have pruned:
    if pruner is not None and initial_resource_calc is not None and tree_ix in pruner.tree_ix_2_list_of_initial_kernel_ixs.keys():

        # print(display_string)
        display_string += get_string_of_pruned(tree_ix, initial_resource_calc, pruner, limit_chars_in_line)

        # just take one real_kernel_ix we are sure exists
        real_kernel_ix = 0
        initial_kernel_ix = pruner.tree_ix_2_list_of_initial_kernel_ixs[tree_ix][real_kernel_ix]

        try:
            

            inextricable_following_to_prune = pruner.kernel_connection_fn(tree_ix, real_kernel_ix, pruner.conv_tree_ixs, pruner.lowest_level_modules)        

            following_to_prune = pruner.input_slice_connection_fn(tree_ix, initial_kernel_ix, pruner.conv_tree_ixs, pruner.lowest_level_modules)

            display_string += 3*"\n" + 10*"=" + "\n"


            if len(inextricable_following_to_prune) > 0:
                display_string += 3*"\n"
                display_string += f"\nInextricable pruned connections:"

                for following_tree_ix, _ in inextricable_following_to_prune:
                    to_add = get_string_of_pruned(following_tree_ix, initial_resource_calc, pruner, limit_chars_in_line)
                    lowest_level_modules_index = lowest_level_modules.index(following_tree_ix)
                    display_string += f"\n{lowest_level_modules_index}. LLM: {to_add}"
                    display_string += "\n" + 10*"-" + "\n"
                
                display_string += 10*"=" + "\n"
                



            
            if len(following_to_prune) > 0:
                display_string += 3*"\n"
                display_string += f"\nInput slice pruned connections:"

                for following_tree_ix, _ in following_to_prune:
                    to_add = get_string_of_pruned(following_tree_ix, initial_resource_calc, pruner, limit_chars_in_line)
                    lowest_level_modules_index = lowest_level_modules.index(following_tree_ix)
                    display_string += f"\n{lowest_level_modules_index}. LLM: {to_add}"
                    display_string += "\n" + 10*"-" + "\n"
                
                display_string += 10*"=" + "\n"

        except:
            pass
        
        # now for all that this connects to:
        










    # -------------------- ACTUAL DISPLAY PART --------------------


    def break_lines(text, width):
        lines = text.split('\n')
        final_lines = []
        for line in lines:
            broken_line = [line[i:i+width] for i in range(0, len(line), width)]
            final_lines.extend(broken_line)
        
        final_str = '\n'.join(final_lines)

        return final_str

    wrapped_text = break_lines(display_string, limit_chars_in_line)

    if is_for_img:

        def calculate_font_size(chars_in_line, num_lines, horiz_multiplier, vert_multiplier, rect_width, rect_height):
            """
            fontsize=8 means 8/72 of an inch is the char_height of characters.
            We do not know what the char_width is - we will have to estimate it as 0.6*height.
            We choose to use a monospace font, where the width of each char is the same (otherwise w is wider than i).

            
            rect_width and char_width are in different units. We have no idea of the conversion rate.
            1_rect_width = k2 * 1_char_width


            rect_width = chars_in_line * (k2 * char_width)
            rect_width = chars_in_line * (k2 * (0.6 * fontsize/72) )
            fontsize = (72 / (0.6 * k2)) * (rect_width / chars_in_line)

            
            But also, we would like the text to fit in the height of the rect.
            And we assume the unit conversion is the same, so:    1_rect_height = k2 * 1_char_height
            rect_height = num_lines * (k2 * char_height)
            rect_height = num_lines * (k2 * (fontsize/72))
            fontsize = (72 / k2) * (rect_height / num_lines)

            We will then take the smaller one of the two font sizes.

            We will determine k2 experimentally.
            More precisely, we will determine the 
            multiplier := 1/k2.


            Turns out, there's weird stuff happening - we cant get this right.
            So we will introduce vert_multiplier, to try to make it work.

            Do it like this:
            First calibrate the width. Make height not matter, and make width mult so high that the chars fit exacltly in the width.
            Then go and calibrate the height so that these things start squeezing and find the mult that makes it fit exactly in the height.
            # And boom, you've won.



            Wheat we expect it to be:
            The figsize is 50*cm x 50cm*cm  in inches (*cm is the converter)
            ax_size is 1.0x1.0
            so 1.0ax = 50/2.54 inches = 19.685 inches
            So this should be the k2.
            So since mult is 1/k2, we expect it to be 1/19.685 = 0.0507
            


            """

            # fontsize_w = multiplier * (72 / 0.6) * (rect_width / chars_in_line)

            fontsize_w = horiz_multiplier * (72 / 0.6) * (rect_width / chars_in_line)


            fontsize_h = vert_multiplier * (72) * (rect_height / num_lines)

            # print(f"fontsize_w: {fontsize_w}")
            # print(f"fontsize_h: {fontsize_h}")

            fontsize = min(fontsize_w, fontsize_h)

            # for now lets care only about width
            # fontsize = fontsize_w

            return fontsize

              
        

        max_chars_in_line = max(len(line) for line in wrapped_text.split('\n'))
        num_lines = len(wrapped_text.split('\n'))

        adjusted_width = rect_width * 14/16 # I want to leave 1/16 on each side
        fontsize = calculate_font_size(max_chars_in_line, num_lines, physical_width * 15.5/50, physical_height *13/50, adjusted_width, rect_height)
        

        if fontsize < 1:
            print(f"WARNING! Fontsize less than 1. Plt will mess this up. fontsize: {fontsize}")
        # print(f"wrapped_text: {wrapped_text}")
        # print(f"num_lines: {num_lines}")
        # print(f"max_chars_in_line: {max_chars_in_line}")
        # print(f"fontsize: {fontsize}")
        # print(f"rect_width: {rect_width}")
        # print(f"rect_height: {rect_height}")
        # print(f"\n\n")
        






        linewidth_k = 0.01

        rect = patches.Rectangle((lu_x, ll_y), rect_width, rect_height, 
                                transform=ax.transAxes, linewidth=linewidth_k*physical_width,
                                edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        # text = ax.text(x + width/2, y + height/2, display_string, ha='center', va='center')
        # text = ax.text(x + width/2, y + 15/16* height, display_string, ha='center', va='top')


        text = ax.text(lu_x + rect_width/2, ll_y + 15/16* rect_height, wrapped_text, 
                    ha='center', va='top',
                    fontsize=fontsize, fontfamily='monospace')


    else:


        linewidth_k = 0.01

        rect = patches.Rectangle((lu_x, ll_y), rect_width, rect_height, 
                                linewidth=linewidth_k*physical_width,
                                edgecolor='black', facecolor='none')
        ax.add_patch(rect)

        # text = ax.text(x + width/2, y + height/2, display_string, ha='center', va='center')
        # text = ax.text(x + width/2, y + 15/16* height, display_string, ha='center', va='top')

        fontsize_k = 50/100
        fontsize = physical_width * fontsize_k    # adjusted with physical width, so it looks nice in every context. 
        # Make the zeroth or first row nicely readable and that's that.
        text = ax.text(lu_x + rect_width/2, ll_y + 15/16* rect_height, wrapped_text, 
                    ha='center', va='top',
                    fontsize=fontsize, fontfamily='monospace')

    # text = ax.text(x + width/2, y + height/2, wrapped_text,
    #               ha='center', va='center',
    #               fontsize=8)




    # def calculate_font_size(text_str, width, height):
    #     # Rough estimation - adjust multiplier as needed
    #     area = width * height
    #     text_length = len(text_str)
    #     import numpy as np
    #     return min(8, np.sqrt(area/(text_length + 1)) * 200)

    # fontsize = calculate_font_size(display_string, width, height)
    # text = ax.text(x + width/2, y + height/2, display_string, 
    #               ha='center', va='center',
    #               fontsize=fontsize)




    # If you want dynamic resizing of the text:
    if False:
        global RECTS, TEXTS
        RECTS.append(rect)
        TEXTS.append(text)

        # Connect the draw event
        fig.canvas.mpl_connect('draw_event', on_draw)




def get_dict_of_rectangle_info(curr_tree_ix, resource_calc, curr_level, lu_x, rect_width, min_child_width_limit):
    """
    We used to calculate left_lower_y and rect_height too, and pass them as params.
    But now we would rather just know what level this rect is on: the top row being 0, the next row down being 1, etc.
    Previously we would calculate the rect_height as 1/max_depth, where max_depth would be true if the graph would never be broken due to min_child_width_limit.
    By just holding the level, we can calculate max_depth of this specific dict_of_rectangle_info we are about to draw, 
    and this way we get a drawing across the entire plot.
    """

    res_dict = {
        curr_tree_ix: {
            "curr_level": curr_level,
            "lu_x": lu_x,
            "rect_width": rect_width,
            "new_root_id": None
        }
    }








    # Find children of the current index
    children = resource_calc.module_tree_ix_2_children_tree_ix_list[curr_tree_ix]

    if len(children) == 0:
        return res_dict, []
    
    else:

        child_width = rect_width / len(children)
        
        # the if rect_width > 0.99 is important to prevent infinite recursion
        if rect_width > 0.99 or child_width >= min_child_width_limit:
            # Everything is normal

            # The normal path
            children_dict = {}
            children_root_res_dicts = []

            for i, child_tree_ix in enumerate(sort_tree_ixs(children)):
                child_dict, list_of_new_root_dicts = get_dict_of_rectangle_info(child_tree_ix, resource_calc, curr_level=(curr_level+1), lu_x = (lu_x + i * child_width), rect_width=child_width, 
                                                                                min_child_width_limit=min_child_width_limit)
                children_dict.update(child_dict)
                children_root_res_dicts += list_of_new_root_dicts

            res_dict.update(children_dict)

            return res_dict, children_root_res_dicts
        




        else: # child_width < min_child_width_limit:
                
            
            # This is for the old dict_of_rectangle_info, where we have to break the tree.
            # This lets us know something became a new root.
            # The parent dict ends at this node tho.
            curr_id = random.randint(0, 999999)
            res_dict[curr_tree_ix]["new_root_id"] = curr_id
            
            


            curr_level = 0
            rect_width = 1.0
            lu_x = 0.0

            # This call is essentially the same as when we first call get_dict_of_rectangle_info from somewhere outside the function.
            # Now we just make it that the root is lower down in the tree.
            new_root_res_dict, children_root_res_dicts = get_dict_of_rectangle_info(curr_tree_ix, resource_calc, curr_level=curr_level, lu_x=lu_x, rect_width=rect_width, 
                                                                                    min_child_width_limit=min_child_width_limit)



            complete_list = [(new_root_res_dict, curr_id)] + children_root_res_dicts

            return res_dict, complete_list



def add_rect_height_and_ll_y_to_dict_of_rectangle_info(dict_of_rectangle_info):

    max_level = max(rect_info["curr_level"] for rect_info in dict_of_rectangle_info.values())
    rect_height = 1/(max_level + 1)  # +1 because levels are 0-indexed, so their num is one more than the highest ix.

    for rect_info in dict_of_rectangle_info.values():
        curr_level = rect_info["curr_level"]
        rect_info["rect_height"] = rect_height
        rect_info["ll_y"] = 1.0 - (curr_level + 1) * rect_height

    return dict_of_rectangle_info












def model_graph(resource_calc, initial_resource_calc=None, pruner=None, width=20, height=None, for_img_dict=None):
    """
    Width and height are the physical size of the initial window in centimeters.

    If you pass for img dict, this signals we are in for_img mode. Even if the dict is empty.

    """

    if height is None:
        height = width



    is_for_img = not for_img_dict is None

    min_child_width_limit = for_img_dict.get("min_child_width_limit", 0.00) if is_for_img else 0.00

    





    lowest_level_modules = []
    for tree_ix, children_list in resource_calc.module_tree_ix_2_children_tree_ix_list.items():
        if len(children_list) == 0:
            lowest_level_modules.append(tree_ix)
    

    
    # We precompute the rect_height, because we know ho many levels there are.
    max_depth = max(len(denest_tuple(tree_ix)) for tree_ix in resource_calc.module_tree_ixs)
    rect_height = 1/max_depth
    ll_y = 1.0 - rect_height

    root_tree_ix = (0,) # this is always the case in the indexing system of ConvResourceCalc


    """
    With matplotlib, the origin is at the bottom left corner, so there y = 0.0 and x=0.0.
    
    By default the axis coordinates are from 0.0-1.0. We use these coordinates in draw_tree (use of transform=ax.transAxes).
    That is different from the data coordinates, which i suspect would here be the actual physical width and height? But I don't know.

    When specifying rect or text position, you are specifying the lower left corner of the text or rect.



    A HUGE PROBLEM IS THAT MINIMAL FONTSIZE IS 1.
    WHA. IS. THIS. FACT. SO. HARD. TO. FIND!!!!
    So since each lower layer has a smaller and smaller rect width, the text is getting exponentially smaller.
    And eventually we hit the limit of 1, and the text starts spilling over into neighbouting rects.

    One solution is to make width and height hude, so we don't crash into that limit. But then for some reason we quickly hit RAM problems.
    Another solution would be that under a certain width, we start to make new figures instead of displaying them on the same one.

    

    For ll_y (left upper y) we will start at 1.0, and then go down by rect_height for each level.

    For lu_x: it will be calculated dynamically in draw_tree - we start with 0.0, and then divide the width by 
    the number of children of the current node to know the x of the children.

    The initial rect_width is 1.0, because the first level spans the whole width, because it is just onw module - the name of the network architecture.

    The initial rect_height is 1/max_depth, because we know how many levels there are, so we can precompute it.
    """



    dict_of_rectangle_info, list_of_new_root_rect_dicts = get_dict_of_rectangle_info(root_tree_ix, resource_calc, curr_level=0, lu_x=0.0, rect_width=1.0, min_child_width_limit=min_child_width_limit)

    main_dict_tuple = (dict_of_rectangle_info, "Main")
    all_dicts_to_draw = [main_dict_tuple] + list_of_new_root_rect_dicts

    all_dicts_to_draw = [(add_rect_height_and_ll_y_to_dict_of_rectangle_info(curr_res_dict), curr_id) for curr_res_dict, curr_id in all_dicts_to_draw]



    all_fig_ax_id_tuples = []
    for curr_res_dict, curr_id in all_dicts_to_draw:

        fig, ax = plt.subplots(figsize=(width*cm, height*cm))
        # set title to fig
        fig.suptitle(curr_id, fontsize=16)

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.axis('off')

        dict_of_unchanging = {
            "is_for_img": is_for_img,
            "limit_chars_in_line": 35,
            "physical_width": width,
            "physical_height": height,

            "fig": fig,
            "ax": ax,
            "resource_calc": resource_calc,
            "initial_resource_calc": initial_resource_calc,
            "pruner": pruner,
            "lowest_level_modules": lowest_level_modules
        }

        for tree_ix, rect_info in curr_res_dict.items():
            ll_y = rect_info["ll_y"]
            lu_x = rect_info["lu_x"]
            rect_height = rect_info["rect_height"]
            rect_width = rect_info["rect_width"]
            new_root_id = rect_info["new_root_id"]
            draw_rect(tree_ix, ll_y=ll_y, lu_x=lu_x, rect_width=rect_width, rect_height=rect_height, new_root_id=new_root_id, dict_of_unchanging=dict_of_unchanging)

        all_fig_ax_id_tuples.append((fig, ax, curr_id))




        # plt.show(block=False)





    return all_fig_ax_id_tuples









