





import os
import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on



MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)





import numpy as np
import torch
import os
# import random
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import os.path as osp
# from utils import one_hot2dist


import inspect





SHOW_IMAGE_IX = [0]
def show_image(passed_img, title="", close_all_limit=1e9):

    try:
        # passed_img can be np.ndarray, Image.Image, or torch.Tensor
        # If passed_img is a list, it will show all images in the list on one plot.
        # passed_img as a list can also have tuples with titles: e.g. [(img1, title1), (img2, title2]

        # Close all open figures
        figure_numbers = plt.get_fignums()
        if len(figure_numbers) >= close_all_limit:
            plt.close('all')


        if not isinstance(passed_img, list):
            passed_img = [passed_img]


        imgs = passed_img

        # determine rown and columns:
        if len(imgs) == 1:
            rc = (1,1)
        elif len(imgs) == 2:
            rc = (1,2)
        elif len(imgs) <= 4:
            rc = (2,2)
        elif len(imgs) <= 6:
            rc = (2,3)
        elif len(imgs) <= 9:
            rc = (3,3)
        else:
            cols = len(imgs) // 3 + 1
            rc = (3, cols)
        
        fig, axes = plt.subplots(rc[0], rc[1])

        # when rc = (1,1), axes is not a np.array of many axes, but a single Axes object. And then flatten doesn't work, and iteration doesn't work.
        # It's just easier to make it into a np.array.
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        # Iterate over images and axesk
        for i, ax in enumerate(axes):
            if i < len(imgs):

                curr_img = imgs[i][0] if isinstance(imgs[i], tuple) else imgs[i]
                curr_title = imgs[i][1] if isinstance(imgs[i], tuple) else title


                try:
                    # this clones the image anyway
                    img = smart_conversion(curr_img, 'ndarray', 'uint8')
                except Exception as e:
                    py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])
                    raise e

                ax.imshow(img, cmap='gray')
                ax.set_title(f"{curr_title} ({SHOW_IMAGE_IX[0]})")
                ax.axis('off')
                SHOW_IMAGE_IX[0] += 1
        
        # set main title to the line where this function was called from
        caller_frame = inspect.currentframe().f_back
        caller_line = caller_frame.f_lineno
        caller_func = caller_frame.f_code.co_name
        caller_file = caller_frame.f_code.co_filename
        caller_file = osp.basename(caller_file)
        fig.suptitle(f"Called from line {caller_line} in {caller_func} in {caller_file}")

        initial_fig_name = plt.get_current_fig_manager().get_window_title()
        plt.get_current_fig_manager().set_window_title(f"{initial_fig_name}, line {caller_line} in {caller_func} in {caller_file}")

        plt.show(block=False)
        plt.pause(0.001)

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e







# Block of fns for conversions of image types and representations:


def to_type(given_img, goal_type_name):
    # type name can be 'ndarray', 'tensor', 'Image'

    if isinstance(given_img, np.ndarray):

        img = given_img.copy()

        if goal_type_name == 'ndarray':
            return img
        elif goal_type_name == 'tensor':
            img = torch.from_numpy(img)
            if len(img.shape) == 3:
                img = img.permute(2, 0, 1)
            return img
        elif goal_type_name == 'Image':
            return Image.fromarray(img)
        
    elif isinstance(given_img, Image.Image):

        img = given_img.copy()

        if goal_type_name == 'ndarray':
            return np.array(img)
        elif goal_type_name == 'tensor':
            return transforms.ToTensor()(img)
        elif goal_type_name == 'Image':
            return img
        
    elif isinstance(given_img, torch.Tensor):

        img = given_img.clone()

        if goal_type_name == 'ndarray':
            if len(img.shape) == 3:
                img = img.permute(1, 2, 0)
            return img.numpy()
        elif goal_type_name == 'tensor':
            return img
        elif goal_type_name == 'Image':
            return transforms.ToPILImage()(img)
        

    raise ValueError("goal_type_name must be 'ndarray', 'tensor', or 'Image', and img must be np.ndarray, Image.Image, or torch.Tensor")

def to_img_repr(given_img, goal_img_repr):
    # The two img reprs are [0, 255] (uint8) and [0, 1] (float32)
    # goal_img_repr can be "uint8", "float32"

    if isinstance(given_img, np.ndarray):

        img = given_img.copy()

        if img.dtype == np.uint8:
            if goal_img_repr == "uint8":
                return img
            elif goal_img_repr == "float32":
                img = img.astype(np.float32)
                img /= 255
                return img
            
        elif img.dtype == np.float32:
            if goal_img_repr == "uint8":
                img *= 255
                img = img.astype(np.uint8)
                return img
            elif goal_img_repr == "float32":
                return img
    
    elif isinstance(given_img, Image.Image):

        img = given_img.copy()

        if goal_img_repr == "uint8":
            return given_img
        
        raise ValueError("Image.Image can only be uint8")
    
    elif isinstance(given_img, torch.Tensor):
            
            img = given_img.clone()
    
            if img.dtype == torch.uint8:
                if goal_img_repr == "uint8":
                    return img
                elif goal_img_repr == "float32":
                    img = img.float()
                    img /= 255
                    return img
                
            elif img.dtype == torch.float32:
                if goal_img_repr == "uint8":
                    img *= 255
                    img = img.byte()
                    return img
                elif goal_img_repr == "float32":
                    return img

    raise ValueError("goal_img_repr must be 'uint8' or 'float32', and img must be np.ndarray, Image.Image, or torch.Tensor")

def to_type_and_then_img_repr(img, goal_type_name, goal_img_repr):
    return to_img_repr(to_type(img, goal_type_name), goal_img_repr)

def to_img_repr_and_then_type(img, goal_type_name, goal_img_repr):
    return to_type(to_img_repr(img, goal_img_repr), goal_type_name)

def smart_conversion(img, goal_type_name, goal_img_repr):

    # If start_type and goal_type are both in ['ndarray', 'tensor'], then the order of to_type and to_img_repr doesn't matter anyway.
    
    # When converting to uint8, it's best to first go to uint8 and then to the type.
    # If start_type is in ['Image'], the order doesn't matter.
    # If the start_img_repr is uint8, the order doesn't matter.
    # But if the goal type is in ['Image'], and the start_img_repr is float32, then it's only possible to go to uint8 first and then to Image.
    # Because Image doesn't have float32 representation.

    # When converting to float32, it's best to first go to type and then to float32.
    # Same reason. Image doesn't have float32 representation.
    # So if start_type is "Image", we have to first convert the type and then the img_repr.
    # And if the goal type is "Image", this will always fail anyway, because it's impossible to do.

    if isinstance(img, torch.Tensor) and img.dtype == torch.int64:
        img = img.clone()
        img = img.byte()


    if goal_img_repr == 'uint8':
        return to_img_repr_and_then_type(img, goal_type_name, goal_img_repr)
    elif goal_img_repr == 'float32':
        return to_type_and_then_img_repr(img, goal_type_name, goal_img_repr)