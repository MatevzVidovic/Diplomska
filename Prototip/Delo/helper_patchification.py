


import os
import logging
import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)








from helper_img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs, save_imgs_quick_figs

import numpy as np
import torch

import matplotlib.pyplot as plt
import os.path as osp
# from utils import one_hot2dist

np.random.seed(7)






def unfold_3chan(tensor_img, patch_shape, stride: tuple):

    try:
        # tensor_img_unf = tensor_img.unfold(0, patch_shape[0], stride[0])
        # tensor_img_unf = tensor_img_unf.unfold(1, patch_shape[1], stride[1])

        y_ix = 0
        x_ix = 0

        if y_ix + patch_shape[0] <= tensor_img.size(1) and x_ix + patch_shape[1] <= tensor_img.size(2):
            patches = tensor_img[:, y_ix:y_ix + patch_shape[0], x_ix:x_ix + patch_shape[1]]
            patches = patches.unsqueeze(0)
            left_upper_ixs = [(y_ix, x_ix)]
            y_ix += stride[0]
        else:
            raise ValueError("Patch shape is bigger than the image shape.")
        

        while x_ix + patch_shape[1] <= tensor_img.size(2):
            while y_ix + patch_shape[0] <= tensor_img.size(1):
                patch = tensor_img[:, y_ix:y_ix + patch_shape[0], x_ix:x_ix + patch_shape[1]]
                patch = patch.unsqueeze(0)
                patches = torch.cat([patches, patch], dim=0)
                left_upper_ixs.append((y_ix, x_ix))
                y_ix += stride[0]
            y_ix = 0
            x_ix += stride[1]
        
        return patches, left_upper_ixs
    except Exception as e:
        py_log.log_stack(MY_LOGGER)
        raise e




def patchify(tensor_img, patch_shape, stride: tuple):

    try:
        main_patches, main_lu_ixs = unfold_3chan(tensor_img, patch_shape, stride)




        # right_bottom_patches:

        # right_patches
        # bottom_patches
        # right_bottom_corner

        # with left up ixs for all of them. And then you concat all of that.
        # And at test time you add that to that parts of the accumulator.
        # Possibly you could count how many times each part of the accumulator was added to and devide by that,
        # but I'm not sure we need that, because it's logits anyway.




        # right_patches

        right_slice = tensor_img[:, :, -patch_shape[1]:]

        right_patches, right_lu_ixs = unfold_3chan(right_slice, patch_shape, stride)
        right_lu_ixs = [(i[0], i[1]+tensor_img.size(2)-patch_shape[1]) for i in right_lu_ixs]



        # bottom_patches

        bottom_slice = tensor_img[:, -patch_shape[0]:, :]
        bottom_patches, bottom_lu_ixs = unfold_3chan(bottom_slice, patch_shape, stride)
        bottom_lu_ixs = [(i[0]+tensor_img.size(1)-patch_shape[0], i[1]) for i in bottom_lu_ixs]



        # right_bottom_corner

        right_bottom_corner = tensor_img[:, -patch_shape[0]:, -patch_shape[1]:]
        right_bottom_patches, right_bottom_lu_ixs = unfold_3chan(right_bottom_corner, patch_shape, stride)
        right_bottom_lu_ixs = [(i[0]+tensor_img.size(1)-patch_shape[0], i[1]+tensor_img.size(2)-patch_shape[1]) for i in right_bottom_lu_ixs]
        

        patch_dict = {
            "main_patches" : main_patches,
            "main_lu_ixs" : main_lu_ixs,
            "right_patches" : right_patches,
            "right_lu_ixs" : right_lu_ixs,
            "bottom_patches" : bottom_patches,
            "bottom_lu_ixs" : bottom_lu_ixs,
            "right_bottom_corner" : right_bottom_patches,
            "right_bottom_corner_lu_ixs" : right_bottom_lu_ixs
        }

        py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"])
        return patch_dict
    

    except Exception as e:
        py_log.log_stack(MY_LOGGER)
        raise e



def accumulate_patches(prediction_tensor_shape, patch_shape, patch_dict):

    try:
    


        all_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
        all_ixs = patch_dict["main_lu_ixs"] + patch_dict["right_lu_ixs"] + patch_dict["bottom_lu_ixs"] + patch_dict["right_bottom_corner_lu_ixs"]


        accumulating_tensor = torch.zeros(prediction_tensor_shape)
        num_of_addings = torch.zeros(prediction_tensor_shape)

        # accumulating_tensor = accumulating_tensor.to(device)
        # num_of_addings = num_of_addings.to(device)

        accumulating_tensor = accumulating_tensor.to(all_patches.device)
        num_of_addings = num_of_addings.to(all_patches.device)

        for i, (lu_ix, patch) in enumerate(zip(all_ixs, all_patches)):
            y1, x1 = lu_ix
            y2, x2 = y1 + patch.size(1), x1 + patch.size(2)
            lala = accumulating_tensor[:, y1:y2, x1:x2]
            accumulating_tensor[:, y1:y2, x1:x2] += patch
            num_of_addings[:, y1:y2, x1:x2] += 1


        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["main_lu_ixs"], patch_dict["main_patches"])):

        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch
        
        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["right_lu_ixs"], patch_dict["right_patches"])):
            
        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch

        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["bottom_lu_ixs"], patch_dict["bottom_patches"])):

        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch
        
        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["right_bottom_corner_lu_ixs"], patch_dict["right_bottom_corner"])):
            
        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch


        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"])
        return accumulating_tensor, num_of_addings
    except Exception as e:
        py_log.log_stack(MY_LOGGER)
        raise e



