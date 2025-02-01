
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




import numpy as np
import torch

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
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


def get_random_patches(tensor_imgs_list, patch_shape, num_of_patches_from_img, prob_zero_patch_resample=None, resample_gt=None):
    """
    tensor_imgs_list is e.g. [image, veins, sclera]
    It is a list of same sized 4d tensors, that are describing the same object.
    These tensors are [1, ch, h, w] shaped.
    We want to return a list of patches that are shaped:
    [num_of_patches_from_img, ch, patch_shape[0], patch_shape[1]]
    The first tensor in the returner corresponds to the first tensor in tensor_imgs_list.

    If prob_zero_patch_resample is not None, then when we chose a point, we look at that point on resample_gt (ground truth).
    If that point is 0, we resample with probability prob_zero_patch_resample.
    In our case resample_gt is sclera. We do this, so that our patches won't mostly be of the background.
    """

    try:
        returning_list = [None for _ in tensor_imgs_list]

        tensor_img = tensor_imgs_list[0]
        lu_ixs = [] # left upper ixs

        while len(lu_ixs) < num_of_patches_from_img:
            y_ix = np.random.randint(0, tensor_img.size(2) - patch_shape[0])
            x_ix = np.random.randint(0, tensor_img.size(3) - patch_shape[1])

            if prob_zero_patch_resample is not None:
                mid_y_ix = y_ix + patch_shape[0] // 2
                mid_x_ix = x_ix + patch_shape[1] // 2
                if resample_gt[0, 0, mid_y_ix, mid_x_ix] == 0:
                    if np.random.rand() < prob_zero_patch_resample:
                        continue

            lu_ixs.append((y_ix, x_ix))
        
        for ix, tensor_img in enumerate(tensor_imgs_list):
            for y_ix, x_ix in lu_ixs:
                patch = tensor_img[:, :, y_ix:y_ix+patch_shape[0], x_ix:x_ix+patch_shape[1]]
                if returning_list[ix] is None:
                    returning_list[ix] = patch
                else:
                    returning_list[ix] = torch.cat([returning_list[ix], patch], dim=0)


        return returning_list


    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
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
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e



def accumulate_patches(prediction_tensor_shape, patch_dict):

    try:
    


        all_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
        all_ixs = patch_dict["main_lu_ixs"] + patch_dict["right_lu_ixs"] + patch_dict["bottom_lu_ixs"] + patch_dict["right_bottom_corner_lu_ixs"]


        accumulating_tensor = torch.zeros(prediction_tensor_shape)
        num_of_addings = torch.zeros(prediction_tensor_shape)

        # accumulating_tensor = accumulating_tensor.to(device)
        # num_of_addings = num_of_addings.to(device)

        accumulating_tensor = accumulating_tensor.to(all_patches.device)
        num_of_addings = num_of_addings.to(all_patches.device)

        for (lu_ix, patch) in zip(all_ixs, all_patches):
            y1, x1 = lu_ix
            y2, x2 = y1 + patch.size(1), x1 + patch.size(2)
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
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e



