



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


python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024, var_blacklist=["tree_ix_2_module", "mask_path"])
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path)



from img_augments import horizontal_flip, rotation, gaussian_blur


from y_helpers.helper_img_and_fig_tools import smart_conversion, save_plt_fig_quick_figs

from enum import Enum

import numpy as np
import torch
from torch.utils.data import Dataset
import os
# import random
from PIL import Image
from torchvision import transforms
import cv2

import matplotlib.pyplot as plt
import os.path as osp
# from utils import one_hot2dist

np.random.seed(7)

import matplotlib.pyplot as plt



transform = transforms.Compose(
    [
     transforms.Normalize([0.5], [0.5])
    ])







def mask_exists(mask_folder_path, file_name_no_suffix):
    mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
    return osp.exists(mask_filename)

def get_mask(mask_folder_path, file_name_no_suffix) -> np.array:

    mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')

    mask = Image.open(mask_filename).convert("RGB")
    return mask















import argparse

if __name__ == "__main__":


    # Suggested plan:
    
    # This is mostly meant for test and val splits. In this way we can then go over all of the parts of the image and get how good the model was.
    # In this way our test and val scores do not have overlapping pixels, which would skew the result.

    # This does no resizing. Simply ignores the right and bottom parts of the image that don't fit the tiling.
    

    # cores probs help because of C vectorization in numpy operations
    # srun -c 30 --gpus=A100  python3 v_partial_preaug_data_creator.py --ow 1500 --oh 1000

    parser = argparse.ArgumentParser(description='Data Augmentation for Vein Sclera Segmentation')


    parser.add_argument('--fp', type=str, default='./Data/vein_and_sclera_data', help='Folder path of the dataset')
    parser.add_argument('--split', type=str, default='all', help='Split of the dataset. Can be train, val, or test, or all')

    args = parser.parse_args()
    
    folderpath = args.fp
    parsed_split = args.split
    

    try:

        y_dims_imgs = []
        x_dims_imgs = []
        pixels_imgs = []
        y_dims_masks = []
        x_dims_masks = []
        pixels_masks = []
        



        splits = []
        if parsed_split == 'all':
            splits = ['train', 'val', 'test']
        else:
            splits = [parsed_split]
        


        for split in splits:

                
            images_without_mask = []
            images_with_masks = []
            
            imgs_path = osp.join(folderpath, split, 'Images')
            masks_path = osp.join(folderpath, split, 'Veins')
            all_images = os.listdir(imgs_path)
            all_images.sort()



            for file in all_images:
                
                file_without_suffix = file.strip(".jpg")

                if mask_exists(masks_path, file_without_suffix):
                    images_with_masks.append(file_without_suffix)
                else:
                    images_without_mask.append(file_without_suffix)


            #summary
            print('summary for ' + str(split))
            print('valid images: ' + str(len(images_with_masks)))




            for ix, img_name in enumerate(images_with_masks):


            

                image_path = osp.join(imgs_path, f"{img_name}.jpg")

                img = Image.open(image_path).convert("RGB")



                file_name_no_suffix = img_name
                mask = get_mask(masks_path, file_name_no_suffix) # is of type Image.Image


                # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
                img = smart_conversion(img, 'ndarray', 'uint8')
                mask = smart_conversion(mask, 'ndarray', 'uint8')

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

                y_dim_img, x_dim_img, _ = img.shape
                y_dims_imgs.append(y_dim_img)
                x_dims_imgs.append(x_dim_img)
                pixels_imgs.append(y_dim_img * x_dim_img)

                y_dim_mask, x_dim_mask, _ = mask.shape
                y_dims_masks.append(y_dim_mask)
                x_dims_masks.append(x_dim_mask)
                pixels_masks.append(y_dim_mask * x_dim_mask)
        











        
        infos = [(y_dims_imgs, 'y_dims_imgs'), (x_dims_imgs, 'x_dims_imgs'), (pixels_imgs, 'pixels_imgs'), (y_dims_masks, 'y_dims_masks'), (x_dims_masks, 'x_dims_masks'), (pixels_masks, 'pixels_masks')]

        for info in infos:
            print(info[1])
            print('mean: ' + str(np.mean(info[0])))
            print('std: ' + str(np.std(info[0])))
            print('max: ' + str(np.max(info[0])))
            print('min: ' + str(np.min(info[0])))

            # make histograms
            fig, ax = plt.subplots()
            ax.hist(info[0], bins=100)
            ax.set_title(info[1])
            save_plt_fig_quick_figs(fig, info[1] + "_" + parsed_split, formats=["svg"])

            # make box plots
            fig, ax = plt.subplots()
            ax.boxplot(info[0])

            # Calculate statistics
            median = np.median(info[0])
            q1 = np.percentile(info[0], 25)
            q3 = np.percentile(info[0], 75)
            min_val = np.min(info[0])
            max_val = np.max(info[0])
            
            
            # Define the offset as a fraction of the axes width
            offset_fraction = 0.0
            
            # Get the current axes limits
            xlim = ax.get_xlim()
            x_offset = (xlim[1] - xlim[0]) * offset_fraction

            text_offset = 0
            
            # Annotate the boxplot
            ax.annotate(f'Median: {median:.2f}', xy=(1 + x_offset, median), xycoords=('data', 'data'),
                        xytext=(text_offset, 0), textcoords='offset points', ha='left', va='center')
            ax.annotate(f'Q1: {q1:.2f}', xy=(1 + x_offset, q1), xycoords=('data', 'data'),
                        xytext=(text_offset, 0), textcoords='offset points', ha='left', va='center')
            ax.annotate(f'Q3: {q3:.2f}', xy=(1 + x_offset, q3), xycoords=('data', 'data'),
                        xytext=(text_offset, 0), textcoords='offset points', ha='left', va='center')
            ax.annotate(f'Min: {min_val:.2f}', xy=(1 + x_offset, min_val), xycoords=('data', 'data'),
                        xytext=(text_offset, 0), textcoords='offset points', ha='left', va='center')
            ax.annotate(f'Max: {max_val:.2f}', xy=(1 + x_offset, max_val), xycoords=('data', 'data'),
                        xytext=(text_offset, 0), textcoords='offset points', ha='left', va='center')
    
            ax.set_title(info[1])
            save_plt_fig_quick_figs(fig, "box_" + info[1] + "_" + parsed_split, formats=["svg"])


        


        
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e



