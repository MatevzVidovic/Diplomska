


import os.path as osp
import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)



from img_augments import horizontal_flip, rotation, gaussian_blur


from helper_img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs, save_img

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



















def get_tiling(img, mask, patch_width, patch_height):
    patches = []

    x_start = 0
    y_start = 0
    x_next = patch_width
    y_next = patch_height
    while x_next <= img.shape[1]:
        while y_next <= img.shape[0]:
            img_patch = img[y_start: y_next, x_start: x_next, :]
            mask_patch = mask[y_start: y_next, x_start: x_next, :]

            patches.append((img_patch, mask_patch))

            y_start += patch_height
            y_next += patch_height

        x_start += patch_width
        x_next += patch_width
        y_start = 0
        y_next = patch_height
    
    return patches



def get_random_patches(img, mask, patch_width, patch_height, num_of_patches):

    patches = []

    for i in range(num_of_patches):
        x = np.random.randint(0, img.shape[0] - patch_width)
        y = np.random.randint(0, img.shape[1] - patch_height)

        img_patch = img[x: x + patch_width, y: y + patch_height, :]
        mask_patch = mask[x: x + patch_width, y: y + patch_height, :]

        patches.append((img_patch, mask_patch))
    
    return patches



import argparse

if __name__ == "__main__":


    # Suggested plan:
    
    # This is mostly meant for test and val splits. In this way we can then go over all of the parts of the image and get how good the model was.
    # In this way our test and val scores do not have overlapping pixels, which would skew the result.

    # This does no resizing. Simply ignores the right and bottom parts of the image that don't fit the tiling.
    

    # cores probs help because of C vectorization in numpy operations
    # srun -c 30 --gpus=A100  python3 v_partial_preaug_data_creator.py --ow 1500 --oh 1000

    parser = argparse.ArgumentParser(description='Data Augmentation for Vein Sclera Segmentation')

    parser.add_argument('--ph', type=int, default=128, help='Patch height of the image')
    parser.add_argument('--pw', type=int, default=128, help='Patch width of the image')
    parser.add_argument('--pn', type=int, default=50, help="Num of patches for the random mode.")

    parser.add_argument('--fp', type=str, default='./vein_sclera_data', help='Folder path of the dataset')
    parser.add_argument('--dafp', type=str, default='./vein_sclera_data_patchified', help='Augmented data folderpath. To this name the ow and oh get appended.')
    parser.add_argument('--split', type=str, default='test', help='Split of the dataset')
    parser.add_argument('--mode', type=str, default='tiling', help='Modes tiling, random, both. Tiling is used for test and val. Both is generally used for train.')

    args = parser.parse_args()
    
    folderpath = args.fp
    da_folderpath = args.dafp
    split = args.split
    mode = args.mode
    
    patch_height = args.ph
    patch_width = args.pw
    patch_num = args.pn
    
    da_folderpath = f"{da_folderpath}_{patch_height}x{patch_width}"

    
        
    images_without_mask = []
    images_with_masks = []
    
    imgs_path = osp.join(folderpath, split, 'Images')
    masks_path = osp.join(folderpath, split, 'Masks')
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


    try:
        for ix, img_name in enumerate(images_with_masks):


        

            image_path = osp.join(imgs_path, f"{img_name}.jpg")

            img = Image.open(image_path).convert("RGB")



            # Since our data augmentation should simulate errors and changes in how the image is taken, we should do gamma correction and clahe after the augmentation.
            # Because we should first simulate the errors, that is then what our taken img would have been,
            # and then we do our preprocessing (correction) from there.
            # We pretend that the data augmentations were actually pictures that we took.




            file_name_no_suffix = img_name
            mask = get_mask(masks_path, file_name_no_suffix) # is of type Image.Image


            # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')
            mask = smart_conversion(mask, 'ndarray', 'uint8')

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])


            if mode == 'tiling':
                patches = get_tiling(img, mask, patch_width, patch_height)
            elif mode == 'random':
                patches = get_random_patches(img, mask, patch_width, patch_height, patch_num)
            elif mode == 'both':
                patches = get_tiling(img, mask, patch_width, patch_height)
                patches += get_random_patches(img, mask, patch_width, patch_height, patch_num)
            
            for ix_0, (img, mask) in enumerate(patches):

            
                # Making the mask binary, as it is meant to be.
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

                mask[mask < 127] = 0
                mask[mask >= 127] = 255

                # da_suffix = f"_{is_mirror}{is_gauss}{rotation_angle}{zoom_level}{zoom_position}"
                da_suffix = f"_{ix_0}"
                new_img_name = f"{img_name}{da_suffix}"
                # Mask has same name as image, just different file extension.
                new_folderpath = osp.join(da_folderpath, split)
                new_img_path = osp.join(new_folderpath, 'Images')
                new_mask_path = osp.join(new_folderpath, 'Masks')

                os.makedirs(new_folderpath, exist_ok=True)
                os.makedirs(osp.join(new_folderpath, 'Images'), exist_ok=True)
                os.makedirs(osp.join(new_folderpath, 'Masks'), exist_ok=True)

                # # Saving the image and mask
                # cv2.imwrite(new_img_path, img)
                # cv2.imwrite(new_mask_path, mask)
                
                img = smart_conversion(img, 'Image', 'uint8')
                mask = smart_conversion(mask, 'Image', 'uint8')
                

                save_img(img, new_img_path, new_img_name + '.jpg')
                save_img(mask, new_mask_path, new_img_name + '.png')

                # input("Press Enter to continue...")

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])



                # while True:
                #     plt.pause(0.1)


        
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e


























# old:
"""
def get_tiling(img, mask, num_patches_x, num_patches_y, patch_width, patch_height):
    patches = []

    for i in range(num_patches_x):
        for j in range(num_patches_y):
            img_patch = img[j * patch_height: (j + 1) * patch_height, i * patch_width: (i + 1) * patch_width, :]
            mask_patch = mask[j * patch_height: (j + 1) * patch_height, i * patch_width: (i + 1) * patch_width, :]

            patches.append((img_patch, mask_patch))

    return patches

def get_random_patches(img, mask, patch_width, patch_height, num_of_patches):

    patches = []

    for i in range(num_of_patches):
        x = np.random.randint(0, img.shape[0] - patch_width)
        y = np.random.randint(0, img.shape[1] - patch_height)

        img_patch = img[x: x + patch_width, y: y + patch_height, :]
        mask_patch = mask[x: x + patch_width, y: y + patch_height, :]

        patches.append((img_patch, mask_patch))
    
    return patches



import argparse

if __name__ == "__main__":


    # Suggested plan:
    
    # This is mostly meant for test and val splits. In this way we can then go over all of the parts of the image and get how good the model was.
    # In this way our test and val scores do not have overlapping pixels, which would skew the result.
    

    # cores probs help because of C vectorization in numpy operations
    # srun -c 30 --gpus=A100  python3 v_partial_preaug_data_creator.py --ow 1500 --oh 1000

    parser = argparse.ArgumentParser(description='Data Augmentation for Vein Sclera Segmentation')
    parser.add_argument('--ow', type=int, default=2048, help='Output width of the image')
    parser.add_argument('--oh', type=int, default=1024, help='Output height of the image')
    parser.add_argument('--pw', type=int, default=128, help='Patch width of the image')
    parser.add_argument('--ph', type=int, default=128, help='Patch height of the image')
    parser.add_argument('--px', type=int, default=16, help="Num of patches in the x direction")
    parser.add_argument('--py', type=int, default=8, help="Num of patches in the y direction")
    parser.add_argument('--fp', type=str, default='./vein_sclera_data', help='Folder path of the dataset')
    parser.add_argument('--dafp', type=str, default='./vein_sclera_data_patchified', help='Augmented data folderpath. To this name the ow and oh get appended.')
    parser.add_argument('--split', type=str, default='test', help='Split of the dataset')
    parser.add_argument('--mode', type=str, default='tiling', help='Modes tiling, random, both. Tiling is used for test and val. Both is generally used for train.')

    args = parser.parse_args()
    
    folderpath = args.fp
    da_folderpath = args.dafp
    split = args.split
    mode = args.mode
    
    output_width = args.ow
    output_height = args.oh
    patch_width = args.pw
    patch_height = args.ph
    num_patches_x = args.px
    num_patches_y = args.py

    da_folderpath = f"{da_folderpath}_{patch_height}_{patch_width}"

    if output_width % num_patches_x != 0 or output_height % num_patches_y != 0:
        raise ValueError("The output width and height should be divisible by the number of patches in x and y direction respectively")
    
    if patch_height * num_patches_y != output_height or patch_width * num_patches_x != output_width:
        raise ValueError("The patch width and height should be such that the output width and height are divisible by the number of patches in x and y direction respectively")

        
    images_without_mask = []
    images_with_masks = []
    
    imgs_path = osp.join(folderpath, split, 'Images')
    masks_path = osp.join(folderpath, split, 'Masks')
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


    try:
        for ix, img_name in enumerate(images_with_masks):


        

            image_path = osp.join(imgs_path, f"{img_name}.jpg")

            img = Image.open(image_path).convert("RGB")



            # Since our data augmentation should simulate errors and changes in how the image is taken, we should do gamma correction and clahe after the augmentation.
            # Because we should first simulate the errors, that is then what our taken img would have been,
            # and then we do our preprocessing (correction) from there.
            # We pretend that the data augmentations were actually pictures that we took.




            file_name_no_suffix = img_name
            mask = get_mask(masks_path, file_name_no_suffix) # is of type Image.Image


            # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')
            mask = smart_conversion(mask, 'ndarray', 'uint8')

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)

            if mode == 'tiling':
                patches = get_tiling(img, mask, num_patches_x, num_patches_y, patch_width, patch_height)
            elif mode == 'random':
                patches = get_random_patches(img, mask, patch_width, patch_height, num_patches_x * num_patches_y)
            elif mode == 'both':
                patches = get_tiling(img, mask, num_patches_x, num_patches_y, patch_width, patch_height)
                patches += get_random_patches(img, mask, patch_width, patch_height, num_patches_x * num_patches_y)
            
            for ix_0, (img, mask) in enumerate(patches):

            
                # Making the mask binary, as it is meant to be.
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

                mask[mask < 127] = 0
                mask[mask >= 127] = 255

                # da_suffix = f"_{is_mirror}{is_gauss}{rotation_angle}{zoom_level}{zoom_position}"
                da_suffix = f"_{ix_0}"
                new_img_name = f"{img_name}{da_suffix}"
                # Mask has same name as image, just different file extension.
                new_folderpath = osp.join(da_folderpath, split)
                new_img_path = osp.join(new_folderpath, 'Images')
                new_mask_path = osp.join(new_folderpath, 'Masks')

                os.makedirs(new_folderpath, exist_ok=True)
                os.makedirs(osp.join(new_folderpath, 'Images'), exist_ok=True)
                os.makedirs(osp.join(new_folderpath, 'Masks'), exist_ok=True)

                # # Saving the image and mask
                # cv2.imwrite(new_img_path, img)
                # cv2.imwrite(new_mask_path, mask)
                
                img = smart_conversion(img, 'Image', 'uint8')
                mask = smart_conversion(mask, 'Image', 'uint8')
                

                save_img(img, new_img_path, new_img_name + '.jpg')
                save_img(mask, new_mask_path, new_img_name + '.png')

                # input("Press Enter to continue...")

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])



                # while True:
                #     plt.pause(0.1)


        
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e
"""