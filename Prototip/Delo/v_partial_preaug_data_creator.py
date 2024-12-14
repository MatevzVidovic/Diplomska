#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:47:44 2019

@author: Aayush

This file contains the dataloader and the augmentations and preprocessing done

Required Preprocessing for all images (test, train and validation set):
1) Gamma correction by a factor of 0.8
2) local Contrast limited adaptive histogram equalization algorithm with clipLimit=1.5, tileGridSize=(8,8)
3) Normalization
    
Train Image Augmentation Procedure Followed 
1) Random horizontal flip with 50% probability.
2) Starburst pattern augmentation with 20% probability. 
3) Random length lines augmentation around a random center with 20% probability. 
4) Gaussian blur with kernel size (7,7) and random sigma with 20% probability. 
5) Translation of image and masks in any direction with random factor less than 20.
"""


import os.path as osp
import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)




from albumentations import Compose, ShiftScaleRotate


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







# Horizontal flip and gaussian blur funtions:

def horizontal_flip(img, mask):
    # Takes PIL img as input, returns PIL img.
    # If input not PIL img, automatically transforms it to PIL img.

    img = smart_conversion(img, 'Image', 'uint8')
    mask = smart_conversion(mask, 'Image', 'uint8')

    aug_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    aug_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    #py_log_always_on.log_time(MY_LOGGER, "test_da")
    return aug_img, aug_mask



def gaussian_blur(img, possible_sigma_vals_list=range(2, 7), ker_size=7):
    # Takes np.array img as input. Returns np.array.
    # If input not np.array, automatically transforms it to np.array.

    img = smart_conversion(img, 'ndarray', 'uint8')

    sigma_ix = np.random.randint(len(possible_sigma_vals_list))
    sigma_value = possible_sigma_vals_list[sigma_ix]

    aug_img = cv2.GaussianBlur(img, (ker_size, ker_size), sigma_value)

    #py_log_always_on.log_time(MY_LOGGER, "test_da")
    return aug_img








# Random rotation block of functions:


def maxHist(row):
    # Create an empty stack. The stack holds
    # indexes of hist array / The bars stored
    # in stack are always in increasing order
    # of their heights.
    stack = []

    # Top of stack
    top_val = 0

    # Initialize max area in current
    max_area = 0
    height_of_max_area = 0
    left_index = 0
    right_index = 0
    # row (or histogram)

    area = 0  # Initialize area with current top

    # Run through all bars of given
    # histogram (or row)
    i = 0
    while (i < len(row)):

        # If this bar is higher than the
        # bar on top stack, push it to stack
        if (len(stack) == 0) or (row[stack[-1]] <= row[i]):
            stack.append(i)
            i += 1

        # This will pop stuff from stack until we get a bar that is lower than the currently next bar.
        else:

            # If this bar is lower than top of stack,
            # then calculate area of rectangle with
            # stack top as the smallest (or minimum
            # height) bar. 'i' is 'right index' for
            # the top and element before top in stack
            # is 'left index'
            top_val = row[stack.pop()]
            area = top_val * i

            if len(stack) > 0:
                # i is the ix of the curr next bar
                # stack[-1] + 1 is the ix of the currently popped bar
                area = top_val * (i - (stack[-1] + 1))

            if area > max_area:
                max_area = area
                height_of_max_area = top_val
                left_index = stack[-1] + 1 if stack else 0
                right_index = i

    # Now pop the remaining bars from stack
    # and calculate area with every popped
    # bar as the smallest bar
    while (len(stack)):
        top_val = row[stack.pop()]
        area = top_val * i
        if (len(stack)):
            area = top_val * (i - stack[-1] - 1)

        if area > max_area:
            max_area = area
            height_of_max_area = top_val
            left_index = stack[-1] + 1 if stack else 0
            right_index = i - 1

    return max_area, left_index, right_index, height_of_max_area



def max_histogram_area_with_indices(heights):
    stack = []
    max_area = 0
    left_index = 0
    right_index = 0
    height_index = 0
    index = 0

    while index < len(heights):
        if len(stack) == 0 or heights[stack[-1]] <= heights[index]:
            stack.append(index)
            index += 1
        else:
            top_of_stack = stack.pop()
            area = (heights[top_of_stack] *
                    ((index - stack[-1] - 1) if stack else index))
            if area > max_area:
                max_area = area
                height_index = top_of_stack
                left_index = stack[-1] + 1 if stack else 0
                right_index = index - 1

    while stack:
        top_of_stack = stack.pop()
        area = (heights[top_of_stack] *
                ((index - stack[-1] - 1) if stack else index))
        if area > max_area:
            max_area = area
            height_index = top_of_stack
            left_index = stack[-1] + 1 if stack else 0
            right_index = index - 1

    return max_area, left_index, right_index, height_index



def maximal_rectangle_with_indices(matrix):
    
    max_area = 0
    max_coords = None
    heights = [0] * len(matrix[0])

    for row_index, row in enumerate(matrix):
        for i in range(len(row)):
            heights[i] = heights[i] + 1 if row[i] == 1 else 0

        # area, left, right, height = max_histogram_area_with_indices(heights)
        area, left, right, height = maxHist(heights)
        if area > max_area:
            max_area = area
            max_coords = (row_index - height + 1, left, row_index, right)

    return max_area, max_coords


def crop_to_nonzero_in_fourth_channel(img, mask, crop="all_zero"):

    # crop can be "all_zero" or "hug_nonzero"
    

    try:

        # If you want to crop the image to the non-zero values in the fourth channel of the image and mask.
        # Zero values still remain, the frame of img just hugs the non-zero values now.

        if crop == "hug_nonzero":
        
            # Assuming the last channel is the one added with ones
            non_zero_indices = np.where(img[..., -1] != 0)
            # Note: img[..., -1] is the same as img[:, :, -1]
            
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            
            # Crop the image and mask
            cropped_img = img[min_y:max_y+1, min_x:max_x+1]
            cropped_mask = mask[min_y:max_y+1, min_x:max_x+1]
        


        # This squeezes the frame until there are no zero values in the fourth channel anywhere:
        elif crop == "all_zero":
            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img[...,:-1], mask])


            # This doesn't give you the biggest AREA!!!
            # At least not in general.
            # It just works well enough.
            # And in our case of a rotated rect, it might even work.

            method = 1
            if method == 0:

                # Find the rect that is inside the area of 1s in the fourth channel
                # that has the maximum area.

                # Since this is a rotated rectangle, this should work:
                # Go from the right. Keep checking the pixel in the top row and the bottom row.
                # As soon as you one of them is 1, crop to that distance.
                # Now do the same from the left.
                # Now you have a new cropped image, which was cropped the minimum amount that you would have to anyway from both sides.

                # Now on the new cropped image go from top down and bottom up, but both pixels have to be 1 and that's when you stop.

                y_min = 0
                y_max = img.shape[0] - 1
                x_min = 0
                x_max = img.shape[1] - 1
                
                # for fraction_of_img == 100,
                # if the img dim  size is <= 99, launch step is 1, 
                # so it can't possibly skip an area of 1s, and it is performance neutral.
                # if the img dim size is 1000, launch step is 11, so it most likely won't skip over an area of 1s.
                # It will, however, make the code 10x faster.
                # So even if we did skip an area of 1s, we didn't waste much time.

                # fraction_of_img == 100 might even be too conservative.
                # The thing is, the middle pixel in the uppermost row is bound to be 1. And a bunch of pixels around it too.

                # We don't want fraction_of_img to be too small from the speedup perspective.
                # Because then we are doing a lot of if-checks for nothing because we are constantly going into 1s.

                # fraction_of_img = 100
                fraction_of_img = 50 # 20x speedup

                # ensure launch step is at least 1, so we aren't doing these if-checks
                # for nothing - at worst it's neutral (1 if-check for movement of 1)
                launch_step = img.shape[0] // fraction_of_img + 1 # small enough that it doesn't skip over the 1s to the other side of 0s
                while img[y_min, x_min, -1] == 0 and img[y_max, x_min, -1] == 0:
                    x_min += 1
                    if img[y_min, x_min + launch_step, -1] == 0 and img[y_max, x_min + launch_step, -1] == 0:
                        x_min += launch_step
                    else:
                        launch_step = (launch_step // 2) + 1 # so it is always at least 1 and the if check isn't for nothing
                
                launch_step = img.shape[0] // fraction_of_img + 1
                while img[y_min, x_max, -1] == 0 and img[y_max, x_max, -1] == 0:
                    x_max -= 1
                    if img[y_min, x_max - launch_step, -1] == 0 and img[y_max, x_max - launch_step, -1] == 0:
                        x_max -= launch_step
                    else:
                        launch_step = (launch_step // 2) + 1



                # Now we have the new leftmost and rightmost column. It is as if we cropped it from the left and right.
                
                launch_step = img.shape[1] // fraction_of_img + 1
                while img[y_min, x_min, -1] == 0 or img[y_min, x_max, -1] == 0:
                    y_min += 1
                    if img[y_min + launch_step, x_min, -1] == 0 or img[y_min + launch_step, x_max, -1] == 0:
                        y_min += launch_step
                    else:
                        launch_step = (launch_step // 2) + 1
                
                launch_step = img.shape[1] // fraction_of_img + 1
                while img[y_max, x_min, -1] == 0 or img[y_max, x_max, -1] == 0:
                    y_max -= 1
                    if img[y_max - launch_step, x_min, -1] == 0 or img[y_max - launch_step, x_max, -1] == 0:
                        y_max -= launch_step
                    else:
                        launch_step = (launch_step // 2) + 1
            


                # If the area is too small, we probably made a mistake with the launch_step.
                # (We stepped over the 1s to the other side of the 0s.)
                # So we just do it the slow way.
                a = (y_max - y_min)
                b = (x_max - x_min)
                if  a < 0 or b < 0 or a*b < 0.5 * img.shape[0] * img.shape[1]:

                    y_min = 0
                    y_max = img.shape[0] - 1
                    x_min = 0
                    x_max = img.shape[1] - 1
                    
                    while img[y_min, x_min, -1] == 0 and img[y_max, x_min, -1] == 0:
                        x_min += 1
                    
                    while img[y_min, x_max, -1] == 0 and img[y_max, x_max, -1] == 0:
                        x_max -= 1

                    # Now we have the new leftmost and rightmost column. It is as if we cropped it from the left and right.
                    
                    while img[y_min, x_min, -1] == 0 or img[y_min, x_max, -1] == 0:
                        y_min += 1
                    
                    while img[y_max, x_min, -1] == 0 or img[y_max, x_max, -1] == 0:
                        y_max -= 1
            

            elif method == 1:

                y_min = 0
                y_max = img.shape[0] - 1
                x_min = 0
                x_max = img.shape[1] - 1
                
                while img[y_min, x_min, -1] == 0 and img[y_max, x_min, -1] == 0:
                    x_min += 1
                
                while img[y_min, x_max, -1] == 0 and img[y_max, x_max, -1] == 0:
                    x_max -= 1

                # Now we have the new leftmost and rightmost column. It is as if we cropped it from the left and right.
                
                while img[y_min, x_min, -1] == 0 or img[y_min, x_max, -1] == 0:
                    y_min += 1
                
                while img[y_max, x_min, -1] == 0 or img[y_max, x_max, -1] == 0:
                    y_max -= 1


            # Works but uses more python so it's probably slower:
            else:


                # We use the maximum rectangle algorithm for this, which uses the maximum histogram algorithm.
                # https://www.geeksforgeeks.org/maximum-size-rectangle-binary-sub-matrix-1s/
                # https://www.youtube.com/watch?v=zx5Sw9130L0 !!!!!!!!
                # https://www.youtube.com/watch?v=ZmnqCZp9bBs
                # https://www.youtube.com/watch?v=g8bSdXCG-lA 

                # I tried with my own algorithm, but this one is faster.

                _, (y_min, x_min, y_max, x_max) = maximal_rectangle_with_indices(img[..., -1])





            cropped_img = img[y_min:y_max+1, x_min:x_max+1]
            cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
            


        return cropped_img, cropped_mask
    
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e


def rotation(img, mask, angle, rotate_type="shrink"):

    
    # THE SIZE OF THE IMAGE MIGHT BE SMALLER AFTER THIS FUNCTION!!!!!

    # IF NOT, IT INTRODUCES BLACK PIXELS!!!!!

    # CREATES BLURRYNESS!!!!! (because of interpolation)


    # The size of the returned image MAY NOT BE THE SAME as the input image.
    # If using "shrink", the size of the returned image will be smaller.
    
    # Rotating the image around its center introduces black pixels.
    # The size of the image remains the same in the method we use
    # (could also get bigger if we wanted to retain all pixels of the original img).
    # If "shrink" we shrink the frame of the image from each direction until there are no newly introduced black pixels.
    # (Don't worry, it doesn't actually look at blackness of pixels to crop, it looks at the fourth channel we added.
    # So if your image is mostly black at the edges, don't worry, we won't crop into your img.)


    try:
    
        # These give coppies anyway, so don't worry about changing them.
        img = smart_conversion(img, 'ndarray', 'uint8')
        mask = smart_conversion(mask, 'ndarray', 'uint8')

        
        
        # Add a channel with ones
        ones_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate((img, ones_channel), axis=-1)
        # mask = np.concatenate((mask, ones_channel), axis=-1)
        

        # Get the image center
        center = (img.shape[1] // 2, img.shape[0] // 2)

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        aug_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        aug_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        #py_log_always_on.log_time(MY_LOGGER, "test_da")

        # with np.printoptions(threshold=np.inf):
        #     # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([aug_img[...,:-1], aug_mask])

        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([aug_img[...,:-1], aug_mask])

        if rotate_type == "shrink":
            aug_img, aug_mask = crop_to_nonzero_in_fourth_channel(aug_img, aug_mask, crop="all_zero")
        

        # Remove the last channel
        aug_img = aug_img[..., :-1]


        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

        #py_log_always_on.log_time(MY_LOGGER, "test_da")
        return aug_img, aug_mask

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e








def zoom_and_offset(img, mask, scale_percent, offset_percent_x=0.5, offset_percent_y=0.5):

    try:
        # This function zooms in on a random part of the image.


        # Takes np.array img as input. Returns np.array.
        # If input not np.array, automatically transforms it to np.array.

        img = smart_conversion(img, 'ndarray', 'uint8')
        mask = smart_conversion(mask, 'ndarray', 'uint8')

        remain_percent = 1 - scale_percent
        
        

        # How many picels we keep in each direction:
        vert_pix_num = int(img.shape[0] * remain_percent)
        horiz_pix_num = int(img.shape[1] * remain_percent)

        # In the very extreme case, we should keep at least one pixel:
        if vert_pix_num <= 0:
            vert_pix_num = 1
        if horiz_pix_num <= 0:
            horiz_pix_num = 1
        
        max_offset_vert = img.shape[0] - vert_pix_num
        max_offset_horiz = img.shape[1] - horiz_pix_num

        offset_vert = int(offset_percent_y * max_offset_vert)
        offset_horiz = int(offset_percent_x * max_offset_horiz)
        
        aug_img = img[offset_vert:offset_vert+vert_pix_num, offset_horiz:offset_horiz+horiz_pix_num, :]
        aug_mask = mask[offset_vert:offset_vert+vert_pix_num, offset_horiz:offset_horiz+horiz_pix_num, :]

        # aug_img = img[vert_pix_num:-vert_pix_num, horiz_pix_num:-horiz_pix_num, :]
        # aug_mask = mask[vert_pix_num:-vert_pix_num, horiz_pix_num:-horiz_pix_num, :]

        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])


        #py_log_always_on.log_time(MY_LOGGER, "test_da")
        return aug_img, aug_mask

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER)
        raise e
















def mask_exists(mask_folder_path, file_name_no_suffix):
    mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
    return osp.exists(mask_filename)

def get_mask(mask_folder_path, file_name_no_suffix) -> np.array:

    mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')

    mask = Image.open(mask_filename).convert("RGB")
    return mask






import argparse

if __name__ == "__main__":
    


    parser = argparse.ArgumentParser(description='Data Augmentation for Vein Sclera Segmentation')
    parser.add_argument('--fp', type=str, default='./vein_sclera_data', help='Folder path of the dataset')
    parser.add_argument('--dafp', type=str, default='./partial_preaug_vein_sclera_data', help='Data augmented folderpath. Folder path of the augmented dataset')
    parser.add_argument('--split', type=str, default='train', help='Split of the dataset')
    parser.add_argument('--ow', type=int, default=512, help='Output width of the image')
    parser.add_argument('--oh', type=int, default=512, help='Output height of the image')

    args = parser.parse_args()
    
    folderpath = args.fp
    da_folderpath = args.dafp
    split = args.split
    
    output_width = args.ow
    output_height = args.oh

        
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


            base_img = smart_conversion(img, 'Image', 'uint8')
            base_mask = smart_conversion(mask, 'Image', 'uint8')


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])


            # Idea:
            # On one side we have on-the-fly augmentations, which are done during training.
            # But augmenting a 3000x2000 image is slow. 3 ops of 1.5 secs.
            # So what if we augment the data beforehand, and then just load it in during training?
            # But then:
            #    - The augmentations are static (instead of always getting a slightly differently augmented image, we only have one random choice of augmentation saved)
            #   - It's hard to keep impact of non-augmented images high: We have an original image. We tak 3 rotated ones from it. For each one of them we do 9 different zooms.
            #     So we have 27 images from one original. If we have 100 originals, we have 2700 images. But only 100 of them are original. So the impact of the original is low.
            #     And this is a problem if the augmented images are not perfectly what we want to train on. Eg we are interested at a specific scale, but we do zooms because our dataset is too small.
            #     So this helps, but if we have such an imbalance of actual correct scale (original) and incorrect scale (augmented), then we will barely be learning the correct scale.
            #     So we need to keep the impact of the original high. And partial augmentation does that, see for yourself.

            # So we want to do preaugmentations only for some augmentations that are still quite close to the original.


            # Pipeline of partial augmentations:
            # - mirror or dont
            # - slight gauss or dont
            # - if gauss and otherwise also very augmented, randomly abort 90% of the time, so I don't have that many gaussed up images.
            # - rotate for different fixed angles
            # - then save the image and mask

            # And on the fly we will only sometimes be doing zooms.
            # And only sometimes be doing smaller random rotations.


            # (1.25 is a guesstimate of how often we actually do gauss. If we do it 0.25, then we get 1.0 for not gauss, and so togehter it's 1.25)
            # For each original image, we will have 2*(2*1.25)*3 = 15 images.
            # So if original is 0.5GB, we will have 7.5GB of data.
            # We had 89 imgs, now we have 1335 imgs. This is a nice number.



            for ix_0, is_mirror in enumerate([False, True]):
                for ix_1, is_gauss in enumerate([False, True]):
                    for ix_2, rotation_angle in  enumerate([None, 10, 20]):

                        img = base_img.copy()
                        mask = base_mask.copy()


                        # If the img is already augmented with both rotation and zoom, make it very rare to be gaussed also. Those imgs are already very different,
                        # and we don't want too much gaussing, because too different from our original dataset.
                        if is_gauss and (rotation_angle is not None) and np.random.random() < 0.9:
                            continue


                        if is_mirror:

                            img, mask = horizontal_flip(img, mask)

                        if is_gauss:

                            # Has to be here, because at this point the img is surely still a PIL image, so .size() is correct
                            ksize = (img.size[0]//20)
                            ksize = ksize+1 if ksize % 2 == 0 else ksize
                            img = gaussian_blur(img, possible_sigma_vals_list=np.linspace(1, 5, 50), ker_size=ksize)


                        if rotation_angle is not None:
                            img, mask = rotation(img, mask, angle= ((2*rotation_angle)*np.random.random() - rotation_angle), rotate_type="shrink")
                        

                        # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
                        img = smart_conversion(img, 'ndarray', 'uint8')
                        mask = smart_conversion(mask, 'ndarray', 'uint8')

                        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

                        # performing the necessary resizing
                        img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                        mask = cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                        
                        # Making the mask binary, as it is meant to be.
                        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

                        mask[mask < 127] = 0
                        mask[mask >= 127] = 255




                        # da_suffix = f"_{is_mirror}{is_gauss}{rotation_angle}{zoom_level}{zoom_position}"
                        da_suffix = f"_{ix_0}{ix_1}{ix_2}"
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
