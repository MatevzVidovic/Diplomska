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


import os
import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on



MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)



from albumentations import Compose, ShiftScaleRotate


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






# Initialize the last call time
last_call_time = None

import time
import inspect

def print_time_since_last_call():
    global last_call_time
    
    # Get the current time
    current_time = time.perf_counter()
    

    # Get the caller's frame
    caller_frame = inspect.currentframe().f_back
    caller_line = caller_frame.f_lineno
    caller_func = caller_frame.f_code.co_name
    caller_file = caller_frame.f_code.co_filename
    caller_file = osp.basename(caller_file)
    
    # Calculate the time elapsed since the last call
    if last_call_time is not None:
        elapsed_time = current_time - last_call_time
        print(f"Time since last call: {elapsed_time:.6f} seconds. Called from line {caller_line} in {caller_func} in {caller_file}.")
    else:
        print(f"This is the first call. Called from line {caller_line} in {caller_file}.")
    
    # Update the last call time
    last_call_time = current_time
    




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







# Horizontal flip and gaussian blur funtions:

def random_horizontal_flip(img, mask, prob=0.5):
    # Takes PIL img as input, returns PIL img.
    # If input not PIL img, automatically transforms it to PIL img.
    
    if np.random.random() > prob:
        return img,mask


    img = smart_conversion(img, 'Image', 'uint8')
    mask = smart_conversion(mask, 'Image', 'uint8')

    aug_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    aug_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    #print_time_since_last_call()
    return aug_img, aug_mask



def gaussian_blur(img, possible_sigma_vals_list=range(2, 7), ker_size=7, prob=0.2):
    # Takes np.array img as input. Returns np.array.
    # If input not np.array, automatically transforms it to np.array.

    if np.random.random() > prob:
        return img
    
    img = smart_conversion(img, 'ndarray', 'uint8')

    sigma_ix = np.random.randint(len(possible_sigma_vals_list))
    sigma_value = possible_sigma_vals_list[sigma_ix]

    aug_img = cv2.GaussianBlur(img, (ker_size, ker_size), sigma_value)

    #print_time_since_last_call()
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


def random_rotation(img, mask, max_angle=15, rotate_type="shrink", prob=0.2):
    
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


    if np.random.random() > prob:
        return img, mask
    
    # These give coppies anyway, so don't worry about changing them.
    img = smart_conversion(img, 'ndarray', 'uint8')
    mask = smart_conversion(mask, 'ndarray', 'uint8')

    
    
    # Add a channel with ones
    ones_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
    img = np.concatenate((img, ones_channel), axis=-1)
    # mask = np.concatenate((mask, ones_channel), axis=-1)
    

    # Randomly choose an angle between -max_angle and max_angle
    angle = np.random.uniform(-max_angle, max_angle)

    # Get the image center
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    aug_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    aug_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    #print_time_since_last_call()

    # with np.printoptions(threshold=np.inf):
    #     # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([aug_img[...,:-1], aug_mask])

    # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([aug_img[...,:-1], aug_mask])

    if rotate_type == "shrink":
        aug_img, aug_mask = crop_to_nonzero_in_fourth_channel(aug_img, aug_mask, crop="all_zero")
    

    # Remove the last channel
    aug_img = aug_img[..., :-1]


    # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

    #print_time_since_last_call()
    return aug_img, aug_mask






# Translation block of functions:


def shift_horiz(img, shift):
    dims_num = len(img.shape)
    if dims_num == 2:
        img = np.expand_dims(img, axis=2)
    
    if shift > 0:
        img = np.pad(img, ((0, 0), (shift, 0), (0, 0)), mode='constant')[:, :-shift, :]
    elif shift < 0:
        shift = -shift
        img = np.pad(img, ((0, 0), (0, shift), (0, 0)), mode='constant')[:, shift:, :]
    
    if dims_num == 2:
        img = np.squeeze(img, axis=2)
    
    return img

def shift_vert(img, shift):
    dims_num = len(img.shape)
    if dims_num == 2:
        img = np.expand_dims(img, axis=2)
    
    if shift > 0:
        img = np.pad(img, ((shift, 0), (0, 0), (0, 0)), mode='constant')[:-shift, :, :]
    elif shift < 0:
        shift = -shift
        img = np.pad(img, ((0, shift), (0, 0), (0, 0)), mode='constant')[shift:, :, :]
    
    if dims_num == 2:
        img = np.squeeze(img, axis=2)
    
    return img



def crop_horiz(img, hor_shift):
    dims_num = len(img.shape)
    if dims_num == 2:
        img = np.expand_dims(img, axis=2)
    
    if hor_shift > 0:
        img = img[:, :-hor_shift, :]
    elif hor_shift < 0:
        hor_shift = -hor_shift
        img = img[:, hor_shift:, :]
    
    if dims_num == 2:
        img = np.squeeze(img, axis=2)
    
    return img

def crop_vert(img, vert_shift):

    dims_num = len(img.shape)
    if dims_num == 2:
        img = np.expand_dims(img, axis=2)
    
    if vert_shift > 0:
        img = img[:-vert_shift, :, :]
    elif vert_shift < 0:
        vert_shift = -vert_shift
        img = img[vert_shift:, :, :]
    
    if dims_num == 2:
        img = np.squeeze(img, axis=2)
    
    return img



def crop(img, hor_shift, vert_shift):

    

    dims_num = len(img.shape)
    if dims_num == 2:
        img = np.expand_dims(img, axis=2)

    

    if hor_shift > 0 and vert_shift > 0:
        img = img[:-vert_shift, :-hor_shift, :]
    elif hor_shift > 0 and vert_shift < 0:
        vert_shift = -vert_shift
        img = img[vert_shift:, :-hor_shift, :]
    elif hor_shift < 0 and vert_shift > 0:
        hor_shift = -hor_shift
        img = img[:-vert_shift, hor_shift:, :]
    elif hor_shift < 0 and vert_shift < 0:
        hor_shift = -hor_shift
        vert_shift = -vert_shift
        img = img[vert_shift:, hor_shift:, :]
    
    
    
    if dims_num == 2:
        img = np.squeeze(img, axis=2)
    
    


    return img

def translation(img, mask, v_max_perc=0.1, h_max_perc=0.1, trans_type="crop", prob=0.2):
    # THE SIZE OF THE IMAGE MIGHT BE SMALLER AFTER THIS FUNCTION!!!!!
    # IF NOT, IT INTRODUCES BLACK PIXELS!!!!!

    # trans_type can be "crop" or "black_pixels"

    # "crop" doesn't introduce black pixels. Just crops the image.
    # This is like tranlating the image and then zooming in on the image to remove the black pixels.
    # "black_pixels" introduces black pixels. It shifts the image and doesn't remove the black pixels.


    # The size of the returned image MAY NOT BE THE SAME as the input image!!!!
    # If using "crop", the size of the returned image will be smaller.
    # To get the image of the same size back, we need to resize it after this function.
    # Resize it with cv2.resize(img, (new_width, new_height), interpolation=cv2.something)  
    # to a larger size with any interpolation method you like. (cv.INTER_LINEAR, cv.INTER_CUBIC, cv.INTER_NEAREST, cv.INTER_LANCZOS4)
    

    
    
    # Takes np.array img as input. Returns np.array.
    # If input not np.array, automatically transforms it to np.array.

    if np.random.random() > prob:
        return img, mask
    
    img = smart_conversion(img, 'ndarray', 'uint8')
    mask = smart_conversion(mask, 'ndarray', 'uint8')
    
    # Calculate the maximum translation in pixels
    v_max_shift = int(v_max_perc * img.shape[0])
    h_max_shift = int(h_max_perc * img.shape[1])
    
    # Randomly choose the translation amount
    v_shift = np.random.randint(-v_max_shift, v_max_shift + 1)
    h_shift = np.random.randint(-h_max_shift, h_max_shift + 1)

    if trans_type == "crop":
        # Apply the shifts
        img = crop(img, h_shift, v_shift)
        mask = crop(mask, h_shift, v_shift)
    elif trans_type == "black_pixels":
        # Apply the shifts
        img = shift_vert(img, v_shift)
        img = shift_horiz(img, h_shift)
        mask = shift_vert(mask, v_shift)
        mask = shift_horiz(mask, h_shift)
    
    #print_time_since_last_call()
    return img, mask






def scale(img, mask, max_scale_percent=0.2, scale_type="only_zoom", prob=0.2):

    # IF scale_type != "only_zoom", IT INTRODUCES BLACK PIXELS!!!!!

    # scale_type can be "only_zoom" or "zoom_and_shrink"

    # Takes np.array img as input. Returns np.array.
    # If input not np.array, automatically transforms it to np.array.

    if np.random.random() > prob:
        return img, mask

    img = smart_conversion(img, 'ndarray', 'uint8')
    mask = smart_conversion(mask, 'ndarray', 'uint8')

    scale_percent = np.random.uniform(-max_scale_percent, max_scale_percent)

    if scale_type == "only_zoom":
        scale_percent = abs(scale_percent)
    elif scale_type != "zoom_and_shrink":
        raise ValueError("scale_type must be 'only_zoom' or 'zoom_and_shrink'")
    
    
    # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])
    if scale_percent >= 0:
        halved_scale_percent = scale_percent / 2
        vert_pix_num = int(img.shape[0] * halved_scale_percent)
        horiz_pix_num = int(img.shape[1] * halved_scale_percent)

        # We have to crop by at leas 1.
        # Otherwise, e.g. img[0:0, 3:-1, :] can happen and this is empty.
        # So wasy fix is to ensure that we crop by at least 1.
        if vert_pix_num == 0:
            vert_pix_num = 1
        if horiz_pix_num == 0:
            horiz_pix_num = 1
            

        aug_img = img[vert_pix_num:-vert_pix_num, horiz_pix_num:-horiz_pix_num, :]
        aug_mask = mask[vert_pix_num:-vert_pix_num, horiz_pix_num:-horiz_pix_num, :]

        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

    else:
        
        aug = Compose([
            ShiftScaleRotate(shift_limit=0, scale_limit=(scale_percent, scale_percent), rotate_limit=0, p=1, border_mode=cv2.BORDER_CONSTANT),
        ], additional_targets={'mask': 'image'})

        augmented = aug(image=img, mask=mask)

        aug_img = augmented['image']
        aug_mask = augmented['mask']

    #print_time_since_last_call()
    return aug_img, aug_mask






class IrisDataset(Dataset):
    def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, clipLimit=1.5, **kwargs):
        
        self.transform = transform
        self.filepath= osp.join(filepath, split)
        
        self.input_width = kwargs['input_width']
        self.input_height = kwargs['input_height']
        self.output_width = kwargs['output_width']
        self.output_height = kwargs['output_height']

        self.testrun_length = kwargs['testrun_size']
        
        self.split = split
        self.classes = n_classes

        self.images_without_mask = [] # seznam slik, ki nimajo maske
        

        images_with_masks = []
        
        all_images = os.listdir(osp.join(self.filepath,'Images'))
        masks_folder = osp.join(self.filepath,'Masks')
        all_images.sort()



        for file in all_images:
            
            file_without_suffix = file.strip(".jpg")

            if self.mask_exists(masks_folder, file_without_suffix):
                images_with_masks.append(file_without_suffix)


        self.list_files = images_with_masks
        self.testrun = testrun

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))

        #summary
        print('summary for ' + str(split))
        print('valid images: ' + str(len(self.list_files)))


    def __len__(self):
        real_size = len(self.list_files)
        if self.testrun:
            return min(self.testrun_length, real_size)
        return real_size
    
    def mask_exists(self, mask_folder_path, file_name_no_suffix):
        mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
        return osp.exists(mask_filename)

    def get_mask(self, mask_folder_path, file_name_no_suffix) -> np.array:

        mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
        try:
            mask = Image.open(mask_filename).convert("RGB")
            return mask

        except FileNotFoundError:
            print('file not found: ' + str(mask_filename))
            self.images_without_mask.append(file_name_no_suffix)
            return None

    @py_log.log(passed_logger=MY_LOGGER)
    def __getitem__(self, idx):

        try:


            image_path = osp.join(self.filepath,'Images',self.list_files[idx]+'.jpg')

            img = Image.open(image_path).convert("RGB")

            preprocessing_resize_size = (1024, 1024) # img.size # or just (1024, 1024)

            img = img.resize(preprocessing_resize_size, Image.BILINEAR) #ali Image.BICUBIC ali Image.LANCZOS



            # Since our data augmentation should simulate errors and changes in how the image is taken, we should do gamma correction and clahe after the augmentation.
            # Because we should first simulate the errors, that is then what our taken img would have been,
            # and then we do our preprocessing (correction) from there.
            # We pretend that the data augmentations were actually pictures that we took.




            mask_path = osp.join(self.filepath,'Masks')
            file_name_no_suffix = self.list_files[idx]
            mask = self.get_mask(mask_path, file_name_no_suffix) # is of type Image.Image
            mask = mask.resize(preprocessing_resize_size, Image.NEAREST)


            img = smart_conversion(img, 'Image', 'uint8')
            mask = smart_conversion(mask, 'Image', 'uint8')


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])



            show_testrun = False

            # Testing
            if show_testrun:

                img_test = img
                mask_test = mask
                

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Original image")


                #print_time_since_last_call()
                img_test, mask_test = random_horizontal_flip(img_test, mask_test, prob=1.0)
                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Horizontal flip")

                ksize = (img_test.size[0]//20)
                ksize = ksize+1 if ksize % 2 == 0 else ksize
                #print_time_since_last_call()
                img_test = gaussian_blur(img_test, possible_sigma_vals_list=np.linspace(1, 10, 50), ker_size=ksize, prob=1.0)
                
                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Gaussian blur")



                if True:
                    img = img_test
                    mask = mask_test





            # Conducting pre-normalization data augmentation with random probabilities:
            
            if self.split == 'train':

                img, mask = random_horizontal_flip(img, mask, prob=0.5)
                
                ksize = (img.size[0]//20)
                ksize = (ksize + 1) if ksize % 2 == 0 else ksize
                img = gaussian_blur(img, possible_sigma_vals_list=np.linspace(1, 10, 50), ker_size=ksize, prob=0.2)
                
                

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])






            """
            These steps seem to work more for grayscale imgs.
            For RGB imgs, they really make the image look bad.

            Normalization,
            gamma correction,
            CLAHE"""



            if False:
                # Do the normalization transform:

                img = smart_conversion(img, 'tensor', 'float32')

                img = self.transform(img)



            """
            - Gamma correction by a factor of 0.8
            - CLAHE
            """


            if False:
                # Do gamma correction and clahe:

                img = smart_conversion(img, 'ndarray', 'uint8')
                mask = smart_conversion(mask, 'ndarray', 'uint8')
                # We convert RGB to LAB (L = lightness, A = green-red, B = blue-yellow) to do gamma correction and clahe

                lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])
            
                # Split the LAB image to different channels
                l, a, b = cv2.split(lab)

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])


                #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
                # Gamma correction by a factor of 0.8
                # This makes light regions darker to improve contrast
                table = 255.0*(np.linspace(0, 1, 256)**0.8)
                gammad_l = cv2.LUT(l, table)

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

                #local Contrast limited adaptive histogram equalization algorithm
                # I think for normalizing for brightness.
                # It works on grayscale, because just one channel which is brightness.

                # gammad_l isfloat64. clahe needs uint8 or uint16.
                gammad_l = gammad_l.astype(np.uint8)

                if True:
                    cl = self.clahe.apply(gammad_l)
                else:
                    cl = gammad_l

                limg = cv2.merge((cl, a, b))

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

                # converting back to RGB     
                img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([(img, "Gamma corr and possibly clahe"), mask])

            




            # mask = Image.fromarray(mask)



            img = smart_conversion(img, 'ndarray', 'uint8')
            mask = smart_conversion(mask, 'ndarray', 'uint8')

            # Testing
            if show_testrun:

                img_test = img
                mask_test = mask
                


                #print_time_since_last_call()
                img_test, mask_test = translation(img_test, mask_test, v_max_perc=0.4, h_max_perc=0.4, trans_type="crop", prob=1.0)
                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Translation")
                
                #print_time_since_last_call()
                img_test, mask_test = random_rotation(img_test, mask_test, max_angle=15, prob=1.0)
                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Rotation")

                # Should be at the end, because if trans or rot happened, they introduced black pixels at edges.
                # So if we scale in the zooming direction, we will redce the number of those black pixels.
                #print_time_since_last_call()
                img_test, mask_test = scale(img_test, mask_test, max_scale_percent=0.4, prob=1.0)
                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Scaling")
                

                if True:
                    img = img_test
                    mask = mask_test





            # Conducting POST-normalization data augmentation:
            
            if self.split == 'train':

                img, mask = translation(img, mask, v_max_perc=0.05, h_max_perc=0.05, prob=0.2)

                img, mask = random_rotation(img, mask, max_angle=15, prob=0.2)

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

                img, mask = scale(img, mask, max_scale_percent=0.1, prob=0.2)

                # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])





            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])






            # Converting img and mask to correct dimensions, binarization, types, ans removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')
            mask = smart_conversion(mask, 'ndarray', 'uint8')

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.output_width, self.output_height), interpolation=cv2.INTER_NEAREST)
            
            # Making the mask binary, as it is meant to be.
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask[mask < 127] = 0
            mask[mask >= 127] = 1

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])

            # Conversion to standard types that pytorch can work with.
            # target tensor is long, because it is a classification task (in regression it would also be float32)
            img = smart_conversion(img, "tensor", "float32") # converts to float32
            mask = smart_conversion(mask, 'tensor', "uint8").long() # converts to int64

            # mask mustn't have channels. It is a target, not an image.
            # And since the output of our network is (batch_size, n_classes, height, width), our target has to be (batch_size, height, width).
            # So here we need to return (height, width) mask, not (height, width, 1) mask.
            mask = mask.squeeze() # This function removes all dimensions of size 1 from the tensor


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])




            # while True:
            #     plt.pause(0.1)



            return img, mask
        
        except Exception as e:
            py_log.log_stack(MY_LOGGER)
            raise e


if __name__ == "__main__":
    
    ds = IrisDataset('Semantic_Segmentation_Dataset',split='train',transform=transform)
#    for i in range(1000):
    img, mask, idx,x,y= ds[0]
    plt.subplot(121)
    plt.imshow(np.array(mask))
    plt.subplot(122)
    plt.imshow(np.array(img)[0,:,:],cmap='gray')



# %%
