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



from helper_img_and_fig_tools import smart_conversion, show_image

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






class IrisDataset(Dataset):
    def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, clipLimit=1.5, **kwargs):
        
        self.transform = transform
        self.filepath= osp.join(filepath, split)
        
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

            mask_path = osp.join(self.filepath,'Masks')
            file_name_no_suffix = self.list_files[idx]
            mask = self.get_mask(mask_path, file_name_no_suffix) # is of type Image.Image
            



            # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')
            mask = smart_conversion(mask, 'ndarray', 'uint8')

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (self.output_width, self.output_height), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (self.output_width, self.output_height), interpolation=cv2.INTER_LANCZOS4)
            
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
            py_log_always_on.log_stack(MY_LOGGER)
            raise e


if __name__ == "__main__":
    
    ds = IrisDataset('Semantic_Segmentation_Dataset',split='train',transform=transform)
#    for i in range(1000):
    img, mask, idx,x,y= ds[0]
    plt.subplot(121)
    plt.imshow(np.array(mask))
    plt.subplot(122)
    plt.imshow(np.array(img)[0,:,:],cmap='gray')

