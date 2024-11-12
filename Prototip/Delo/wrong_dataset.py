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



import numpy as np
import torch
from torch.utils.data import Dataset 
import os
# import random
from PIL import Image
from torchvision import transforms
import cv2

import os.path as osp
# from utils import one_hot2dist
import copy

import random
np.random.seed(7)

import time


def get_random_number():
    #print('random called')
    random_number = np.random.random()
    #print(random_number)
    return random_number

def get_numpy_randint(a, b):
    #print('randint called')
    random_number = np.random.randint(a, b)
    return random_number

def get_numpy_rand(a):
    return np.random.rand(a)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5]) #TODO ali dam to ven? ker v članku ne najdem tega
    ])

#%%
class RandomHorizontalFlip(object):
    def __call__(self, img,mask, prob=.5):
        if get_random_number() < prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT),\
                        mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img,mask

class RandomRotation(object):
    def __call__(self, img,mask, prob=.2, max_angle=15):
        if get_random_number() < prob:
            angle = np.random.uniform(-max_angle, max_angle)
            return img.rotate(angle), mask.rotate(angle)
        return img,mask

class Starburst_augment(object):
    ## We have generated the starburst pattern from a train image 000000240768.png
    ## Please follow the file Starburst_generation_from_train_image_000000240768.pdf attached in the folder 
    ## This procedure is used in order to handle people with multiple reflections for glasses
    ## a random translation of mask of starburst pattern
    def __call__(self, img):
        x=get_numpy_randint(1, 40)
        y=get_numpy_randint(1, 40)
        mode = get_numpy_randint(0, 2)
        starburst=Image.open('starburst_black.png').convert("L")
        if mode == 0:
            starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
            starburst = starburst[:, :-x]
        if mode == 1:
            starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
            starburst = starburst[:, x:]

        img[92+y:549+y,0:400]=np.array(img)[92+y:549+y,0:400]*((255-np.array(starburst))/255)+np.array(starburst)
        return Image.fromarray(img)

def getRandomLine(xc, yc, theta):
    x1 = xc - 50*get_numpy_rand(1)*(1 if get_numpy_rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150*get_numpy_rand(1) + 50)*(1 if get_numpy_rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    return x1, y1, x2, y2

class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value=get_numpy_randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img,(7,7),sigma_value))

class Translation(object):
    def __call__(self, base,mask):
        factor_h = 2*get_numpy_randint(1, 20)
        factor_v = 2*get_numpy_randint(1, 20)
        mode = get_numpy_randint(0, 4)
        #print (mode,factor_h,factor_v,base.shape,mask.shape)
        if mode == 0:
            aug_base = np.pad(base, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_base = aug_base[:-factor_v, :]
            aug_mask = aug_mask[:-factor_v, :]
        if mode == 1:
            aug_base = np.pad(base, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_base = aug_base[factor_v:, :]
            aug_mask = aug_mask[factor_v:, :]
        if mode == 2:
            #print(base)
            aug_base = np.pad(base, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            #print(aug_base)
            #print('*******************************************************')
            #print(mask)
            aug_mask = np.pad(mask, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_base = aug_base[:, :-factor_h]
            aug_mask = aug_mask[:, :-factor_h]
        if mode == 3:
            aug_base = np.pad(base, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_base = aug_base[:, factor_h:]
            aug_mask = aug_mask[:, factor_h:]
        return Image.fromarray(aug_base), Image.fromarray(aug_mask)     

class Line_augment(object):
    def __call__(self, base):
        yc, xc = (0.3 + 0.4*get_numpy_rand(1))*base.shape
        aug_base = copy.deepcopy(base)
        num_lines = get_numpy_randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*get_numpy_rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta)

            # print("aug_base")
            # print(aug_base)
            # print("x1, y1, x2, y2")
            # print(x1, y1, x2, y2)
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       













class IrisDataset(Dataset):
    def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, **kwargs):
        
        self.transform = transform
        self.filepath= osp.join(filepath, split)
        
        self.width = kwargs['width']
        self.height = kwargs['height']
        
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
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

        #summary
        print('summary for ' + str(split))
        print('valid images: ' + str(len(self.list_files)))

    def __len__(self):
        if self.testrun:
            return 30
        return len(self.list_files)
    
    def mask_exists(self, mask_folder_path, file_name_no_suffix):
        mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
        return osp.exists(mask_filename)

    def get_mask(self, mask_folder_path, file_name_no_suffix):

        mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
        try:
            mask = np.array(Image.open(mask_filename).convert("L").resize((self.width, self.height), Image.NEAREST)).astype(np.uint8)
            mask[mask <= 127] = 0
            mask[mask > 127] = 1
            return mask

        except FileNotFoundError:
            print('file not found: ' + str(mask_filename))
            self.images_without_mask.append(file_name_no_suffix)
            return None


    def __getitem__(self, idx):

        image_path = osp.join(self.filepath,'Images',self.list_files[idx]+'.jpg')

        pil_img = Image.open(image_path).convert("L").resize((self.width, self.height), Image.BILINEAR) #ali Image.BICUBIC ali Image.LANCZOS

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
        #Fixed gamma value for      
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
        pil_img = cv2.LUT(np.array(pil_img), table)

        #print('pil_img size: ' + str(pil_img.shape))



        mask_path = osp.join(self.filepath,'Masks')
        file_name_no_suffix = self.list_files[idx]
        if self.split != 'test':
            mask = self.get_mask(mask_path, file_name_no_suffix)
            #print('1unique masks: ' + str(np.unique(mask)))
            #print("mask: size="+str(np.array(mask).shape))
            #mask = cv2.LUT(mask, table) # mask already np.array



        # Conducting data augmentation with random probabilities:
        if self.transform is not None:
            if self.split == 'train':
                # if get_random_number() < 0.2:
                #     #pil_img = Starburst_augment()(np.array(pil_img))
                #     pass
                # if get_random_number() < 0.2:
                #     pil_img = Line_augment()(np.array(pil_img))
                if get_random_number() < 0.2:
                    pil_img = Gaussian_blur()(np.array(pil_img))
                if get_random_number() < 0.4:
                    pil_img, mask = Translation()(np.array(pil_img),np.array(mask))



        # Zakaj to ne velja tudi za train?
        if self.split == 'train' or self.split == 'validation':
            mask = np.array(np.uint8(mask))  # mogoce uporabi drugo spremenljivko
            mask = Image.fromarray(mask)


        img = self.clahe.apply(np.array(np.uint8(pil_img)))
        img = Image.fromarray(img)
        # pil_img_test = Image.fromarray(pil_img)
        #img.show()
        #input()



        # Aditional data augmentation:
        if self.transform is not None:
            if self.split == 'train':
                img, mask = RandomHorizontalFlip()(img, mask)
                img, mask = RandomRotation()(img, mask)
                # img.show()
                # input()
                # mask.show()
                # input()

            img = self.transform(img)
            # tensor to pil image
            # img_test = transforms.ToPILImage()(img).convert("L")
            # img_test.show()
            # input()



        # This was probably for change of weights in multiclass segmentation.
        # This can be red in the famous Unet paper.
        # Pixels which are near to 2 objects (e.g. are a border of two cells)
        # are given more importance because they are hard to learn but very important,
        # and so we want to prioritize them.
        """
        if self.split != 'test':
            ## This is for boundary aware cross entropy calculation
            spatialWeights = cv2.Canny(np.array(mask),0,3)/255
            spatialWeights = cv2.dilate(spatialWeights,(3,3),iterations = 1)*20

            ##This is the implementation for the surface loss
            # Distance map for each class
            distMap = []
            for i in range(0, self.classes):
                distMap.append(one_hot2dist(np.array(mask)==i))
            distMap = np.stack(distMap, 0)
        """






        # I have no idea why it's written like this, but let it be for now.

        if self.split == 'test':
            mask = self.get_mask(osp.join(self.filepath, 'Masks'), self.list_files[idx])
            ##since mask, spatialWeights and distMap is not needed for test images
            mask = MaskToTensor()(mask)
            mask = mask.view(self.height, self.width)
            return img, mask    #,self.list_files[idx]   #,0,0



        mask = MaskToTensor()(mask)
        mask = mask.view(self.height, self.width) # 1x640x400 => 640x400

        return img, mask #, self.list_files[idx]   #, spatialWeights, np.float32(distMap)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = IrisDataset('Semantic_Segmentation_Dataset',split='train',transform=transform)
#    for i in range(1000):
    img, mask, idx,x,y= ds[0]
    plt.subplot(121)
    plt.imshow(np.array(mask))
    plt.subplot(122)
    plt.imshow(np.array(img)[0,:,:],cmap='gray')



# %%
