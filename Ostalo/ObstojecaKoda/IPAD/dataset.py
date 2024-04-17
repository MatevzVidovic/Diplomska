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
5) Translation of image and labels in any direction with random factor less than 20.
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
from utils import one_hot2dist
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
    def __call__(self, img,label, prob=.5):
        if get_random_number() < prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT),\
                        label.transpose(Image.FLIP_LEFT_RIGHT)
        return img,label

class RandomRotation(object):
    def __call__(self, img,label, prob=.2, max_angle=15):
        if get_random_number() < prob:
            angle = np.random.uniform(-max_angle, max_angle)
            return img.rotate(angle), label.rotate(angle)
        return img,label

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
            aug_base = cv2.line(aug_base, (x1, y1), (x2, y2), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       


class IrisDataset(Dataset):
    def __init__(self, filepath, split='train',transform=None,n_classes=4,**args):
        self.transform = transform
        self.filepath= osp.join(filepath,split)
        self.width = args.get('width')
        self.height = args.get('height')
        self.split = split
        self.classes = n_classes
        self.images_with_partial_labels = [] # seznam slik ki imajo vse tri fajle ampak iz enega fajla nismo prebrali nobenega piksla!
        self.images_without_all_labels = [] # seznam slik, ki sploh nimajo vseh treh fajlov

        listall = []
        all_files = os.listdir(osp.join(self.filepath,'images'))
        labels_folder = osp.join(self.filepath,'labels')
        all_files.sort()
        #print('searching labels in ' + str(labels_folder))
        for file in all_files:
            #print(file)
            if file.endswith(".jpg"): # to je za train file (vsi train fajli so jpg - mobius in sbvpi)
                file_without_suffix = file.strip(".jpg")
                # TODO poglej ce ima vse tri anotacije!!
                label = self.get_label(labels_folder, file_without_suffix)
                if label is not None: # means that we found all three masks
                    listall.append(file_without_suffix)

        self.list_files=listall
        self.testrun = args.get('testrun')

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

        #summary
        print('summary for ' + str(split))
        print('valid images: ' + str(len(self.list_files)))
        print('images_with_partial_labels (not all three but all three files are present): ' + str(len(self.images_with_partial_labels)))
        print('images_without_all_labels: ' + str(len(self.images_without_all_labels)))

    def __len__(self):
        if self.testrun:
            return 10
        return len(self.list_files)

    def get_label(self, label_path, train_filename):
        if self.classes == 4:
            merged_label = np.zeros((self.height, self.width), dtype=np.uint8)
            pixel_values = [1, 2, 3]
            try:
                label_sclera_filename = osp.join(label_path, train_filename + '_sclera.png')
                label_iris_filename = osp.join(label_path, train_filename + '_iris.png')
                label_pupil_filename = osp.join(label_path, train_filename + '_pupil.png')
                for (labelpath, pixel_value) in zip([label_sclera_filename, label_iris_filename, label_pupil_filename], pixel_values):
                    label = Image.open(labelpath).convert("L").resize((self.width, self.height), Image.NEAREST)
                    label = np.array(label)
                    #if len(np.unique(label)) != 2:
                    #    print(labelpath)
                    #    print(np.unique(label))
                    label[label <= 127] = 0
                    label[label > 127] = pixel_value
                    label.astype(np.uint8)
                    merged_label = merged_label + label
            except FileNotFoundError:
                #print('file not found: ' + str(label_sclera_filename))
                self.images_without_all_labels.append(train_filename)
                return None

            # add zero
            pixel_values.append(0)
            #print(np.unique(merged_label))
            #print(np.array(pixel_values))
            if not np.array_equal(np.sort(np.unique(merged_label)), np.sort(np.array(pixel_values))):
                print(train_filename)
                self.images_with_partial_labels.append(train_filename)
                return None

            return merged_label

        elif self.classes <= 2:
            label_filename = osp.join(label_path, train_filename + '.png')
            try:
                label = np.array(Image.open(label_filename).convert("L").resize((self.width, self.height), Image.NEAREST)).astype(np.uint8)
                label[label <= 127] = 0
                label[label > 127] = 1
                return label
            except FileNotFoundError:
                print('file not found: ' + str(label_filename))
                self.images_without_all_labels.append(train_filename)
                return None

        else:
            raise ValueError(f"Cannot process masks with n_classes {self.classes}")

    def __getitem__(self, idx):
        imagepath = osp.join(self.filepath,'images',self.list_files[idx]+'.jpg')
        #print(imagepath)
        pilimg = Image.open(imagepath).convert("L").resize((self.width, self.height), Image.BILINEAR) #ali Image.BICUBIC ali Image.LANCZOS

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
        #Fixed gamma value for      
        table = 255.0*(np.linspace(0, 1, 256)**0.8)
        pilimg = cv2.LUT(np.array(pilimg), table)

        #print('pilimg size: ' + str(pilimg.shape))

        if self.split != 'test':
            label = self.get_label(osp.join(self.filepath, 'labels'), self.list_files[idx])
            #print('1unique labels: ' + str(np.unique(label)))
            #print("LABEL: size="+str(np.array(label).shape))
            #label = cv2.LUT(label, table) # label already np.array


        if self.transform is not None:
            if self.split == 'train':
                if get_random_number() < 0.2:
                    #pilimg = Starburst_augment()(np.array(pilimg))
                    pass
                if get_random_number() < 0.2:
                    pilimg = Line_augment()(np.array(pilimg))
                if get_random_number() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))
                if get_random_number() < 0.4:
                    pilimg, label = Translation()(np.array(pilimg),np.array(label))

        if self.split == 'train' or self.split == 'validation':
            label = np.array(np.uint8(label))  # mogoce uporabi drugo spremenljivko
            label = Image.fromarray(label)

        img = self.clahe.apply(np.array(np.uint8(pilimg)))
        img = Image.fromarray(img)
        # pilimg_test = Image.fromarray(pilimg)
        #img.show()
        #input()

        if self.transform is not None:
            if self.split == 'train':
                img, label = RandomHorizontalFlip()(img,label)
                img, label = RandomRotation()(img,label)
                # img.show()
                # input()
                # label.show()
                # input()

            img = self.transform(img)
            # tensor to pil image
            # img_test = transforms.ToPILImage()(img).convert("L")
            # img_test.show()
            # input()

        if self.split != 'test':
            ## This is for boundary aware cross entropy calculation
            spatialWeights = cv2.Canny(np.array(label),0,3)/255
            spatialWeights=cv2.dilate(spatialWeights,(3,3),iterations = 1)*20

            ##This is the implementation for the surface loss
            # Distance map for each class
            distMap = []
            for i in range(0, self.classes):
                distMap.append(one_hot2dist(np.array(label)==i))
            distMap = np.stack(distMap, 0)

        if self.split == 'test':
            label = self.get_label(osp.join(self.filepath, 'labels'), self.list_files[idx])
            ##since label, spatialWeights and distMap is not needed for test images
            label = MaskToTensor()(label)
            label = label.view(self.height, self.width)
            return img,label,self.list_files[idx],0,0

        label = MaskToTensor()(label)
        label = label.view(self.height, self.width) # 1x640x400 => 640x400
        return img, label, self.list_files[idx],spatialWeights,np.float32(distMap)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds = IrisDataset('Semantic_Segmentation_Dataset',split='train',transform=transform)
#    for i in range(1000):
    img, label, idx,x,y= ds[0]
    plt.subplot(121)
    plt.imshow(np.array(label))
    plt.subplot(122)
    plt.imshow(np.array(img)[0,:,:],cmap='gray')



# %%
