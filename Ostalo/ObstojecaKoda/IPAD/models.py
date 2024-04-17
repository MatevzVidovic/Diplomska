#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:50:11 2019

@author: manoj
"""


from typing import no_type_check_decorator
from densenet import DenseNet2D, DenseNet2D_original, DenseNet2DLayer135_compressed_down_blocks12_up_blocks1
# referencni modeli
from densenet import DenseNet2D_216k, DenseNet2D_188k, DenseNet2D_139k
from densenet import DenseNet2D_216k_alternative
from densenet import DenseNet2D_188k_alternative
from densenet import DenseNet2D_139k_alternative


model_dict = {}


# TEST / VISUALIZE_MASKS
model_dict['densenet'] = DenseNet2D(out_channels=2, dropout=True, prob=0.2, downsize_activations=True,activation_height=40, activation_width=25,activations_weighting_power=None)
model_dict['densenet4'] = DenseNet2D(out_channels=4, dropout=True, prob=0.2, downsize_activations=True,activation_height=40, activation_width=25,activations_weighting_power=None)
#model_dict['densenet'] = DenseNet2D_original(dropout=True, prob=0.2)
#model_dict['densenet'] = DenseNet2DLayer135_compressed_down_blocks12_up_blocks1(dropout=True, prob=0.2)
#model_dict['densenet'] = DenseNet2D_139k_alternative(dropout=True, prob=0.2)


# TRAIN WITH KNOWLEDGE DISTILLATION
#model_dict['teacher'] = DenseNet2D_original(dropout=True, prob=0.2)
#model_dict['student'] = DenseNet2DLayer135_compressed_down_blocks12_up_blocks1(dropout=True, prob=0.2)


# TRAIN WITH PRUNING? ALI TRAIN PRUNED MODEL
model_dict['teacher-densenet'] = DenseNet2D_original(out_channels=2, dropout=True, prob=0.2)  # FOR PRUNING DESTILATION
model_dict['teacher-densenet4'] = DenseNet2D_original(out_channels=4, dropout=True, prob=0.2)  # FOR PRUNING DESTILATION
model_dict['student-densenet'] = DenseNet2D(out_channels=2, dropout=True, prob=0.2, downsize_activations=True,activation_height=40, activation_width=25,activations_weighting_power=None)
model_dict['student-densenet4'] = DenseNet2D(out_channels=4, dropout=True, prob=0.2, downsize_activations=True,activation_height=40, activation_width=25,activations_weighting_power=None)

# reference models
#model_dict['student'] = DenseNet2D_139k_alternative(dropout=True, prob=0.2)
#model_dict['student'] = DenseNet2D_188k(dropout=True, prob=0.2)
#model_dict['student'] = DenseNet2D_139k(dropout=True, prob=0.2)

#model_dict['tmp3'] = DenseNet2D_139k_alternative(dropout=True, prob=0.2)


# U-Net
#from unet import UNet  # Original
from unet_prunable import UNet  # Prunable (with saved activations and sums)
model_dict['unet'] = UNet(n_classes=2, pretrained=True)
model_dict['unet4'] = UNet(n_classes=4, pretrained=True)

model_dict['teacher-unet'] = UNet(n_classes=2, pretrained=False)
model_dict['teacher-unet4'] = UNet(n_classes=4, pretrained=False)
model_dict['student-unet'] = UNet(n_classes=2, pretrained=False)
model_dict['student-unet4'] = UNet(n_classes=4, pretrained=False)
