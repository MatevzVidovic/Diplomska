#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:20:33 2019

@author: Shusil Dangi

References:
    https://github.com/ShusilDangi/DenseUNet-K
It is a simplied version of DenseNet with U-NET architecture.
2D implementation
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


# original
class DenseNet2D_down_block(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=(1, 1),
                                padding=(0, 0))
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
            last_fsp_layer = out
        else:
            x1 = self.relu(self.conv1(x))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
            last_fsp_layer = out
        return self.bn(out), first_fsp_layer, last_fsp_layer


# without 1st and 2nd CONV
class DenseNet2D_down_block_without12Conv(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block_without12Conv, self).__init__()
        # self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv31 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1),
                                padding=(0, 0))
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        # self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            # x1 = self.relu(self.dropout1(self.conv1(x)))
            # first_fsp_layer = x1
            # x21 = torch.cat((x, x1), dim=1)
            x21 = x
            x22 = self.relu(self.dropout2(self.conv22(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
            # last_fsp_layer = out
        else:
            # x1 = self.relu(self.conv1(x))
            # first_fsp_layer = x1
            # x21 = torch.cat((x, x1), dim=1)
            x21 = x
            x22 = self.relu(self.conv22(x21))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
            # last_fsp_layer = out
        return self.bn(out)  # , first_fsp_layer, last_fsp_layer


# without 3rd CONV
class DenseNet2D_down_block_without3Conv(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block_without3Conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        # self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=(1, 1),
                                padding=(0, 0))
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
            # last_fsp_layer = out
        else:
            x1 = self.relu(self.conv1(x))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv21(x21))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
            # last_fsp_layer = out
        return self.bn(out)  # , first_fsp_layer, last_fsp_layer


# without 2nd and 3rd CONV === without 4th and 5th CONV
class DenseNet2D_down_block_without23Conv(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block_without23Conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=(1, 1),
        #                        padding=(0, 0))
        # self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        # self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            # x31 = torch.cat((x21, x22), dim=1)
            # out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
            # last_fsp_layer = out
        else:
            x1 = self.relu(self.conv1(x))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
            # x31 = torch.cat((x21, x22), dim=1)
            # out = self.relu(self.conv32(self.conv31(x31)))
            # last_fsp_layer = out
        return self.bn(out)  # , first_fsp_layer, last_fsp_layer


# without 5th CONV
class DenseNet2D_down_block_without5Conv(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block_without5Conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=(1, 1),
                                padding=(0, 0))
        # self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv31(x31)))
            # last_fsp_layer = out
        else:
            x1 = self.relu(self.conv1(x))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv22(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv31(x31))
            # last_fsp_layer = out
        return self.bn(out)  # , first_fsp_layer, last_fsp_layer


# original
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,
                                kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            last_fsp_layer = out
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
            last_fsp_layer = out
        return out, first_fsp_layer, last_fsp_layer


# without 2nd CONV
class DenseNet2D_up_block_concat_without2Conv(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_without2Conv, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        # self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,
                                kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv11(x)))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            # last_fsp_layer = out
        else:
            x1 = self.relu(self.conv11(x))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
            # last_fsp_layer = out
        return out  # , first_fsp_layer, last_fsp_layer


# without 1st and 2nd CONV === without 3rd and 4th CONV
class DenseNet2D_up_block_concat_without12Conv(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_without12Conv, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
        #                        kernel_size=(1,1),padding=(0,0))
        # self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        # self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            out = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            # first_fsp_layer = x1
            # x21 = torch.cat((x,x1),dim=1)
            # out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            # last_fsp_layer = out
        else:
            out = self.relu(self.conv12(self.conv11(x)))
            # first_fsp_layer = x1
            # x21 = torch.cat((x,x1),dim=1)
            # out = self.relu(self.conv22(self.conv21(x21)))
            # last_fsp_layer = out
        return out  # , first_fsp_layer, last_fsp_layer


# without 4th CONV
class DenseNet2D_up_block_concat_without4Conv(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_without4Conv, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,
                                kernel_size=(1, 1), padding=(0, 0))
        # self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv21(x21)))
            # last_fsp_layer = out
        else:
            x1 = self.relu(self.conv12(self.conv11(x)))
            # first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv21(x21))
            # last_fsp_layer = out
        return out  # , first_fsp_layer, last_fsp_layer


# todo: mogocep probaj samo z drugo konvolucijo (brez prve sicer malo povecas parametre ampak a jih tok?? --> naslednja
# without 1st, 2nd and 4th CONV = just conv that reduces parameters
class DenseNet2D_up_block_concat_without124Conv(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_without124Conv, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        # self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
        #                        kernel_size=(1,1),padding=(0,0))
        # self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        # self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            out = self.relu(self.dropout1(self.conv11(x)))
            # first_fsp_layer = x1
            # x21 = torch.cat((x,x1),dim=1)
            # out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            # last_fsp_layer = out
        else:
            out = self.relu(self.conv11(x))
            # first_fsp_layer = x1
            # x21 = torch.cat((x,x1),dim=1)
            # out = self.relu(self.conv22(self.conv21(x21)))
            # last_fsp_layer = out
        return out  # , first_fsp_layer, last_fsp_layer


# without 1st, 2nd and 3rd CONV = just regular conv
class DenseNet2D_up_block_concat_just_one_regular_conv(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_just_one_regular_conv, self).__init__()
        # self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        # self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        # self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
        #                        kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        # self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            # out = self.relu(self.dropout1(self.conv11(x)))
            # first_fsp_layer = x1
            # x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.dropout2(self.conv22(x)))
            # last_fsp_layer = out
        else:
            # out = self.relu(self.conv11(x))
            # first_fsp_layer = x1
            # x21 = torch.cat((x,x1),dim=1)
            out = self.relu(self.conv22(x))
            # last_fsp_layer = out
        return out  # , first_fsp_layer, last_fsp_layer


# original
class DenseNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4, self.x5)
        self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3, self.x6)
        self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        # 2. definicija s sum
        self.att_map_x1_sum, self.att_map_x2_sum, self.att_map_x3_sum, self.att_map_x4_sum, self.att_map_x5_sum, \
        self.att_map_x6_sum, self.att_map_x7_sum, self.att_map_x8_sum, self.att_map_x9_sum = \
            [g.mean(1) for g in (self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8, self.x9)]

        # 3. definicija z max
        # print(self.x1.shape) # [4, 32, 640, 400]
        self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x7_max, self.att_map_x9_max = [
            g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x7, self.x9)]
        return out


# remove from decoder 2nd Conv in each (up)block
class DenseNet2D_without2Conv_in_each_upblock(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D_without2Conv_in_each_upblock, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_without2Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_without2Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_without2Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_without2Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


# remove from decoder 12Conv in each (up)block
class DenseNet2D_without12Conv_in_each_upblock(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D_without12Conv_in_each_upblock, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_without12Conv(skip_channels=channel_size,
                                                                  input_channels=channel_size,
                                                                  output_channels=channel_size, up_stride=(2, 2),
                                                                  dropout=dropout,
                                                                  prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_without12Conv(skip_channels=channel_size,
                                                                  input_channels=channel_size,
                                                                  output_channels=channel_size, up_stride=(2, 2),
                                                                  dropout=dropout,
                                                                  prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_without12Conv(skip_channels=channel_size,
                                                                  input_channels=channel_size,
                                                                  output_channels=channel_size, up_stride=(2, 2),
                                                                  dropout=dropout,
                                                                  prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_without12Conv(skip_channels=channel_size,
                                                                  input_channels=channel_size,
                                                                  output_channels=channel_size, up_stride=(2, 2),
                                                                  dropout=dropout,
                                                                  prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


# removed from decoder 4COnv in each (up)block
class DenseNet2D_without4Conv_in_each_upblock(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D_without4Conv_in_each_upblock, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_without4Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_without4Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_without4Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_without4Conv(skip_channels=channel_size,
                                                                 input_channels=channel_size,
                                                                 output_channels=channel_size, up_stride=(2, 2),
                                                                 dropout=dropout,
                                                                 prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


# remove from decoder 124Conv in each (up)block
# todo nevem ce je smiselno da imas v dekoderju samo konvolucijo ki zmanjsastevilo parametrov -> IDEJA: PROBAJ VSAKO DRUGO DAT NAVADNO KONVOLUCIJO -- naslednja
class DenseNet2D_without124Conv_in_each_upblock(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D_without124Conv_in_each_upblock, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_without124Conv(skip_channels=channel_size,
                                                                   input_channels=channel_size,
                                                                   output_channels=channel_size, up_stride=(2, 2),
                                                                   dropout=dropout,
                                                                   prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_without124Conv(skip_channels=channel_size,
                                                                   input_channels=channel_size,
                                                                   output_channels=channel_size, up_stride=(2, 2),
                                                                   dropout=dropout,
                                                                   prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_without124Conv(skip_channels=channel_size,
                                                                   input_channels=channel_size,
                                                                   output_channels=channel_size, up_stride=(2, 2),
                                                                   dropout=dropout,
                                                                   prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_without124Conv(skip_channels=channel_size,
                                                                   input_channels=channel_size,
                                                                   output_channels=channel_size, up_stride=(2, 2),
                                                                   dropout=dropout,
                                                                   prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


# todo nevem ce je smiselno da imas v dekoderju samo konvolucijo ki zmanjsastevilo parametrov -> IDEJA: PROBAJ VSAKO DRUGO DAT NAVADNO KONVOLUCIJO
class DenseNet2D_without124Conv_and_one_regular_interchanging_only_decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D_without124Conv_and_one_regular_interchanging_only_decoder, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_without124Conv(skip_channels=channel_size,
                                                                   input_channels=channel_size,
                                                                   output_channels=channel_size, up_stride=(2, 2),
                                                                   dropout=dropout,
                                                                   prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_just_one_regular_conv(skip_channels=channel_size,
                                                                          input_channels=channel_size,
                                                                          output_channels=channel_size,
                                                                          up_stride=(2, 2), dropout=dropout,
                                                                          prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_without124Conv(skip_channels=channel_size,
                                                                   input_channels=channel_size,
                                                                   output_channels=channel_size, up_stride=(2, 2),
                                                                   dropout=dropout,
                                                                   prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_just_one_regular_conv(skip_channels=channel_size,
                                                                          input_channels=channel_size,
                                                                          output_channels=channel_size,
                                                                          up_stride=(2, 2), dropout=dropout,
                                                                          prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


class DenseNet2D_down_block12(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block12, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
            last_fsp_layer = out
            # x31 = torch.cat((x21, x22), dim=1)
            # out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
            last_fsp_layer = out
            # x31 = torch.cat((x21, x22), dim=1)
            # out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out), first_fsp_layer, last_fsp_layer


"""
class DenseNet2D_down_block13(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block13, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

        self.conv31 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(1, 1),
                                padding=(0, 0))
        self.conv32 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x31 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            x31 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out)



# remove two 1x1 conv in whole block
class DenseNet2D_down_block_1Conv(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block_1Conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.conv21 = nn.Conv2d(input_channels + output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

        self.conv31 = nn.Conv2d(input_channels + 2 * output_channels, output_channels, kernel_size=(3, 3),
                                padding=(1, 1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv1(x)))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.dropout2(self.conv21(x21)))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.dropout3(self.conv31(x31)))
        else:
            x1 = self.relu(self.conv1(x))
            x21 = torch.cat((x, x1), dim=1)
            x22 = self.relu(self.conv21(x21))
            x31 = torch.cat((x21, x22), dim=1)
            out = self.relu(self.conv31(x31))
        return self.bn(out)

"""


# up block brez 1x1 konvolucij, ki zmanjsajo stevilo parametrov
class DenseNet2D_up_block_concat_1Conv(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_1Conv, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))

        self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,
                                kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            x1 = self.relu(self.dropout1(self.conv11(x)))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.dropout2(self.conv21(x21)))

        else:
            x1 = self.relu(self.conv11(x))
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv21(x21))
        return out


class DenseNet2D_up_block_concat1(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat1, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        # self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,
        #                       kernel_size=(1, 1), padding=(0, 0))
        # self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

    def forward(self, prev_feature_map, x, residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x, scale_factor=self.up_stride, mode='nearest')

        x = torch.cat((x, prev_feature_map), dim=1)
        if self.dropout:
            out = self.relu(self.dropout1(self.conv12(self.conv11(x))))
            first_fsp_layer = out
            # x21 = torch.cat((x, x1), dim=1)
            # out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            out = self.relu(self.conv12(self.conv11(x)))
            first_fsp_layer = out
            # x21 = torch.cat((x, x1), dim=1)
            # out = self.relu(self.conv22(self.conv21(x21)))
        return out, first_fsp_layer


"""
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

"""

"""
# po zgledu wrn, povecas stevilo featurjev
class DenseNet2DWRN(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2DWRN, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=32, down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=32, output_channels=160, down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=160, output_channels=320, down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block(input_channels=320, output_channels=640, down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=640, output_channels=640, down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=640, input_channels=640, output_channels=320, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=320, input_channels=320, output_channels=160, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=160, input_channels=160, output_channels=32, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=32, input_channels=32, output_channels=32, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4, self.x5)
        self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3, self.x6)
        self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        # 2. definicija s sum
        self.att_map_x1_sum, self.att_map_x3_sum, self.att_map_x5_sum, self.att_map_x7_sum, self.att_map_x9_sum = [
            g.pow(2).mean(1) for g in (self.x1, self.x3, self.x5, self.x7, self.x9)]

        # 3. definicija z max
        # print(self.x1.shape) # [4, 32, 640, 400]
        self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x7_max, self.att_map_x9_max = [
            g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x7, self.x9)]
        return out



# extend every down block and up block
class DenseNet2DExtended(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DExtended, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        #extended
        self.down_block1_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block2_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block3_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block4_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block1_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)


        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block2_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block3_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block4_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x1_1 = self.down_block1_1(self.x1)
        self.x2 = self.down_block2(self.x1_1)
        self.x2_1 = self.down_block2_1(self.x2)
        self.x3 = self.down_block3(self.x2_1)
        self.x3_1 = self.down_block3_1(self.x3)
        self.x4 = self.down_block4(self.x3_1)
        self.x4_1 = self.down_block4_1(self.x4)
        self.x5 = self.down_block5(self.x4_1)

        self.x6 = self.up_block1(self.x4_1, self.x5)
        self.x6_1 = self.up_block1_1(self.x4, self.x6)
        self.x7 = self.up_block2(self.x3_1, self.x6_1)
        self.x7_1 = self.up_block2_1(self.x3, self.x7)
        self.x8 = self.up_block3(self.x2_1, self.x7_1)
        self.x8_1 = self.up_block3_1(self.x2, self.x8)
        self.x9 = self.up_block4(self.x1_1, self.x8_1)
        self.x9_1 = self.up_block4_1(self.x1, self.x9)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9_1))
        else:
            out = self.out_conv1(self.x9_1)

        return out




# extend every down block and up block, add residual connections
#TODO namesto DenseNet2D_down_block dej DenseNet2D_down_block_1Conv -> poglej razliko v velikosti
#TODO namesto DenseNet2D_up_block_concat dej DenseNet2D_up_block_concat_1Conv
class DenseNet2DExtendedResidual(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DExtendedResidual, self).__init__()

        self.down_block1 = DenseNet2D_down_block_1Conv(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        #extended
        self.down_block1_1 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block2 = DenseNet2D_down_block_1Conv(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block2_1 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block3 = DenseNet2D_down_block_1Conv(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block3_1 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block_1Conv(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block4_1 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block_1Conv(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block1_1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)


        self.up_block2 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size*2,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block2_1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.up_block3 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size*2,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block3_1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.up_block4 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size*2,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block4_1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x1_1 = self.down_block1_1(self.x1)

        self.x2 = self.down_block2(self.x1_1, self.x1)
        self.x2_1 = self.down_block2_1(self.x2)
        self.x3 = self.down_block3(self.x2_1, self.x2)
        self.x3_1 = self.down_block3_1(self.x3)
        self.x4 = self.down_block4(self.x3_1, self.x3)
        self.x4_1 = self.down_block4_1(self.x4)
        self.x5 = self.down_block5(self.x4_1, self.x4)

        self.x6 = self.up_block1(self.x4_1, self.x5)
        self.x6_1 = self.up_block1_1(self.x4, self.x6)
        self.x7 = self.up_block2(self.x3_1, self.x6_1, self.x6)
        self.x7_1 = self.up_block2_1(self.x3, self.x7)
        self.x8 = self.up_block3(self.x2_1, self.x7_1, self.x7)
        self.x8_1 = self.up_block3_1(self.x2, self.x8)
        self.x9 = self.up_block4(self.x1_1, self.x8_1, self.x8)
        self.x9_1 = self.up_block4_1(self.x1, self.x9)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9_1))
        else:
            out = self.out_conv1(self.x9_1)

        return out




# extend every down block and up block, add residual connections, add atrous convolution at the middle
class DenseNet2DExtendedResidualAtrous(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DExtendedResidualAtrous, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        #extended
        self.down_block1_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block2_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block3_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block4_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size*2, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)


        # na sredini dodaj atrous convolution
        # dobiš ven 25*40 * 32
        # conv 1*1, rate=1
        # prvi argument (inplanes) je koliko channelov ima izhod iz prejšnjega (down block 5)
        # ALI BI MOGU BIT IZHOD VEČ CHANNELOV?
        self.aspp1 = _ASPPModule(channel_size, channel_size, 1, padding=0, dilation=1, BatchNorm=nn.BatchNorm2d) # Deeplab ima možnost synchronized batch norm
        # conv 3x3, rate 6
        self.aspp2 = _ASPPModule(channel_size, channel_size, 3, padding=6, dilation=6, BatchNorm=nn.BatchNorm2d) # Deeplab ima možnost synchronized batch norm
        #conv 3x3, rate 12
        self.aspp3 = _ASPPModule(channel_size, channel_size, 3, padding=12, dilation=12, BatchNorm=nn.BatchNorm2d) # Deeplab ima možnost synchronized batch norm
        #conv 3x3, rate 18
        self.aspp4 = _ASPPModule(channel_size, channel_size, 3, padding=18, dilation=18, BatchNorm=nn.BatchNorm2d) # Deeplab ima možnost synchronized batch norm

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(channel_size, channel_size, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(channel_size),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(5*channel_size, channel_size, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.relu = nn.ReLU()
        self.dropout_aspp = nn.Dropout(0.5)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block1_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)


        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size*2,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block2_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size*2,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block3_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size*2,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block4_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights() # ČE NE DELA, DEJ ASPP V SVOJ MODUL IN SVOJO INICIALIZACIJO

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x1_1 = self.down_block1_1(self.x1)

        self.x2 = self.down_block2(self.x1_1, self.x1)
        self.x2_1 = self.down_block2_1(self.x2)
        self.x3 = self.down_block3(self.x2_1, self.x2)
        self.x3_1 = self.down_block3_1(self.x3)
        self.x4 = self.down_block4(self.x3_1, self.x3)
        self.x4_1 = self.down_block4_1(self.x4)
        self.x5 = self.down_block5(self.x4_1, self.x4)

        self.x1_aspp = self.aspp1(self.x5)
        self.x2_aspp = self.aspp2(self.x5)
        self.x3_aspp = self.aspp3(self.x5)
        self.x4_aspp = self.aspp4(self.x5)
        self.x5_avg_pool = self.global_avg_pool(self.x5)
        self.x5_avg_pool = F.interpolate(self.x5_avg_pool, size=self.x4_aspp.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((self.x1_aspp, self.x2_aspp, self.x3_aspp, self.x4_aspp, self.x5_avg_pool), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout_aspp(x)

        self.x6 = self.up_block1(self.x4_1, x)
        self.x6_1 = self.up_block1_1(self.x4, self.x6)
        self.x7 = self.up_block2(self.x3_1, self.x6_1, self.x6)
        self.x7_1 = self.up_block2_1(self.x3, self.x7)
        self.x8 = self.up_block3(self.x2_1, self.x7_1, self.x7)
        self.x8_1 = self.up_block3_1(self.x2, self.x8)
        self.x9 = self.up_block4(self.x1_1, self.x8_1, self.x8)
        self.x9_1 = self.up_block4_1(self.x1, self.x9)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9_1))
        else:
            out = self.out_conv1(self.x9_1)

        return out


# extend only with 2 up blocks and 2 down blocks
class DenseNet2DExtended2(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DExtended2, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        #extended
        self.down_block1_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)


        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)


        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        # extended
        self.down_block4_1 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block1_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)


        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)


        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)


        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        # extended
        self.up_block4_1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=None, dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x1_1 = self.down_block1_1(self.x1)
        self.x2 = self.down_block2(self.x1_1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x4_1 = self.down_block4_1(self.x4)
        self.x5 = self.down_block5(self.x4_1)

        self.x6 = self.up_block1(self.x4_1, self.x5)
        self.x6_1 = self.up_block1_1(self.x4, self.x6)
        self.x7 = self.up_block2(self.x3, self.x6_1)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1_1, self.x8)
        self.x9_1 = self.up_block4_1(self.x1, self.x9)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9_1))
        else:
            out = self.out_conv1(self.x9_1)

        return out

# ******************************************************************************************************


class DenseNet2DLayer123(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DLayer123, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)


        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x5 = self.down_block5(self.x3)
        self.x6 = self.up_block1(self.x3, self.x5)
        self.x7 = self.up_block2(self.x2, self.x6)
        self.x8 = self.up_block3(self.x1, self.x7)

        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x8))
        else:
            out = self.out_conv1(self.x8)

        return out


class DenseNet2DLayer124(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DLayer124, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)

        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x4 = self.down_block4(self.x2)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x2, self.x6)
        self.x8 = self.up_block4(self.x1, self.x7)

        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x8))
        else:
            out = self.out_conv1(self.x8)

        return out


class DenseNet2DLayer135(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DLayer135, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x3 = self.down_block3(self.x1)
        self.x5 = self.down_block5(self.x3)
        self.x6 = self.up_block1(self.x3, self.x5)
        self.x7 = self.up_block2(self.x1, self.x6)

        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x7))
        else:
            out = self.out_conv1(self.x7)

        return out


# ******************************************************************************************
# use compressed blocks


class DenseNet2D_compressed_blocks(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_compressed_blocks, self).__init__()

        self.down_block1 = DenseNet2D_down_block_1Conv(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.up_block1(self.x4, self.x5)
        self.x7 = self.up_block2(self.x3, self.x6)
        self.x8 = self.up_block3(self.x2, self.x7)
        self.x9 = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out





class DenseNet2DLayer135_compressed_blocks(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DLayer135_compressed_blocks, self).__init__()

        self.down_block1 = DenseNet2D_down_block_1Conv(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block_1Conv(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x3 = self.down_block3(self.x1)
        self.x5 = self.down_block5(self.x3)
        self.x6 = self.up_block1(self.x3, self.x5)
        self.x7 = self.up_block2(self.x1, self.x6)

        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x7))
        else:
            out = self.out_conv1(self.x7)

        return out



class DenseNet2DLayer135_compressed_down_blocks12_up_blocks1conv(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DLayer135_compressed_down_blocks12_up_blocks1conv, self).__init__()

        self.down_block1 = DenseNet2D_down_block12(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_1Conv(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                    prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x3 = self.down_block3(self.x1)
        self.x5 = self.down_block5(self.x3)
        self.x6 = self.up_block1(self.x3, self.x5)
        self.x7 = self.up_block2(self.x1, self.x6)

        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x7))
        else:
            out = self.out_conv1(self.x7)

        return out
"""


class DenseNet2DLayer135_compressed_down_blocks12_up_blocks1(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2DLayer135_compressed_down_blocks12_up_blocks1, self).__init__()

        self.down_block1 = DenseNet2D_down_block12(input_channels=in_channels, output_channels=channel_size,
                                                   down_size=None, dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                   down_size=(4, 4), dropout=dropout, prob=prob)

        self.down_block5 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                   down_size=(4, 4), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat1(skip_channels=channel_size, input_channels=channel_size,
                                                     output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                     prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat1(skip_channels=channel_size, input_channels=channel_size,
                                                     output_channels=channel_size, up_stride=(4, 4), dropout=dropout,
                                                     prob=prob)

        self.out_conv1 = nn.Conv2d(in_channels=channel_size, out_channels=out_channels, kernel_size=1, padding=0)
        self.concat = concat
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x1)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x3)
        self.x6, self.ub1_first_fsp_layer = self.up_block1(self.x3, self.x5)
        self.x7, self.ub2_first_fsp_layer = self.up_block2(self.x1, self.x6)

        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x7))
        else:
            out = self.out_conv1(self.x7)

        # 2. definicija s sum
        self.att_map_x1_sum, self.att_map_x3_sum, self.att_map_x5_sum, self.att_map_x6_sum, self.att_map_x7_sum = [
            g.mean(1) for g in (self.x1, self.x3, self.x5, self.x6, self.x7)]

        # 3. definicija z max
        # print(self.x1.shape)  # [4, 32, 640, 400]
        self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x6_max, self.att_map_x7_max = [
            g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x6, self.x7)]

        return out











