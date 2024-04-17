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


# modified original to have pruning capabilities
class DenseNet2D_down_block(nn.Module):
    def __init__(self,input_channels,output_channels,down_size,downsize_activations,activation_height, activation_width,activations_weighting_power, dropout=False,prob=0):
        super(DenseNet2D_down_block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(input_channels+output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv31 = nn.Conv2d(input_channels+2*output_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv32 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.max_pool = nn.AvgPool2d(kernel_size=down_size)

        self.relu = nn.LeakyReLU()
        self.down_size = down_size
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)
        self.dropout3 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)

        if downsize_activations and activations_weighting_power is not None:
            raise NotImplementedError()

        self.output_channels = output_channels
        self.downsize_activations = downsize_activations
        self.activation_height = activation_height
        self.activation_width = activation_width
        self.activations_weighting_power = activations_weighting_power
        self.reset_conv_activations_sum()

    def forward(self, x, prev_x=None):
        if prev_x is not None:
            x = torch.cat((x, prev_x), dim=1)

        if self.down_size != None:
            x = self.max_pool(x)

        if self.dropout:
            self.conv1_activations = self.conv1(x)
            self.conv1_activations = self._set_activations_for_removed_filters_to_zero(self.conv1, self.conv1_activations)
            x1 = self.relu(self.dropout1(self.conv1_activations))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            self.conv21_activations = self.conv21(x21)
            self.conv21_activations = self._set_activations_for_removed_filters_to_zero(self.conv21, self.conv21_activations)
            self.conv22_activations = self.conv22(self.conv21_activations)
            self.conv22_activations = self._set_activations_for_removed_filters_to_zero(self.conv22, self.conv22_activations)
            x22 = self.relu(self.dropout2(self.conv22_activations))
            x31 = torch.cat((x21,x22),dim=1)
            self.conv31_activations = self.conv31(x31)
            self.conv31_activations = self._set_activations_for_removed_filters_to_zero(self.conv31, self.conv31_activations)
            self.conv32_activations = self.conv32(self.conv31_activations)
            self.conv32_activations = self._set_activations_for_removed_filters_to_zero(self.conv32, self.conv32_activations)
            out = self.relu(self.dropout3(self.conv32_activations))
            last_fsp_layer = out

            self.conv1_activations_sum += self.save_activations_and_sum(self.conv1_activations)
            self.conv21_activations_sum += self.save_activations_and_sum(self.conv21_activations)
            self.conv22_activations_sum += self.save_activations_and_sum(self.conv22_activations)
            self.conv31_activations_sum += self.save_activations_and_sum(self.conv31_activations)
            self.conv32_activations_sum += self.save_activations_and_sum(self.conv32_activations)
        else:
            self.conv1_activations = self.conv1(x)
            self.conv1_activations = self._set_activations_for_removed_filters_to_zero(self.conv1, self.conv1_activations)
            x1 = self.relu(self.conv1_activations)
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            self.conv21_activations = self.conv21(x21)
            self.conv21_activations = self._set_activations_for_removed_filters_to_zero(self.conv21, self.conv21_activations)
            self.conv22_activations = self.conv22(self.conv21_activations)
            self.conv22_activations = self._set_activations_for_removed_filters_to_zero(self.conv22, self.conv22_activations)
            x22 = self.relu(self.conv22_activations)
            x31 = torch.cat((x21, x22), dim=1)
            self.conv31_activations = self.conv31(x31)
            self.conv31_activations = self._set_activations_for_removed_filters_to_zero(self.conv31, self.conv31_activations)
            self.conv32_activations = self.conv32(self.conv31_activations)
            self.conv32_activations = self._set_activations_for_removed_filters_to_zero(self.conv32, self.conv32_activations)
            out = self.relu(self.conv32_activations)
            last_fsp_layer = out

            self.conv1_activations_sum += self.save_activations_and_sum(self.conv1_activations)
            self.conv21_activations_sum += self.save_activations_and_sum(self.conv21_activations)
            self.conv22_activations_sum += self.save_activations_and_sum(self.conv22_activations)
            self.conv31_activations_sum += self.save_activations_and_sum(self.conv31_activations)
            self.conv32_activations_sum += self.save_activations_and_sum(self.conv32_activations)

        return self.bn(out)#, first_fsp_layer, last_fsp_layer

    def _set_activations_for_removed_filters_to_zero(self, layer, layer_activations):
        # manualy set activations for removed filters to zero (otherwise we run into some small numbers)
        index_of_removed_filters = self.get_index_of_removed_filters_for_weight(layer)
        layer.number_of_removed_filters = len(index_of_removed_filters)
        if index_of_removed_filters:
            layer_activations[:, index_of_removed_filters, :, :] = torch.zeros(layer_activations.shape[0],
                                                                               len(index_of_removed_filters),
                                                                               layer_activations.shape[2],
                                                                               layer_activations.shape[3]).cuda()
        return layer_activations

    def save_activations_and_sum(self, activations):
        activations = activations.detach()
        if self.downsize_activations:
            activations = nn.functional.interpolate(activations, size=(self.activation_height, self.activation_width),
                                                               mode='bilinear', align_corners=False)  # torch.nn.functional.interpolate(a, size=(4, 3), mode='bilinear', align_corners=False)
        # calculating sum on activations
        activations = activations.pow(2)
        if self.downsize_activations:
            n_summed_elements = activations.shape[0]
        else:
            n_summed_elements = activations.shape[0] * activations.shape[2] * activations.shape[3]

        activations = activations.sum(dim=[0, 2, 3]) / n_summed_elements

        return activations

    def get_index_of_removed_filters_for_weight(self, layer):
        weight = getattr(layer, 'weight')
        bias = getattr(layer, 'bias')
        # should have gradients set to zero and also weights and bias
        assert len(weight.shape) == 4
        assert len(bias.shape) == 1
        index_removed = []
        zero_filter_3d = torch.zeros(weight.shape[1:]).cuda()
        zero_filter_1d = torch.zeros(bias.shape[1:]).cuda()
        for index, (filter_weight, filter_bias) in enumerate(zip(weight, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_bias, zero_filter_1d):
                index_removed.append(index)
        return index_removed

    def reset_conv_activations_sum(self, device=None):
        #print('resetting activations sum for all layers..')
        if device:
            self.conv1_activations_sum = torch.nn.Parameter(torch.zeros(self.conv1.out_channels).to(device), requires_grad=False)
            self.conv21_activations_sum = torch.nn.Parameter(torch.zeros(self.conv21.out_channels).to(device), requires_grad=False)
            self.conv22_activations_sum = torch.nn.Parameter(torch.zeros(self.conv22.out_channels).to(device), requires_grad=False)
            self.conv31_activations_sum = torch.nn.Parameter(torch.zeros(self.conv31.out_channels).to(device), requires_grad=False)
            self.conv32_activations_sum = torch.nn.Parameter(torch.zeros(self.conv32.out_channels).to(device), requires_grad=False)
        else:
            self.conv1_activations_sum = torch.nn.Parameter(torch.zeros(self.conv1.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv21_activations_sum = torch.nn.Parameter(torch.zeros(self.conv21.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv22_activations_sum = torch.nn.Parameter(torch.zeros(self.conv22.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv31_activations_sum = torch.nn.Parameter(torch.zeros(self.conv31.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv32_activations_sum = torch.nn.Parameter(torch.zeros(self.conv32.out_channels), requires_grad=False) # 1, output_c, 640, 400



# modified original to have pruning capabilities
class DenseNet2D_up_block_concat(nn.Module):
    def __init__(self,skip_channels,input_channels,output_channels,up_stride,downsize_activations,activation_height,
                 activation_width,activations_weighting_power,dropout=False,prob=0, ):
        super(DenseNet2D_up_block_concat, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels+input_channels,output_channels,kernel_size=(1,1),padding=(0,0))
        self.conv12 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.conv21 = nn.Conv2d(skip_channels+input_channels+output_channels,output_channels,
                                kernel_size=(1,1),padding=(0,0))
        self.conv22 = nn.Conv2d(output_channels,output_channels,kernel_size=(3,3),padding=(1,1))
        self.relu = nn.LeakyReLU()
        self.up_stride = up_stride
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.dropout2 = nn.Dropout(p=prob)

        if downsize_activations and activations_weighting_power is not None:
            raise NotImplementedError()

        self.output_channels = output_channels
        self.downsize_activations = downsize_activations
        self.activation_height = activation_height
        self.activation_width = activation_width
        self.activations_weighting_power = activations_weighting_power
        self.reset_conv_activations_sum()


    def forward(self,prev_feature_map,x,residual_x=None):
        if residual_x is not None:
            x = torch.cat((x, residual_x), dim=1)

        if self.up_stride != None:
            x = nn.functional.interpolate(x,scale_factor=self.up_stride,mode='nearest')

        x = torch.cat((x,prev_feature_map),dim=1)
        if self.dropout:
            self.conv11_activations = self.conv11(x)
            self.conv11_activations = self._set_activations_for_removed_filters_to_zero(self.conv11, self.conv11_activations)
            self.conv12_activations = self.conv12(self.conv11_activations)
            self.conv12_activations = self._set_activations_for_removed_filters_to_zero(self.conv12, self.conv12_activations)
            x1 = self.relu(self.dropout1(self.conv12_activations))
            first_fsp_layer = x1
            x21 = torch.cat((x,x1),dim=1)
            self.conv21_activations = self.conv21(x21)
            self.conv21_activations = self._set_activations_for_removed_filters_to_zero(self.conv21, self.conv21_activations)
            self.conv22_activations = self.conv22(self.conv21_activations)
            self.conv22_activations = self._set_activations_for_removed_filters_to_zero(self.conv22, self.conv22_activations)
            out = self.relu(self.dropout2(self.conv22_activations))
            last_fsp_layer = out

            self.conv11_activations_sum += self.save_activations_sum(self.conv11_activations)
            self.conv12_activations_sum += self.save_activations_sum(self.conv12_activations)
            self.conv21_activations_sum += self.save_activations_sum(self.conv21_activations)
            self.conv22_activations_sum += self.save_activations_sum(self.conv22_activations)
        else:
            self.conv11_activations = self.conv11(x)
            self.conv11_activations = self._set_activations_for_removed_filters_to_zero(self.conv11, self.conv11_activations)
            self.conv12_activations = self.conv12(self.conv11_activations)
            self.conv12_activations = self._set_activations_for_removed_filters_to_zero(self.conv12, self.conv12_activations)
            x1 = self.relu(self.conv12_activations)
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            self.conv21_activations = self.conv21(x21)
            self.conv21_activations = self._set_activations_for_removed_filters_to_zero(self.conv21, self.conv21_activations)
            self.conv22_activations = self.conv22(self.conv21_activations)
            self.conv22_activations = self._set_activations_for_removed_filters_to_zero(self.conv22, self.conv22_activations)
            out = self.relu(self.conv22_activations)
            last_fsp_layer = out

            self.conv11_activations_sum += self.save_activations_sum(self.conv11_activations)
            self.conv12_activations_sum += self.save_activations_sum(self.conv12_activations)
            self.conv21_activations_sum += self.save_activations_sum(self.conv21_activations)
            self.conv22_activations_sum += self.save_activations_sum(self.conv22_activations)

        return out#, first_fsp_layer, last_fsp_layer

    def _set_activations_for_removed_filters_to_zero(self, layer, layer_activations):
        # manualy set activations for removed filters to zero (otherwise we run into some small numbers)
        index_of_removed_filters = self.get_index_of_removed_filters_for_weight(layer)
        layer.number_of_removed_filters = len(index_of_removed_filters)

        if index_of_removed_filters:
            layer_activations[:, index_of_removed_filters, :, :] = torch.zeros(layer_activations.shape[0],
                                                                               len(index_of_removed_filters),
                                                                               layer_activations.shape[2],
                                                                               layer_activations.shape[3]).cuda()
        return layer_activations


    def save_activations_sum(self, activations):
        activations = activations.detach()
        if self.downsize_activations:
            activations = nn.functional.interpolate(activations, size=(self.activation_height, self.activation_width),
                                                               mode='bilinear', align_corners=False)  # torch.nn.functional.interpolate(a, size=(4, 3), mode='bilinear', align_corners=False)
        # save activations before calculating sum
        # calculating sum on activations
        activations = activations.pow(2)
        if self.downsize_activations:
            n_summed_elements = activations.shape[0]
        else:
            n_summed_elements = activations.shape[0] * activations.shape[2] * activations.shape[3]

        activations = activations.sum(dim=[0, 2, 3]) / n_summed_elements
        return activations

    def get_index_of_removed_filters_for_weight(self, layer):
        weight = getattr(layer, 'weight')
        bias = getattr(layer, 'bias')
        # should have gradients set to zero and also weights and bias
        assert len(weight.shape) == 4
        assert len(bias.shape) == 1
        index_removed = []
        zero_filter_3d = torch.zeros(weight.shape[1:]).cuda()
        zero_filter_1d = torch.zeros(bias.shape[1:]).cuda()
        for index, (filter_weight, filter_bias) in enumerate(zip(weight, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_bias, zero_filter_1d):
                index_removed.append(index)
        return index_removed

    def reset_conv_activations_sum(self, device=None):
        #print('resetting activations sum for all layers..')
        if device:
            self.conv11_activations_sum = torch.nn.Parameter(torch.zeros(self.conv11.out_channels).to(device), requires_grad=False)
            self.conv12_activations_sum = torch.nn.Parameter(torch.zeros(self.conv12.out_channels).to(device), requires_grad=False)
            self.conv21_activations_sum = torch.nn.Parameter(torch.zeros(self.conv21.out_channels).to(device), requires_grad=False)
            self.conv22_activations_sum = torch.nn.Parameter(torch.zeros(self.conv22.out_channels).to(device), requires_grad=False)

        else:
            self.conv11_activations_sum = torch.nn.Parameter(torch.zeros(self.conv11.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv12_activations_sum = torch.nn.Parameter(torch.zeros(self.conv12.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv21_activations_sum = torch.nn.Parameter(torch.zeros(self.conv21.out_channels), requires_grad=False) # 1, output_c, 640, 400
            self.conv22_activations_sum = torch.nn.Parameter(torch.zeros(self.conv22.out_channels), requires_grad=False) # 1, output_c, 640, 400





# modified original to have pruning capabilities
class DenseNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=4,channel_size=32,downsize_activations=False,activation_height=40, activation_width=25,activations_weighting_power=None,concat=True,dropout=False,prob=0):
        super(DenseNet2D, self).__init__()

        self.down_block1 = DenseNet2D_down_block(input_channels=in_channels,output_channels=channel_size, down_size=None, dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.down_block2 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size, down_size=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.down_block3 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size, down_size=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.down_block4 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size, down_size=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.down_block5 = DenseNet2D_down_block(input_channels=channel_size,output_channels=channel_size, down_size=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)

        self.up_block1 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.up_block2 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.up_block3 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        self.up_block4 = DenseNet2D_up_block_concat(skip_channels=channel_size,input_channels=channel_size,
                                                    output_channels=channel_size,up_stride=(2,2),dropout=dropout,prob=prob, downsize_activations=downsize_activations, activation_height=activation_height, activation_width=activation_width, activations_weighting_power=activations_weighting_power)
        # TODO: vkljuci se ta conv v primerjavo
        self.out_conv1 = nn.Conv2d(in_channels=channel_size,out_channels=out_channels,kernel_size=1,padding=0)
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

    def forward(self,x):
        #self.x1, self.db1_first_fsp_layer, self.db1_last_fsp_layer = self.down_block1(x)
        self.x1 = self.down_block1(x)
        #self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x2 = self.down_block2(self.x1)
        #self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x3 = self.down_block3(self.x2)
        #self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x4 = self.down_block4(self.x3)
        #self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x5 = self.down_block5(self.x4)
        #self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4,self.x5)
        self.x6 = self.up_block1(self.x4,self.x5)
        #self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3,self.x6)
        self.x7 = self.up_block2(self.x3,self.x6)
        #self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2,self.x7)
        self.x8 = self.up_block3(self.x2,self.x7)
        #self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1,self.x8)
        self.x9 = self.up_block4(self.x1,self.x8)
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
        #self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x7_max, self.att_map_x9_max = [g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x7, self.x9)]
        return out


    def reset_conv_activations_sum(self, device):
        self.down_block1.reset_conv_activations_sum(device)
        self.down_block2.reset_conv_activations_sum(device)
        self.down_block3.reset_conv_activations_sum(device)
        self.down_block4.reset_conv_activations_sum(device)
        self.down_block5.reset_conv_activations_sum(device)
        self.up_block1.reset_conv_activations_sum(device)
        self.up_block2.reset_conv_activations_sum(device)
        self.up_block3.reset_conv_activations_sum(device)
        self.up_block4.reset_conv_activations_sum(device)


# original (used for teacher)
class DenseNet2D_original(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False,
                 prob=0):  # channel_size=32
        super(DenseNet2D_original, self).__init__()

        self.down_block1 = DenseNet2D_down_block_original(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
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
        #self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x7_max, self.att_map_x9_max = [
        #    g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x7, self.x9)]
        return out









# REFERENCNI MODELI

# original down block
class DenseNet2D_down_block_original(nn.Module):
    def __init__(self, input_channels, output_channels, down_size, dropout=False, prob=0):
        super(DenseNet2D_down_block_original, self).__init__()
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


# odstranis conv31 in conv32
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
            #x31 = torch.cat((x21, x22), dim=1)
            #out = self.relu(self.dropout3(self.conv32(self.conv31(x31))))
        else:
            x1 = self.relu(self.conv1(x))
            first_fsp_layer = x1
            x21 = torch.cat((x, x1), dim=1)
            out = self.relu(self.conv22(self.conv21(x21)))
            last_fsp_layer = out
            #x31 = torch.cat((x21, x22), dim=1)
            #out = self.relu(self.conv32(self.conv31(x31)))
        return self.bn(out), first_fsp_layer, last_fsp_layer

# original up block
class DenseNet2D_up_block_concat_original(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat_original, self).__init__()
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

        if prev_feature_map is not None:
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

# odstranis conv21 in conv22
class DenseNet2D_up_block_concat1(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels, up_stride, dropout=False, prob=0):
        super(DenseNet2D_up_block_concat1, self).__init__()
        self.conv11 = nn.Conv2d(skip_channels + input_channels, output_channels, kernel_size=(1, 1), padding=(0, 0))
        self.conv12 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        #self.conv21 = nn.Conv2d(skip_channels + input_channels + output_channels, output_channels,
         #                       kernel_size=(1, 1), padding=(0, 0))
        #self.conv22 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
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
            #x21 = torch.cat((x, x1), dim=1)
            #out = self.relu(self.dropout2(self.conv22(self.conv21(x21))))
        else:
            out = self.relu(self.conv12(self.conv11(x)))
            first_fsp_layer = out
            #x21 = torch.cat((x, x1), dim=1)
            #out = self.relu(self.conv22(self.conv21(x21)))
        return out, first_fsp_layer


class DenseNet2D_216k(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_216k, self).__init__()

        self.down_block1 = DenseNet2D_down_block_original(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block2 = DenseNet2D_up_block_concat_original(skip_channels=0, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block3 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
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
        #self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x2)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x2)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4, self.x5)
        self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(None, self.x6) # x3 does not exists
        self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        # 2. definicija s sum
        #self.att_map_x1_sum, self.att_map_x2_sum, self.att_map_x3_sum, self.att_map_x4_sum, self.att_map_x5_sum, \
        #self.att_map_x6_sum, self.att_map_x7_sum, self.att_map_x8_sum, self.att_map_x9_sum = \
        #    [g.mean(1) for g in (self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8, self.x9)]

        # 3. definicija z max
        # print(self.x1.shape) # [4, 32, 640, 400]
        #self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x7_max, self.att_map_x9_max = [
        #    g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x7, self.x9)]
        return out


class DenseNet2D_188k(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_188k, self).__init__()

        self.down_block1 = DenseNet2D_down_block12(input_channels=in_channels, output_channels=channel_size,
                                                   down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
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


class DenseNet2D_139k(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_139k, self).__init__()

        self.down_block1 = DenseNet2D_down_block12(input_channels=in_channels, output_channels=channel_size,
                                                   down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block4 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block12(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat1(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block2 = DenseNet2D_up_block_concat1(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block3 = DenseNet2D_up_block_concat1(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout,
                                                    prob=prob)
        self.up_block4 = DenseNet2D_up_block_concat1(skip_channels=channel_size, input_channels=channel_size,
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
        self.x6, _ = self.up_block1(self.x4, self.x5)
        self.x7, _ = self.up_block2(self.x3, self.x6)
        self.x8, _ = self.up_block3(self.x2, self.x7)
        self.x9, _ = self.up_block4(self.x1, self.x8)
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

#================== alternative


# alternativa za 216k (odstranjen DB3): odstranis DB4
class DenseNet2D_216k_alternative(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_216k_alternative, self).__init__()

        self.down_block1 = DenseNet2D_down_block_original(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block2 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.down_block3 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=0, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block2 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block3 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
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
        #self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x2)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x3)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(None, self.x5) # x4 does not exists
        self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3, self.x6)
        self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x8)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


# alternativa za 188k (brez conv31 in conv32): odstranis DB2 in UB3
class DenseNet2D_188k_alternative(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_188k_alternative, self).__init__()

        self.down_block1 = DenseNet2D_down_block_original(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)
        self.down_block3 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(4, 4), dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block2 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)
        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(4, 4), dropout=dropout)
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
        #self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x1)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x3)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4, self.x5)
        self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3, self.x6)
        #self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x7)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out



# alternativa za 139k (brez conv31, conv32 za DB in conv21 in conv22 za UB): odstranis DB2, DB3 in UB2, UB3
class DenseNet2D_139k_alternative(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_139k_alternative, self).__init__()

        self.down_block1 = DenseNet2D_down_block_original(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(8, 8), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)

        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(8, 8), dropout=dropout)
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
        #self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        #self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x1)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x1)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4, self.x5)
        #self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3, self.x6)
        #self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x6)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out



# odstranis DB2, DB3 in UB2, UB3
class DenseNet2D_139k_alternative(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, channel_size=32, concat=True, dropout=False, prob=0):
        super(DenseNet2D_139k_alternative, self).__init__()

        self.down_block1 = DenseNet2D_down_block_original(input_channels=in_channels, output_channels=channel_size,
                                                 down_size=None, dropout=dropout, prob=prob)

        self.down_block4 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(8, 8), dropout=dropout, prob=prob)
        self.down_block5 = DenseNet2D_down_block_original(input_channels=channel_size, output_channels=channel_size,
                                                 down_size=(2, 2), dropout=dropout, prob=prob)

        self.up_block1 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(2, 2), dropout=dropout)

        self.up_block4 = DenseNet2D_up_block_concat_original(skip_channels=channel_size, input_channels=channel_size,
                                                    output_channels=channel_size, up_stride=(8, 8), dropout=dropout)
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
        #self.x2, self.db2_first_fsp_layer, self.db2_last_fsp_layer = self.down_block2(self.x1)
        #self.x3, self.db3_first_fsp_layer, self.db3_last_fsp_layer = self.down_block3(self.x1)
        self.x4, self.db4_first_fsp_layer, self.db4_last_fsp_layer = self.down_block4(self.x1)
        self.x5, self.db5_first_fsp_layer, self.db5_last_fsp_layer = self.down_block5(self.x4)
        self.x6, self.ub1_first_fsp_layer, self.ub1_last_fsp_layer = self.up_block1(self.x4, self.x5)
        #self.x7, self.ub2_first_fsp_layer, self.ub2_last_fsp_layer = self.up_block2(self.x3, self.x6)
        #self.x8, self.ub3_first_fsp_layer, self.ub3_last_fsp_layer = self.up_block3(self.x2, self.x7)
        self.x9, self.ub4_first_fsp_layer, self.ub4_last_fsp_layer = self.up_block4(self.x1, self.x6)
        if self.dropout:
            out = self.out_conv1(self.dropout1(self.x9))
        else:
            out = self.out_conv1(self.x9)

        return out


# REFERENCNI MODELI END







# student za train with knowledge distillation
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
        self.att_map_x1_sum, self.att_map_x3_sum, self.att_map_x5_sum, self.att_map_x6_sum, self.att_map_x7_sum = [g.mean(1) for g in (self.x1, self.x3, self.x5, self.x6, self.x7)]

        # 3. definicija z max
        #print(self.x1.shape)  # [4, 32, 640, 400]
        self.att_map_x1_max, self.att_map_x3_max, self.att_map_x5_max, self.att_map_x6_max, self.att_map_x7_max = [g.pow(2).max(dim=1)[0] for g in (self.x1, self.x3, self.x5, self.x6, self.x7)]

        return out
