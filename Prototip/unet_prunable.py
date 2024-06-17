"""
Code adapted from https://github.com/milesial/Pytorch-UNet/
Licenced under GNU GPLv3
All credit goes to the original authors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
	def __init__(self, n_channels=1, n_classes=4, bilinear=True, pretrained=False):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64, bilinear)
		self.outc = OutConv(64, n_classes)

		self.reset_conv_activations_sum()

		if pretrained:
			pretrained_state = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana').state_dict()
			new_state = self.state_dict()
			# self.inc and self.outc depend on n_channels and n_classes which are different from the Caravana dataset in our case
			pretrained_state = {k: v for k, v in pretrained_state.items() if k in new_state and 'inc.' not in k and 'outc.' not in k}
			new_state.update()
			self.load_state_dict(new_state)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		return logits

	def reset_conv_activations_sum(self, device=None):
		self.inc.reset_conv_activations_sum(device)
		self.down1.reset_conv_activations_sum(device)
		self.down2.reset_conv_activations_sum(device)
		self.down3.reset_conv_activations_sum(device)
		self.down4.reset_conv_activations_sum(device)
		self.up1.reset_conv_activations_sum(device)
		self.up2.reset_conv_activations_sum(device)
		self.up3.reset_conv_activations_sum(device)
		self.up4.reset_conv_activations_sum(device)

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(mid_channels)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu2 = nn.ReLU()

	def forward(self, x):
		self.conv1_activations = self.conv1(x)
		self.conv1_activations = self._set_activations_for_removed_filters_to_zero(self.conv1, self.conv1_activations)
		x = self.relu1(self.bn1(self.conv1_activations))
		self.conv2_activations = self.conv2(x)
		self.conv2_activations = self._set_activations_for_removed_filters_to_zero(self.conv2, self.conv2_activations)
		x = self.relu2(self.bn2(self.conv2_activations))

		self.conv1_activations_sum += self.save_activations_and_sum(self.conv1_activations)
		self.conv2_activations_sum += self.save_activations_and_sum(self.conv2_activations)

		return x

	def _set_activations_for_removed_filters_to_zero(self, layer, layer_activations):
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
		activations = _downsize_activations(activations)
		# calculating sum on activations
		activations = activations.pow(2)
		n_summed_elements = activations.shape[0]  # downsize_activations=True
		# n_summed_elements = activations.shape[0] * activations.shape[2] * activations.shape[3]  # downsize_activations=False
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
		if device:
			self.conv1_activations_sum = torch.nn.Parameter(torch.zeros(self.conv1.out_channels).to(device), requires_grad=False)
			self.conv2_activations_sum = torch.nn.Parameter(torch.zeros(self.conv2.out_channels).to(device), requires_grad=False)
		else:
			self.conv1_activations_sum = torch.nn.Parameter(torch.zeros(self.conv1.out_channels), requires_grad=False)
			self.conv2_activations_sum = torch.nn.Parameter(torch.zeros(self.conv2.out_channels), requires_grad=False)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool = nn.MaxPool2d(2)
		self.conv = DoubleConv(in_channels, out_channels)
		self.maxpool_conv = nn.Sequential(
			self.maxpool,
			self.conv
		)

	def forward(self, x):
		return self.maxpool_conv(x)

	def reset_conv_activations_sum(self, device=None):
		self.conv.reset_conv_activations_sum(device)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
		else:
			self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		# input is CHW
		diffY = x2.size()[2] - x1.size()[2]
		diffX = x2.size()[3] - x1.size()[3]

		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x2, x1], dim=1)
		return self.conv(x)

	def reset_conv_activations_sum(self, device=None):
		self.conv.reset_conv_activations_sum(device)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)


def _downsize_activations(activations):
	# equivalent to downsize_activations=False from densenet
	#return activations
	# equivalent to downsize_activations=True, activation_height=40, activation_width=25
	return nn.functional.interpolate(activations, size=(40, 25), mode='bilinear', align_corners=False)
