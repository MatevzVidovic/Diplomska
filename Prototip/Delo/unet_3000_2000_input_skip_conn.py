"""
Code adapted from https://github.com/milesial/Pytorch-UNet/
Licenced under GNU GPLv3
All credit goes to the original authors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from padding import pad_to_larger, pad_or_resize_to_second_arg, pad_or_resize_to_dims




	


class UNet(nn.Module):
	def __init__(self, output_y, output_x, n_channels=1, n_classes=4, bilinear=True, pretrained=False, expansion=2, starting_kernels=64):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear



		
		in_chans = [n_channels]
		kernels = [starting_kernels]
		
		self.inc = DoubleConv(in_chans[-1], kernels[-1])

		# In every layer we will get the original input as well. They will then resize it to their own needs.
		# This means we need to add n_channels to the in_channels.

		in_chans.append(kernels[-1] + n_channels); kernels.append(int(expansion*kernels[-1]))
		self.down1 = Down(in_chans[-1], kernels[-1])
		
		in_chans.append(kernels[-1] + n_channels); kernels.append(int(expansion*kernels[-1]))
		self.down2 = Down(in_chans[-1], kernels[-1])
		
		in_chans.append(kernels[-1] + n_channels); kernels.append(int(expansion*kernels[-1]))
		self.down3 = Down(in_chans[-1], kernels[-1])
		

		# if bilinear, then a simple parameterless bilinear upsampling is happening.
		# If not, a conv layer is used for the upsampling. 
		factor = 2 if bilinear else 1
		
		in_chans.append(kernels[-1] + n_channels); kernels.append(kernels[3])
		self.down4 = Down(in_chans[-1], kernels[-1])
		
		# kernels == 1024

		# We would like to have the same number of input channels from the previous layer as we get from skip connections.
		# Just to rougly keep things balanced.
		# So we should simply define the number of kernels in the Up path as is the 
		# number of kernels of the next layer's skip connection Down layer.

		# Just look at the pattern below and it will be clear.

		# kernels[-1] + kernels[3] because it concats the outputs of self.down4 and self.down3
		in_chans.append(kernels[-1] + kernels[3] + n_channels); kernels.append(kernels[2])
		self.up1 = Up(in_chans[-1], kernels[-1], bilinear, expansion)
		
		in_chans.append(kernels[-1] + kernels[2] + n_channels); kernels.append(kernels[1])
		self.up2 = Up(in_chans[-1], kernels[-1], bilinear, expansion)
		
		in_chans.append(kernels[-1] + kernels[1] + n_channels); kernels.append(kernels[0])
		self.up3 = Up(in_chans[-1], kernels[-1], bilinear, expansion)
		
		# here we could have any number of kernels, so we choose a number that will have us have roughly the same number of flops in this layer as in the previous
		in_chans.append(kernels[-1] + kernels[0] + n_channels); kernels.append(int(kernels[0] / expansion))
		self.up4 = Up(in_chans[-1], kernels[-1], bilinear, expansion)
		
		in_chans.append(kernels[-1] + n_channels); kernels.append(n_classes)
		self.outc = OutConv(in_chans[-1], kernels[-1], output_y, output_x)

		if pretrained:
			pretrained_state = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana').state_dict()
			new_state = self.state_dict()
			# self.inc and self.outc depend on n_channels and n_classes which are different from the Caravana dataset in our case
			pretrained_state = {k: v for k, v in pretrained_state.items() if k in new_state and 'inc.' not in k and 'outc.' not in k}
			new_state.update()
			self.load_state_dict(new_state)

	def forward(self, x):
		oi = x.clone() # original_input
		x1 = self.inc(x)
		x2 = self.down1(x1, oi)
		x3 = self.down2(x2, oi)
		x4 = self.down3(x3, oi)
		x5 = self.down4(x4, oi)
		x = self.up1(x5, x4, oi)
		x = self.up2(x, x3, oi)
		x = self.up3(x, x2, oi)
		x = self.up4(x, x1, oi)
		logits = self.outc(x, oi)
		return logits

class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2 => [Dropout]"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			# nn.Dropout()
		)

	def forward(self, x):
		return self.double_conv(x)


class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x, oi):
		oi = pad_or_resize_to_second_arg(oi, x)
		x = torch.cat([x, oi], dim=1)
		return self.maxpool_conv(x)


class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, out_channels, bilinear=True, expansion=2):
		super().__init__()

		# if bilinear, use the normal convolutions to reduce the number of channels
		if bilinear:
			self.up = nn.Upsample(scale_factor=expansion, mode='bilinear', align_corners=True)
			self.conv = DoubleConv(in_channels, out_channels)
		else:
			self.up = nn.ConvTranspose2d(in_channels , int(in_channels // expansion), kernel_size=2, stride=2)
			self.conv = DoubleConv(in_channels, out_channels)


	def forward(self, x1, x2, oi):
		x1 = self.up(x1)
		# input is CHW


		x1, x2 = pad_to_larger(x1, x2) # these two now have the same dims
		
		oi = pad_or_resize_to_second_arg(oi, x1)

		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x1, x2, oi], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels, output_y, output_x):
		super(OutConv, self).__init__()
		self.output_y = output_y
		self.output_x = output_x
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x, oi):

		oi = pad_or_resize_to_second_arg(oi, x)
		x = torch.cat([x, oi], dim=1)
		
		x = pad_or_resize_to_dims(x, self.output_y, self.output_x) # we could also do this after self.conv(x). 
		# But since conv retains the dims, it's better to do it before, so that conv can compensate for artifacts of resizing, if resizing is happening.
		return self.conv(x)
