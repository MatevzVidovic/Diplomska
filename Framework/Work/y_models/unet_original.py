"""
Code adapted from https://github.com/milesial/Pytorch-UNet/
Licenced under GNU GPLv3
All credit goes to the original authors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



from y_helpers.padding import pad_to_larger, pad_or_resize_to_second_arg, pad_or_resize_to_dims



class UNet(nn.Module):
	def __init__(self, output_y, output_x, n_channels=1, n_classes=4, starting_kernels=64, expansion=2, depth=4, mode="unpool", pretrained=False):
		super(UNet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes





		# Here is how unet works: (e.g. depth 4)
		# We have, inc, we have down1-down3, down4, up1-up3, up4, (and outc but forget about it for now)
		# Skip connections go:
		# inc -> up4
		# down1 -> up3
		# down2 -> up2
		# down3 -> up1
		# Down4 doesn't produce any skip connection. It is simply the normal part of the input for the up1 layer.

		# Inc takes channels from in_chan to desired starting kernels.
		# Every part of the down path increases the number of kernels by expansion.
		
		# But the interesting thing is with the last layer and upsampling.
		# In the original paper, down4 also increases the number of kernels by expansion.
		# In the upsampling, an up-conv is used, which decreases the number of kernels by expansion.
		# Up1 then gets k channels from down3, and (k*expansion/expansion) channels from down4. So there is this nice symmetry of num of kernels from prev and from skip.
		
		# Moreover, in the rest of the up path, this happens:
		# (Lets define k := the number of out kernels the horizontal down block has on this level)
		# Up layer gets k channels from down block on the same horizontal level.
		# And it gets k channels from the previous layer (after the upsampling that reduced the layers).
		# So it has 2*k in channels. It then does 2 convs:  Conv(2*k, k) and Conv(k, k).
		# Now when passing this to the next up layer the upsampling will reduce the number of kernels by expansion and the symetry will be preserved. 







		# Example: depth == 3, in_chans = 3, starting kernels = 4, expansion = 2
		# kernels == [4, 8, 16, 32]
		# self.inc == DoubleConv(3, 4)
		# self.downs == [Down(4, 8), Down(8, 16), Down(16, 32)]
		# self.ups == [Up(32, 16), Up(16, 8), Up(8, 4)]
		# self.outc = OutConv(4, 2)

		# Or this example written more generally:
		# Example: depth == 3, in_chans = 3, starting kernels = 4, expansion = 2
		# kernels == [4, 8, 16, 32]
		# self.inc == DoubleConv(in_chan, kernels[0])
		# self.downs == [Down(kernels[0], kernels[1]), Down(kernels[1], kernels[2]), Down(kernels[2], kernels[3])]
		# self.ups == [Up(2*kernels[2], kernels[2]), Up(2*kernels[1], kernels[1]), Up(2*kernels[0], kernels[0])]
		# self.outc = OutConv(kernels[0], out_chan)



		kernels = []
		for i in range(depth+1):
			kernels.append(int(starting_kernels * (expansion**i)))


		self.inc = DoubleConv(n_channels, kernels[0], kernels[0])

		downs = []
		for i in range(depth):
			downs.append(Down(kernels[i], kernels[i+1]))
		self.downs = nn.ModuleList(downs)
		
		ups = []
		for i in range(depth):
			# levels_in_channels is then ** in Up()
			# But the reason we give levels_in_channels instead of just 2*kernels[i] is because of the unpooling.
			# Unpooling has to take what we got from the previous layer (had kernels[i+1] channels) and reduce that to levels_in_channels.
			# So we would rather just pass levels_in_channels and *2 in the layer itself.
			ups.insert(0, Up(levels_in_channels=kernels[i], out_channels=kernels[i], prev_out_channels=kernels[i+1]))
		self.ups = nn.ModuleList(ups)

		self.outc = OutConv(kernels[0], n_classes, output_y, output_x)




		if pretrained:
			pretrained_state = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana').state_dict()
			new_state = self.state_dict()
			# self.inc and self.outc depend on n_channels and n_classes which are different from the Caravana dataset in our case
			pretrained_state = {k: v for k, v in pretrained_state.items() if k in new_state and 'inc.' not in k and 'outc.' not in k}
			new_state.update()
			self.load_state_dict(new_state)
			

	def forward(self, x):
		past_xs = []
		
		x = self.inc(x)
		past_xs.append(x)

		for down in self.downs:
			x = down(x)
			past_xs.append(x)
		
		# The past_xs of the last down should never be used. It is simply used as the variable x.
		# This is why we have -2 in the next for loop.

		for i, up in enumerate(self.ups):
			x = up(x, past_xs[-2-i])
		
		logits = self.outc(x)
		return logits


class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2 => [Dropout]"""

	def __init__(self, in_channels, mid_channels, out_channels):
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
		
		self.mp = nn.MaxPool2d(2, stride=2, return_indices=False)
		self.dc = DoubleConv(in_channels, out_channels, out_channels)
		
	def forward(self, x):
		x= self.mp(x)
		x = self.dc(x)
		return x
	

class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, levels_in_channels, out_channels, prev_out_channels):
		super().__init__()
		
		self.up = nn.ConvTranspose2d(prev_out_channels, levels_in_channels, kernel_size=2, stride=2)
		self.conv = DoubleConv(2*levels_in_channels, out_channels, out_channels)



	def forward(self, x1, x2):

		# print(f"{x1.shape=}, {x2.shape=}")
		
		x1 = self.up(x1)

		# print(f"{x1.shape=}")

		x1, x2 = pad_to_larger(x1, x2)
		
		# if you have padding issues, see
		# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
		# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
		x = torch.cat([x1, x2], dim=1)
		return self.conv(x)


class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels, output_y, output_x):
		super(OutConv, self).__init__()
		self.output_y = output_y
		self.output_x = output_x
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		x = pad_or_resize_to_dims(x, self.output_y, self.output_x)
		return self.conv(x)
