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


		# But if in upsampling we use bilinear or unpool, then the num of channels stays the same in the upsampling.
		# So we need to manually adjust things to retain the simetry.
		# - Down4:
		# We need to make down4 not increase the number of kernels by expansion. This way for level 3 (down3, up1) 
		# in the skip connection will receive (k) instead of (k*expansion//expansion) which is cool.
		# - Up:
		# Up channels now correcly receive 2*k channels. But in the normal version they would output k, and for the next up level, this would go to (k//expansion)
		# Since this doesn't happen, instead of Conv(2*k, k) and Conv(k, k), we should have Conv(2*k, k) and Conv(k, k//expansion).


		# In this way we can then use unpool or upsample (bilinear), or we can use convtranspose that retains the num of kernels instead of reducing them.

		# Of these methods, I would suggest unpooling. It is the same as segnet uses. It seems to make more sense then bilinear upsampling.
		# And if you would be using convtranspose, you might as well do the reduction.



		# This implementation will focus on doing UNet without the reduction of kernels in the upsampling.
		# If you want to use the reduction, go look for another file in this project.
		# It seems needless to mesh these two concepts together.
		# It would just create complexity and not make further use any easier to have it in just one file.






		# Example: depth == 3, in_chans = 3, starting kernels = 4, expansion = 2
		# kernels == [4, 8, 16]
		# self.inc == DoubleConv(3, 4)
		# self.downs == [Down(4, 8), Down(8, 16)]
		# self.last_down = Down(16, 16)
		# self.ups == [Up(32, 16, 8), Up(16, 8, 4)]
		# self.last_up = Up(8, 4, 4)
		# self.outc = OutConv(4, 2)

		# Or this example written more generally:
		# Example: depth == 3, in_chans = 3, starting kernels = 4, expansion = 2
		# kernels == [4, 8, 16]
		# self.inc == DoubleConv(in_chan, kernels[0])
		# self.downs == [Down(kernels[0], kernels[1]), Down(kernels[1], kernels[2])]
		# self.last_down = Down(kernels[2], kernels[2])
		# self.ups == [Up(2*kernels[2], kernels[2], kernels[1]), Up(2*kernels[1], kernels[1], kernels[0])]
		# self.last_up = Up(2*kernels[0], kernels[0], kernels[0])
		# self.outc = OutConv(kernels[0], out_chan)

		# Or better for programming in terms of indexing:
		# Example: depth == 3, in_chans = 3, starting kernels = 4, expansion = 2
		# kernels == [4, 8, 16]
		# self.inc == DoubleConv(in_chan, kernels[0])
		# self.downs == [Down(kernels[0], kernels[1]), Down(kernels[1], kernels[2])]
		# self.last_down = Down(kernels[2], kernels[2])
		# self.ups == [Up(2*kernels[-1], kernels[-1], kernels[-2]), Up(2*kernels[-2], kernels[-2], kernels[-3])]
		# self.last_up = Up(2*kernels[-3], kernels[-3], kernels[-3])
		# self.outc = OutConv(kernels[-3], out_chan)




		kernels = []
		for i in range(depth):
			kernels.append(int(starting_kernels * (expansion**i)))

		self.inc = DoubleConv(n_channels, kernels[0], kernels[0])

		downs = []
		for i in range(depth - 1):
			downs.append(Down(kernels[i], kernels[i+1]))
		self.downs = nn.ModuleList(downs)
		
		
		self.last_down = Down(kernels[-1], kernels[-1])
		
		ups = []
		for i in range(depth - 1):
			ups.append(Up(2*kernels[-1-i], kernels[-1-i], kernels[-1-i-1], mode))
		self.ups = nn.ModuleList(ups)

		self.last_up = Up(2*kernels[0], kernels[0], kernels[0], mode)

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
		past_inds = []
		
		x = self.inc(x)
		past_xs.append(x)

		for down in self.downs:
			x, inds = down(x)
			past_xs.append(x)
			past_inds.append(inds)
		
		x, inds = self.last_down(x)
		past_inds.append(inds)

		for i, up in enumerate(self.ups):
			x = up(x, past_xs[-1-i], past_inds[-1-i])
		
		x = self.last_up(x, past_xs[0], past_inds[0])

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
		
		self.mp = nn.MaxPool2d(2, stride=2, return_indices=True)
		self.dc = DoubleConv(in_channels, out_channels, out_channels)
		
	def forward(self, x):
		x, inds = self.mp(x)
		x = self.dc(x)
		return x, inds
	

class Up(nn.Module):
	"""Upscaling then double conv"""

	def __init__(self, in_channels, mid_channels, out_channels, mode="unpool"):
		super().__init__()

		self.mode = mode

		
		if mode == "unpool":
			self.up = nn.MaxUnpool2d(2, stride=2)
		elif mode == "bilinear":
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		elif mode == "convtranspose":
			self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
		else:
			raise NotImplementedError(f"Mode not implemented: {mode}")

		self.conv = DoubleConv(in_channels, mid_channels, out_channels)



	def forward(self, x1, x2, maxpool_inds=None):


		print(f"""
		From unet.py, line 231 - forward method of Up class:
		# Theres a problem with unpooling. If the conv right before unpooling is pruned, then x1 will have e.g. 511 chans, and indices will have 512 chans.
		# And if the conv on the same level in the down path is pruned, then x2 will have 511 chans.
		# So we have to  make sure that if either of those two convs are pruned, the other is also.

		# But since this will not be used, I leave it for later.
		""")

		# Theres a problem with unpooling. If the conv right before unpooling is pruned, then x1 will have e.g. 511 chans, and indices will have 512 chans.
		# And if the conv on the same level in the down path is pruned, then x2 will have 511 chans.
		# So we have to  make sure that if either of those two convs are pruned, the other is also.

		# But since this will not be used, I leave it for later.


		# Why cant be done otherwise:
		# or we can just do a resizing of inds to match x1. And we do a NEAREST method resizing. 
		# And this will give us the most likely positioning that corresponds to the maxpooling indices.

		# But this cant work. Because maxpool indices array isn't an array of true and false with the size of the input of maxpool.
		# It is an array with int tuples that are indices regarding the input, and the tensor is the size of the output.

		# So instead what we could do is, we could resize x1 to indices (along batch dimensions). Then do a maxunpool. 
		# And then resize it back to the num of channels that it had.		
		
		if self.mode == "unpool":
			x1 = self.up(x1, maxpool_inds)
		else:
			x1 = self.up(x1)

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
