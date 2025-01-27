


import torch
import torch.nn.functional as F



def pad_to_larger(x1, x2):

	diffY = x2.size()[2] - x1.size()[2]
	diffX = x2.size()[3] - x1.size()[3]

	pad_y_x1 = 0
	pad_x_x1 = 0
	pad_y_x2 = 0
	pad_x_x2 = 0

	if diffY < 0:
		pad_y_x2 = -diffY
	else:
		pad_y_x1 = diffY
	
	if diffX < 0:
		pad_x_x2 = -diffX
	else:
		pad_x_x1 = diffX

	x1 = F.pad(x1, [pad_x_x1 // 2, pad_x_x1 - pad_x_x1 // 2,
					pad_y_x1 // 2, pad_y_x1 - pad_y_x1 // 2])
	
	x2 = F.pad(x2, [pad_x_x2 // 2, pad_x_x2 - pad_x_x2 // 2,
					pad_y_x2 // 2, pad_y_x2 - pad_y_x2 // 2])

	return x1, x2


def pad_or_resize_to_second_arg(x1, x2):
	# x2 is the larger tensor.

	y = x2.size()[2]
	x = x2.size()[3]

	x1 = pad_or_resize_to_dims(x1, y, x)
	return x1


def pad_or_resize_to_dims(x1, y, x):
	# x2 is the larger tensor.

	diffY = y - x1.size()[2]
	diffX = x - x1.size()[3]


	x_resize = False
	y_resize = False

	if diffY < 0:
		pad_y = 0
		y_resize = True
	else:
		pad_y = diffY
	
	if diffX < 0:
		pad_x = 0
		x_resize = True
	else:
		pad_x = diffX


	x1 = F.pad(x1, [pad_x // 2, pad_x - pad_x // 2,
					pad_y // 2, pad_y - pad_y // 2])
	
	if x_resize or y_resize:
		x1 = F.interpolate(x1, size=(y, x), mode='bilinear')


	return x1