



import os
import logging
import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)



import torch



# a = {
#     1: 'a',
#     2: 'b',
#     3: 'c'
# }

# for k in a:
#     print(k)




# # Weights method:

# L1_over_L2_alpha = 0.5

# # example_weights = torch.rand(3, 3, 5, 5)
# sizes = [3, 3, 2, 2]

# prod = 1
# for s in sizes:
#     prod *= s

# example_weights = torch.arange(0, prod).reshape(*sizes).float()


# # [num_of_kernels, depth, h, w]
# kernels_weights = example_weights # averaging_objects[tree_ix][2]
# overall_weights = kernels_weights.mean(dim=(0))
# d = kernels_weights.shape[1]
# h = kernels_weights.shape[2]
# w = kernels_weights.shape[3]
# diff = kernels_weights - overall_weights
# L1 = torch.abs(diff).sum(dim=(1,2,3)) / (d*h*w)
# L2 = (diff).pow(2).sum(dim=(1,2,3)).sqrt() / (d*h*w)
# kernel_importance = L1_over_L2_alpha * L1 + (1 - L1_over_L2_alpha) * L2


# print(kernel_importance)

# py_log.log_locals(MY_LOGGER)





from torch import nn

pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
                        [ 5.,  6.,  7.,  8.],
                        [ 9., 10., 11., 12.],
                        [13., 14., 15., 16.]]]])
output, indices = pool(input)
unpool(output, indices)

print(indices)

# Now using output_size to resolve an ambiguous size for the inverse
input = torch.tensor([[[[ 1.,  2.,  3.,  4.,  5.],
                        [ 6.,  7.,  8.,  9., 10.],
                        [11., 12., 13., 14., 15.],
                        [16., 17., 18., 19., 20.]]]])
output, indices = pool(input)
# This call will not work without specifying output_size
unpool(output, indices, output_size=input.size())

print(indices)
