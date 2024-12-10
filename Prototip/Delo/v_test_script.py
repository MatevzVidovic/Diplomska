



import os
import logging
import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)











import json_handler as jh

curr_initial_train_iter = 8
j_path = os.path.join(os.path.dirname(__file__), "initial_train_iters.json")
j_dict = jh.load(j_path)

if j_dict is None:
    j_dict = {"initial_train_iters" : [curr_initial_train_iter]}
else:
    j_dict["initial_train_iters"].append(curr_initial_train_iter)

jh.dump(j_path, j_dict)

print(j_dict)















# import orjson

# curr_initial_train_iter = 8

# import os.path as osp

# json_path = osp.join(osp.dirname(__file__), "initial_train_iters.json")

# if osp.exists(json_path):
#     with open(json_path, 'rb') as f:
#         json_str = f.read()
#         json_dict = orjson.loads(json_str)
#     json_dict["initial_train_iters"].append(curr_initial_train_iter)

# else:
#     json_dict = {"initial_train_iters" : [curr_initial_train_iter]}


# with open(json_path, 'wb') as f:
#     json_str = orjson.dumps(json_dict)
#     f.write(json_str)

# print(json_dict)














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





# from torch import nn

# pool = nn.MaxPool2d(2, stride=2, return_indices=True)
# unpool = nn.MaxUnpool2d(2, stride=2)
# input = torch.tensor([[[[ 1.,  2.,  3.,  4.],
#                         [ 5.,  6.,  7.,  8.],
#                         [ 9., 10., 11., 12.],
#                         [13., 14., 15., 16.]]]])
# output, indices = pool(input)
# unpool(output, indices)

# print(indices)

# # Now using output_size to resolve an ambiguous size for the inverse
# input = torch.tensor([[[[ 1.,  2.,  3.,  4.,  5.],
#                         [ 6.,  7.,  8.,  9., 10.],
#                         [11., 12., 13., 14., 15.],
#                         [16., 17., 18., 19., 20.]]]])
# output, indices = pool(input)
# # This call will not work without specifying output_size
# unpool(output, indices, output_size=input.size())

# print(indices)
























# import numpy as np

# arr = np.array([1, 2, 6, 7, 8, 9])
# # np.save(os.path.join(os.path.dirname(__file__), "test.npy"), arr)
# # np.savetxt(os.path.join(os.path.dirname(__file__), "test.txt"), arr, fmt='%s')
# # np.savetxt(os.path.join(os.path.dirname(__file__), "test.txt"), arr)
# np.savetxt(os.path.join(os.path.dirname(__file__), "test.txt"), arr, fmt='%d')



# # arr = np.load(os.path.join(os.path.dirname(__file__), "test.npy"))
# # arr = np.loadtxt(os.path.join(os.path.dirname(__file__), "test.txt"), dtype=str, delimiter=",")
# # arr = np.loadtxt(os.path.join(os.path.dirname(__file__), "test.txt"))
# arr = np.loadtxt(os.path.join(os.path.dirname(__file__), "test.txt"), dtype=int)
# print(arr)








# import torch
# initial_train_iter = 0

# import os
# import pandas as pd
# # main_save_path = os.path.join(os.path.dirname(__file__), 'test')
# main_save_path = os.path.dirname(__file__)

# initial_train_iters_path = os.path.join(main_save_path, "initial_train_iters.csv")

# if os.path.exists(initial_train_iters_path):
#     initial_train_iters = pd.read_csv(initial_train_iters_path)
#     new_row = pd.DataFrame({"initial_train_iters": [initial_train_iters]})
#     print(new_row)
#     print(initial_train_iters)
#     new_df = pd.concat([initial_train_iters, new_row], ignore_index=True, axis=0)
# else:
#     new_df = pd.DataFrame({"initial_train_iters": [initial_train_iter]})

# print(new_df)

# new_df.to_csv(os.path.join(main_save_path, "initial_train_iters.csv"), index=False)


















# import pandas as pd
# import numpy as np


# def csv_to_numpy(csv_path, header=None):
#     df = pd.read_csv(csv_path, index_col=False)
#     print(df)

#     if header != None:
#         cols = df.columns.tolist()

#     df_np = df.to_numpy()
#     print(df_np)

#     returner = df_np
#     if header != None:
#         returner = (df_np, cols)
                    
#     return returner


# def numpy_to_csv(numpy_array, csv_path, header=None):
#     cols = False if header == None else header
#     df = pd.DataFrame(numpy_array)
#     df.to_csv(csv_path, index=False, header=cols)




# import torch
# initial_train_iter = 0

# import os
# # main_save_path = os.path.join(os.path.dirname(__file__), 'test')
# main_save_path = os.path.dirname(__file__)

# csv_path = os.path.join(main_save_path, "initial_train_iters.csv")

# if os.path.exists(csv_path):

#     initial_train_iters, cols = csv_to_numpy(csv_path, True)

#     new_row = np.array([initial_train_iter])

#     print("Here")
#     print(initial_train_iters)
#     print(new_row)
#     print("Here")

#     new_iters = np.vstack((initial_train_iters, new_row))
#     print(new_iters)
#     numpy_to_csv(new_iters, csv_path, cols)

#     # initial_train_iters = pd.read_csv(csv_path, index_col=False)
#     # new_row = pd.DataFrame({"initial_train_iters": [initial_train_iters]})
#     # print(new_row)
#     # print(initial_train_iters)
#     # new_df = pd.concat([initial_train_iters, new_row], ignore_index=True, axis=0)
# else:
#     curr_np = np.array([initial_train_iter])
#     cols = ["initial_train_iters"]
#     numpy_to_csv(curr_np, csv_path, cols)
















# import torch
# curr_initial_train_iter = 8

# import os
# import pandas as pd
# # main_save_path = os.path.join(os.path.dirname(__file__), 'test')
# main_save_path = os.path.dirname(__file__)

# initial_train_iters_path = os.path.join(main_save_path, "initial_train_iters.csv")

# if os.path.exists(initial_train_iters_path):
#     initial_train_iters = pd.read_csv(initial_train_iters_path)
#     new_row = pd.DataFrame({"initial_train_iters": [curr_initial_train_iter]})
#     print(new_row)
#     print("delim")
#     print(initial_train_iters)
#     print("delim")

#     new_df = pd.concat([initial_train_iters, new_row], ignore_index=True, axis=0)
# else:
#     new_df = pd.DataFrame({"initial_train_iters": [curr_initial_train_iter]})

# print(new_df)

# new_df.to_csv(os.path.join(main_save_path, "initial_train_iters.csv"), index=False)


