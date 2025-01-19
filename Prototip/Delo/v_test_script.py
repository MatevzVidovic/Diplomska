



import os
import logging
import python_logger.log_helper as py_log
import python_logger.log_helper as py_log_always_on


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)








from dataset_aug_tf import IrisDataset




# In our UNet implementation the dims can be whatever you want.
# You could even change them between training iterations - but it might be a bad idea because all the weights had been learnt at the scale of the previous dims.
INPUT_DIMS = {
    "width" : 1024,
    "height" : 512,
    "channels" : 5
}

# In our UNet the output width and height have to be the same as the input width and height. 
OUTPUT_DIMS = {
    "width" : INPUT_DIMS["width"],
    "height" : INPUT_DIMS["height"],
    "channels" : 2
}



model_parameters = {
    # layer sizes
    "output_y" : OUTPUT_DIMS["height"],
    "output_x" : OUTPUT_DIMS["width"],
    "n_channels" : INPUT_DIMS["channels"],
    "n_classes" : OUTPUT_DIMS["channels"],
    "starting_kernels" : 64,
    "expansion" : 2,
    "depth" : 6,
}



dataset_args = {

    "testrun" : True,
    "testrun_size" : 2,


    "input_width" : INPUT_DIMS["width"],
    "input_height" : INPUT_DIMS["height"],
    "output_width" : OUTPUT_DIMS["width"],
    "output_height" : OUTPUT_DIMS["height"],
    
    # iris dataset params
    "path_to_sclera_data" : "Data/vein_and_sclera_data",
    # "transform" : transform,
    "n_classes" : OUTPUT_DIMS["channels"],

    "zero_out_non_sclera" : False,
    "add_sclera_to_img" : False,
    "add_bcosfire_to_img" : True,
    "add_coye_to_img" : True
}


data_path = dataset_args["path_to_sclera_data"]
# n_classes = 4 if 'sip' in args.dataset.lower() else 2

train_dataset = IrisDataset(filepath=data_path, split='train', **dataset_args)


real_imgs = train_dataset[0]

print(real_imgs)









# Naive patchification




from helper_img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs, save_imgs_quick_figs

import numpy as np
import torch

import matplotlib.pyplot as plt
import os.path as osp
# from utils import one_hot2dist

np.random.seed(7)






def unfold_3chan(tensor_img, patch_shape, stride: tuple):

    try:
        # tensor_img_unf = tensor_img.unfold(0, patch_shape[0], stride[0])
        # tensor_img_unf = tensor_img_unf.unfold(1, patch_shape[1], stride[1])

        y_ix = 0
        x_ix = 0

        if y_ix + patch_shape[0] <= tensor_img.size(1) and x_ix + patch_shape[1] <= tensor_img.size(2):
            patches = tensor_img[:, y_ix:y_ix + patch_shape[0], x_ix:x_ix + patch_shape[1]]
            patches = patches.unsqueeze(0)
            left_upper_ixs = [(y_ix, x_ix)]
            y_ix += stride[0]
        else:
            return None
        

        while x_ix + patch_shape[1] <= tensor_img.size(2):
            while y_ix + patch_shape[0] <= tensor_img.size(1):
                patch = tensor_img[:, y_ix:y_ix + patch_shape[0], x_ix:x_ix + patch_shape[1]]
                patch = patch.unsqueeze(0)
                patches = torch.cat([patches, patch], dim=0)
                left_upper_ixs.append((y_ix, x_ix))
                y_ix += stride[0]
            y_ix = 0
            x_ix += stride[1]
        
        return patches, left_upper_ixs
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e




def patchify(tensor_img, patch_shape, stride: tuple):

    try:
        main_patches, main_lu_ixs = unfold_3chan(tensor_img, patch_shape, stride)
        print(f"{main_patches=}")
        print(f"{main_patches.shape=}")
        print(f"{main_lu_ixs=}")




        # right_bottom_patches:

        # right_patches
        # bottom_patches
        # right_bottom_corner

        # with left up ixs for all of them. And then you concat all of that.
        # And at test time you add that to that parts of the accumulator.
        # Possibly you could count how many times each part of the accumulator was added to and devide by that,
        # but I'm not sure we need that, because it's logits anyway.




        # right_patches

        right_slice = tensor_img[:, :, -patch_shape[1]:]

        right_patches, right_lu_ixs = unfold_3chan(right_slice, patch_shape, stride)
        right_lu_ixs = [(i[0], i[1]+tensor_img.size(2)-patch_shape[1]) for i in right_lu_ixs]

        print(f"{right_slice=}")
        print(f"{right_patches=}")
        print(f"{right_patches.shape=}")
        print(f"{right_lu_ixs=}")




        # bottom_patches

        bottom_slice = tensor_img[:, -patch_shape[0]:, :]
        bottom_patches, bottom_lu_ixs = unfold_3chan(bottom_slice, patch_shape, stride)
        bottom_lu_ixs = [(i[0]+tensor_img.size(1)-patch_shape[0], i[1]) for i in bottom_lu_ixs]
        print(f"{bottom_slice=}")
        print(f"{bottom_patches=}")
        print(f"{bottom_patches.shape=}")
        print(f"{bottom_lu_ixs=}")




        # right_bottom_corner

        right_bottom_corner = tensor_img[:, -patch_shape[0]:, -patch_shape[1]:]
        right_bottom_patches, right_bottom_lu_ixs = unfold_3chan(right_bottom_corner, patch_shape, stride)
        right_bottom_lu_ixs = [(i[0]+tensor_img.size(1)-patch_shape[0], i[1]+tensor_img.size(2)-patch_shape[1]) for i in right_bottom_lu_ixs]
        print(f"{right_bottom_corner=}")
        print(f"{right_bottom_patches=}")
        print(f"{right_bottom_patches.shape=}")
        print(f"{right_bottom_lu_ixs=}")


        
        # patch_dict = {
        #     "main_patches" : tensor_img_unf,
        #     "main_lu_ixs" : patch_indices,
        #     "right_patches" : rs_patches,
        #     "right_lu_ixs" : rs_ixs,
        #     "bottom_patches" : bs_patches,
        #     "bottom_lu_ixs" : bs_ixs,
        #     "right_bottom_corner" : rbc_patch,
        #     "right_bottom_corner_lu_ixs" : rbc_ixs
        # }

        patch_dict = {
            "main_patches" : main_patches,
            "main_lu_ixs" : main_lu_ixs,
            "right_patches" : right_patches,
            "right_lu_ixs" : right_lu_ixs,
            "bottom_patches" : bottom_patches,
            "bottom_lu_ixs" : bottom_lu_ixs,
            "right_bottom_corner" : right_bottom_patches,
            "right_bottom_corner_lu_ixs" : right_bottom_lu_ixs
        }

        py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"])
        return patch_dict
    

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e



def accumulate_patches(prediction_tensor_shape, patch_shape, stride: tuple, patch_dict):

    try:
    
        accumulating_tensor = torch.zeros(prediction_tensor_shape)
        num_of_addings = torch.zeros(prediction_tensor_shape)
        print(f"{accumulating_tensor=}")

        all_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
        all_ixs = patch_dict["main_lu_ixs"] + patch_dict["right_lu_ixs"] + patch_dict["bottom_lu_ixs"] + patch_dict["right_bottom_corner_lu_ixs"]

        for i, (lu_ix, patch) in enumerate(zip(all_ixs, all_patches)):
            y1, x1 = lu_ix
            y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
            accumulating_tensor[:, y1:y2, x1:x2] += patch
            num_of_addings[:, y1:y2, x1:x2] += 1


        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["main_lu_ixs"], patch_dict["main_patches"])):

        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch
        
        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["right_lu_ixs"], patch_dict["right_patches"])):
            
        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch

        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["bottom_lu_ixs"], patch_dict["bottom_patches"])):

        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch
        
        # for i, (lu_ix, patch) in enumerate(zip(patch_dict["right_bottom_corner_lu_ixs"], patch_dict["right_bottom_corner"])):
            
        #     y1, x1 = lu_ix
        #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
        #     accumulating_tensor[:, y1:y2, x1:x2] += patch
        
        py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"])
        return accumulating_tensor, num_of_addings
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e




def try_patchification():
    try:






        img = train_dataset[0]["images"]


        patch_shape = (90, 110)
        stride = (patch_shape[0]//2, patch_shape[1]//2)






        print(f"img.shape={img.shape}")
        print(f"img={img}")

        patch_dict = patchify(img, patch_shape, stride)

        # The img is 2-channel from the start so the patches are also 2 channel, so they are the same as they would come out of the model

        print(f"{patch_dict['main_patches'].shape=}")
        print(f"{patch_dict['right_patches'].shape=}")
        print(f"{patch_dict['bottom_patches'].shape=}")
        print(f"{patch_dict['right_bottom_corner'].shape=}")

        # make tensors have 2 channels as if they went through the model


        concated_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
        print(f"{concated_patches=}")


        # deconcat patches
        num_main = patch_dict["main_patches"].size(0)
        num_right = patch_dict["right_patches"].size(0)
        num_bottom = patch_dict["bottom_patches"].size(0)
        num_rbc = patch_dict["right_bottom_corner"].size(0)

        pred_patch_dict = {
            "main_patches" : concated_patches[:num_main],
            "right_patches" : concated_patches[num_main:num_main + num_right],
            "bottom_patches" : concated_patches[num_main + num_right:num_main + num_right + num_bottom],
            "right_bottom_corner" : concated_patches[num_main + num_right + num_bottom:],

            "main_lu_ixs" : patch_dict["main_lu_ixs"],
            "right_lu_ixs" : patch_dict["right_lu_ixs"],
            "bottom_lu_ixs" : patch_dict["bottom_lu_ixs"],
            "right_bottom_corner_lu_ixs" : patch_dict["right_bottom_corner_lu_ixs"]
        }

        prediction_tensor_shape = (5, img.shape[1], img.shape[2])

        accumulating_tensor, num_of_addings = accumulate_patches(prediction_tensor_shape, patch_shape, stride, pred_patch_dict)

        reconstructed_img = accumulating_tensor / num_of_addings

        are_same = torch.allclose(reconstructed_img, img, atol=1e-8)
        print(f"{are_same=}")



        print(f"{accumulating_tensor=}")
        print(f"{accumulating_tensor.shape=}")


        py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"])

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


try_patchification()
















def naive_patchification():
    try:






        def get_rising_tensor(shape):

            # Calculate the total number of elements needed
            num_elements = torch.prod(torch.tensor(shape)).item()

            # Create a 1D tensor with sequential numbers
            sequential_tensor = torch.arange(1, num_elements + 1)

            # Reshape the tensor to the desired shape
            reshaped_tensor = sequential_tensor.reshape(shape)

            return reshaped_tensor

        # tensor_img_shape = (4, 6)
        d3_shape = (2, 3, 4)

        # tensor_img = get_rising_tensor(tensor_img_shape)
        d3 = get_rising_tensor(d3_shape)

        patch_shape = (2,2)
        stride = (2,2)





        print(f"{d3=}")
        patch_dict = patchify(d3, patch_shape, stride)

        # The img is 2-channel from the start so the patches are also 2 channel, so they are the same as they would come out of the model

        print(f"{patch_dict['main_patches'].shape=}")
        print(f"{patch_dict['right_patches'].shape=}")
        print(f"{patch_dict['bottom_patches'].shape=}")
        print(f"{patch_dict['right_bottom_corner'].shape=}")

        # make tensors have 2 channels as if they went through the model


        concated_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
        print(f"{concated_patches=}")


        # deconcat patches
        num_main = patch_dict["main_patches"].size(0)
        num_right = patch_dict["right_patches"].size(0)
        num_bottom = patch_dict["bottom_patches"].size(0)
        num_rbc = patch_dict["right_bottom_corner"].size(0)

        pred_patch_dict = {
            "main_patches" : concated_patches[:num_main],
            "right_patches" : concated_patches[num_main:num_main + num_right],
            "bottom_patches" : concated_patches[num_main + num_right:num_main + num_right + num_bottom],
            "right_bottom_corner" : concated_patches[num_main + num_right + num_bottom:],

            "main_lu_ixs" : patch_dict["main_lu_ixs"],
            "right_lu_ixs" : patch_dict["right_lu_ixs"],
            "bottom_lu_ixs" : patch_dict["bottom_lu_ixs"],
            "right_bottom_corner_lu_ixs" : patch_dict["right_bottom_corner_lu_ixs"]
        }

        prediction_tensor_shape = (2, d3_shape[1], d3_shape[2])

        accumulating_tensor, num_of_addings = accumulate_patches(prediction_tensor_shape, patch_shape, stride, pred_patch_dict)

        reconstructed_img = accumulating_tensor / num_of_addings
        # d3 are longs, so we need to convert them to floats
        d3 = d3.float()
        reconstructed_img = reconstructed_img.float()
        is_same = torch.allclose(reconstructed_img, d3, atol=1e-8)

        print(f"{accumulating_tensor=}")
        print(f"{accumulating_tensor.shape=}")

        py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"])

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


naive_patchification()

























# def naive_patchification():
#     try:






#         def get_rising_tensor(shape):

#             # Calculate the total number of elements needed
#             num_elements = torch.prod(torch.tensor(shape)).item()

#             # Create a 1D tensor with sequential numbers
#             sequential_tensor = torch.arange(1, num_elements + 1)

#             # Reshape the tensor to the desired shape
#             reshaped_tensor = sequential_tensor.reshape(shape)

#             return reshaped_tensor

#         # tensor_img_shape = (4, 6)
#         d3_shape = (2, 3, 4)

#         # tensor_img = get_rising_tensor(tensor_img_shape)
#         d3 = get_rising_tensor(d3_shape)

#         patch_shape = (2,2)
#         stride = (2,2)




#         def unfold_3chan(tensor_img, patch_shape, stride: tuple):

#             try:
#                 # tensor_img_unf = tensor_img.unfold(0, patch_shape[0], stride[0])
#                 # tensor_img_unf = tensor_img_unf.unfold(1, patch_shape[1], stride[1])

#                 y_ix = 0
#                 x_ix = 0

#                 if y_ix + patch_shape[0] <= tensor_img.size(1) and x_ix + patch_shape[1] <= tensor_img.size(2):
#                     patches = tensor_img[:, y_ix:y_ix + patch_shape[0], x_ix:x_ix + patch_shape[1]]
#                     patches = patches.unsqueeze(0)
#                     left_upper_ixs = [(y_ix, x_ix)]
#                     y_ix += stride[0]
#                 else:
#                     return None
                

#                 while x_ix + patch_shape[1] <= tensor_img.size(2):
#                     while y_ix + patch_shape[0] <= tensor_img.size(1):
#                         patch = tensor_img[:, y_ix:y_ix + patch_shape[0], x_ix:x_ix + patch_shape[1]]
#                         patch = patch.unsqueeze(0)
#                         patches = torch.cat([patches, patch], dim=0)
#                         left_upper_ixs.append((y_ix, x_ix))
#                         y_ix += stride[0]
#                     y_ix = 0
#                     x_ix += stride[1]
                
#                 return patches, left_upper_ixs
#             except Exception as e:
#                 py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
#                 raise e




#         def patchify(tensor_img, patch_shape, stride: tuple):

#             try:
#                 main_patches, main_lu_ixs = unfold_3chan(tensor_img, patch_shape, stride)
#                 print(f"{main_patches=}")
#                 print(f"{main_patches.shape=}")
#                 print(f"{main_lu_ixs=}")




#                 # right_bottom_patches:

#                 # right_patches
#                 # bottom_patches
#                 # right_bottom_corner

#                 # with left up ixs for all of them. And then you concat all of that.
#                 # And at test time you add that to that parts of the accumulator.
#                 # Possibly you could count how many times each part of the accumulator was added to and devide by that,
#                 # but I'm not sure we need that, because it's logits anyway.




#                 # right_patches

#                 right_slice = tensor_img[:, :, -patch_shape[1]:]

#                 right_patches, right_lu_ixs = unfold_3chan(right_slice, patch_shape, stride)
#                 right_lu_ixs = [(i[0], i[1]+tensor_img.size(2)-patch_shape[1]) for i in right_lu_ixs]

#                 print(f"{right_slice=}")
#                 print(f"{right_patches=}")
#                 print(f"{right_patches.shape=}")
#                 print(f"{right_lu_ixs=}")




#                 # bottom_patches

#                 bottom_slice = tensor_img[:, -patch_shape[0]:, :]
#                 bottom_patches, bottom_lu_ixs = unfold_3chan(bottom_slice, patch_shape, stride)
#                 bottom_lu_ixs = [(i[0]+tensor_img.size(1)-patch_shape[0], i[1]) for i in bottom_lu_ixs]
#                 print(f"{bottom_slice=}")
#                 print(f"{bottom_patches=}")
#                 print(f"{bottom_patches.shape=}")
#                 print(f"{bottom_lu_ixs=}")




#                 # right_bottom_corner

#                 right_bottom_corner = tensor_img[:, -patch_shape[0]:, -patch_shape[1]:]
#                 right_bottom_patches, right_bottom_lu_ixs = unfold_3chan(right_bottom_corner, patch_shape, stride)
#                 right_bottom_lu_ixs = [(i[0]+tensor_img.size(1)-patch_shape[0], i[1]+tensor_img.size(2)-patch_shape[1]) for i in right_bottom_lu_ixs]
#                 print(f"{right_bottom_corner=}")
#                 print(f"{right_bottom_patches=}")
#                 print(f"{right_bottom_patches.shape=}")
#                 print(f"{right_bottom_lu_ixs=}")


                
#                 # patch_dict = {
#                 #     "main_patches" : tensor_img_unf,
#                 #     "main_lu_ixs" : patch_indices,
#                 #     "right_patches" : rs_patches,
#                 #     "right_lu_ixs" : rs_ixs,
#                 #     "bottom_patches" : bs_patches,
#                 #     "bottom_lu_ixs" : bs_ixs,
#                 #     "right_bottom_corner" : rbc_patch,
#                 #     "right_bottom_corner_lu_ixs" : rbc_ixs
#                 # }

#                 patch_dict = {
#                     "main_patches" : main_patches,
#                     "main_lu_ixs" : main_lu_ixs,
#                     "right_patches" : right_patches,
#                     "right_lu_ixs" : right_lu_ixs,
#                     "bottom_patches" : bottom_patches,
#                     "bottom_lu_ixs" : bottom_lu_ixs,
#                     "right_bottom_corner" : right_bottom_patches,
#                     "right_bottom_corner_lu_ixs" : right_bottom_lu_ixs
#                 }

#                 py_log.log_locals(MY_LOGGER)
#                 return patch_dict
            

#             except Exception as e:
#                 py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
#                 raise e



#         def accumulate_patches(prediction_tensor_shape, patch_shape, stride: tuple, patch_dict):

#             try:
            
#                 accumulating_tensor = torch.zeros(prediction_tensor_shape)
#                 num_of_addings = torch.zeros(prediction_tensor_shape)
#                 print(f"{accumulating_tensor=}")

#                 all_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
#                 all_ixs = patch_dict["main_lu_ixs"] + patch_dict["right_lu_ixs"] + patch_dict["bottom_lu_ixs"] + patch_dict["right_bottom_corner_lu_ixs"]

#                 for i, (lu_ix, patch) in enumerate(zip(all_ixs, all_patches)):
#                     y1, x1 = lu_ix
#                     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
#                     accumulating_tensor[:, y1:y2, x1:x2] += patch
#                     num_of_addings[:, y1:y2, x1:x2] += 1


#                 # for i, (lu_ix, patch) in enumerate(zip(patch_dict["main_lu_ixs"], patch_dict["main_patches"])):

#                 #     y1, x1 = lu_ix
#                 #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
#                 #     accumulating_tensor[:, y1:y2, x1:x2] += patch
                
#                 # for i, (lu_ix, patch) in enumerate(zip(patch_dict["right_lu_ixs"], patch_dict["right_patches"])):
                    
#                 #     y1, x1 = lu_ix
#                 #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
#                 #     accumulating_tensor[:, y1:y2, x1:x2] += patch

#                 # for i, (lu_ix, patch) in enumerate(zip(patch_dict["bottom_lu_ixs"], patch_dict["bottom_patches"])):

#                 #     y1, x1 = lu_ix
#                 #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
#                 #     accumulating_tensor[:, y1:y2, x1:x2] += patch
                
#                 # for i, (lu_ix, patch) in enumerate(zip(patch_dict["right_bottom_corner_lu_ixs"], patch_dict["right_bottom_corner"])):
                    
#                 #     y1, x1 = lu_ix
#                 #     y2, x2 = y1 + patch_shape[0], x1 + patch_shape[1]
#                 #     accumulating_tensor[:, y1:y2, x1:x2] += patch
                
#                 py_log.log_locals(MY_LOGGER)
#                 return accumulating_tensor
#             except Exception as e:
#                 py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
#                 raise e




#         print(f"{d3=}")
#         patch_dict = patchify(d3, patch_shape, stride)

#         # The img is 2-channel from the start so the patches are also 2 channel, so they are the same as they would come out of the model

#         print(f"{patch_dict['main_patches'].shape=}")
#         print(f"{patch_dict['right_patches'].shape=}")
#         print(f"{patch_dict['bottom_patches'].shape=}")
#         print(f"{patch_dict['right_bottom_corner'].shape=}")

#         # make tensors have 2 channels as if they went through the model


#         concated_patches = torch.cat([patch_dict["main_patches"], patch_dict["right_patches"], patch_dict["bottom_patches"], patch_dict["right_bottom_corner"]], dim=0)
#         print(f"{concated_patches=}")


#         # deconcat patches
#         num_main = patch_dict["main_patches"].size(0)
#         num_right = patch_dict["right_patches"].size(0)
#         num_bottom = patch_dict["bottom_patches"].size(0)
#         num_rbc = patch_dict["right_bottom_corner"].size(0)

#         pred_patch_dict = {
#             "main_patches" : concated_patches[:num_main],
#             "right_patches" : concated_patches[num_main:num_main + num_right],
#             "bottom_patches" : concated_patches[num_main + num_right:num_main + num_right + num_bottom],
#             "right_bottom_corner" : concated_patches[num_main + num_right + num_bottom:],

#             "main_lu_ixs" : patch_dict["main_lu_ixs"],
#             "right_lu_ixs" : patch_dict["right_lu_ixs"],
#             "bottom_lu_ixs" : patch_dict["bottom_lu_ixs"],
#             "right_bottom_corner_lu_ixs" : patch_dict["right_bottom_corner_lu_ixs"]
#         }

#         prediction_tensor_shape = (2, d3_shape[1], d3_shape[2])

#         accumulating_tensor = accumulate_patches(prediction_tensor_shape, patch_shape, stride, pred_patch_dict)


#         print(f"{accumulating_tensor=}")
#         print(f"{accumulating_tensor.shape=}")

#     except Exception as e:
#         py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
#         raise e


# naive_patchification()
















# import helper_json_handler as jh

# curr_initial_train_iter = 8
# j_path = os.path.join(os.path.dirname(__file__), "initial_train_iters.json")
# j_dict = jh.load(j_path)

# if j_dict is None:
#     j_dict = {"initial_train_iters" : [curr_initial_train_iter]}
# else:
#     j_dict["initial_train_iters"].append(curr_initial_train_iter)

# jh.dump(j_path, j_dict)

# print(j_dict)















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


