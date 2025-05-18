



import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open(f"{osp.join('pylog_configs', 'active_logging_config.txt')}", 'r') as f:
    cfg_name = f.read()
    yaml_path = osp.join('pylog_configs', cfg_name)

log_config_path = osp.join(yaml_path)
do_log = False
if osp.exists(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
        file_log_setting = config.get(osp.basename(__file__), False)
        if file_log_setting:
            do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)





from y_datasets.img_augments import random_rotation, zoom_in_somewhere, random_gaussian_blur, random_horizontal_flip


from y_helpers.img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs, save_imgs_quick_figs

from y_helpers.patchification import get_random_patches

import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from PIL import Image
from torchvision import transforms as tf
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2

import y_helpers.shared as shared
if not shared.PLT_SHOW: # For more info, see shared.py
    import matplotlib
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import os.path as osp
# from utils import one_hot2dist

np.random.seed(7)




# transform = transforms.Compose(
#     [
#      transforms.Normalize([0.5], [0.5])
#     ])










class IrisDataset(Dataset):
    def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, clipLimit=1.5, **kwargs):
        
        self.transform = transform
        self.filepath= osp.join(filepath, split)
        
        self.input_width = kwargs['input_width']
        self.input_height = kwargs['input_height']
        self.output_width = kwargs['output_width']
        self.output_height = kwargs['output_height']

        self.testrun_length = kwargs['testrun_size']

        self.aug_type = kwargs.get('aug_type', 'tf')

        # These are here only to know how many channels of zeros to add to the image
        self.add_bcosfire_to_img = kwargs.get('add_bcosfire_to_img', False)
        self.add_coye_to_img = kwargs.get('add_coye_to_img', False)


        self.patchify = kwargs.get('patchify', False)
        self.patch_shape = kwargs.get('patch_shape', None)
        self.num_of_patches_from_img = kwargs.get('num_of_patches_from_img', None)
        self.prob_zero_patch_resample = kwargs.get('prob_zero_patch_resample', None)

        
        self.split = split
        self.classes = n_classes

        self.images_without_mask = []
        

        images_with_masks = []
        
        image_names = os.listdir(osp.join(self.filepath,'Images'))
        image_names.sort()
        
        to_scleras = osp.join(self.filepath,'Masks')
        


        for img_name in image_names:
            
            img_name_without_suffix = img_name.strip(".jpg")

            if self.masks_exist(to_scleras, img_name_without_suffix):
                images_with_masks.append(img_name_without_suffix)


        self.img_names_no_suffix = images_with_masks
        self.testrun = testrun

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))

        #summary
        print('summary for ' + str(split))
        print('valid images: ' + str(len(self.img_names_no_suffix)))


    def __len__(self):
        real_size = len(self.img_names_no_suffix)
        if self.testrun:
            return min(self.testrun_length, real_size)
        return real_size
    

    def masks_exist(self, scleras_folder_path, img_name_name_no_suffix):
        sclera_mask_filename = osp.join(scleras_folder_path, img_name_name_no_suffix + '.png')
        masks_exist = osp.exists(sclera_mask_filename)
        return masks_exist






    def tf_aug(self, idx, rand_id, show_testrun=False):


        img_name_no_suffix = self.img_names_no_suffix[idx]

        image_path = osp.join(self.filepath,'Images',f'{img_name_no_suffix}.jpg')
        scleras_path = osp.join(self.filepath,'Masks', f"{img_name_no_suffix}.png")
        
        masks = [scleras_path]

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # !!! 2RGB is the main diference from TF augmentation. And this messes up code clenliness.
        masks = [cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY) for mask in masks]


        img = F.to_tensor(img)
        masks = [F.to_tensor(mask) for mask in masks]

        # py_log_always_on.log_manual(MY_LOGGER, img=img, masks0=masks[0], masks1=masks[1], masks2=masks[2])




        # da_params = {
        #     "h_flip" : 0.5,
        #     "affine_params" : {
        #         "degrees" : 30,
        #         "translate" : (0.2, 0.2),
        #         "scale" : (0.8, 1.2),
        #         "shear" : 10
        #     },
        #     "color_jitter_params" : {
        #         "brightness" : 0.1,
        #         "contrast" : 0.1,
        #         "saturation" : 0.1,
        #         "hue" : 0.1
        #     },

        # }


        da_params = {
            "h_flip" : 0.5,
            "affine_params" : {
                "degrees" : 15,
                "translate" : (0.1, 0.1),
                "scale" : (0.8, 1.1),
                "shear" : 0.02
            },
            "color_jitter_params" : {
                "brightness" : 0.02,
                "contrast" : 0.02,
                "saturation" : 0.02,
                "hue" : 0.02
            },

        }



        # da_params = {
        #     "h_flip" : 0.5,
        #     "affine_params" : {
        #         "degrees" : 0,
        #         "translate" : (0.0, 0.0),
        #         "scale" : (1.0, 1.0),
        #         "shear" : 0.0
        #     },
        #     "color_jitter_params" : {
        #         "brightness" : 0.0,
        #         "contrast" : 0.0,
        #         "saturation" : 0.0,
        #         "hue" : 0.0
        #     },

        # }


        #py_log_always_on.log_time(MY_LOGGER, "test_da")

        if show_testrun or self.split == 'train':

            if random.random() > 0.5:
                img = F.hflip(img)
                masks = [F.hflip(mask) for mask in masks]
            
            affine_transform = tf.RandomAffine(**da_params["affine_params"])
            params = affine_transform.get_params(
                affine_transform.degrees, 
                affine_transform.translate, 
                affine_transform.scale, 
                affine_transform.shear,
                [img.shape[1], img.shape[2]])
            
            img = F.affine(img, *params, interpolation=F.InterpolationMode.BILINEAR)
            masks = [F.affine(mask, *params, interpolation=F.InterpolationMode.NEAREST) for mask in masks]

            #py_log_always_on.log_time(MY_LOGGER, "test_da")


            # Only applicable to img
            img = tf.ColorJitter(**da_params["color_jitter_params"])(img)

            #py_log_always_on.log_time(MY_LOGGER, "test_da")


        if show_testrun:
            save_imgs_quick_figs([img, *masks], f"{idx}_{rand_id}_da_over")
        


        img = smart_conversion(img, "ndarray", "uint8")
        masks = [smart_conversion(mask, "ndarray", "uint8") for mask in masks]

        return img, masks, img_name_no_suffix
            






    def np_aug(self, idx, rand_id, show_testrun=False):

        py_log_always_on.log_manual(MY_LOGGER, img_name_no_suffix=self.img_names_no_suffix, idx=idx)

        img_name_no_suffix = self.img_names_no_suffix[idx]

        image_path = osp.join(self.filepath,'Images',f'{img_name_no_suffix}.jpg')
        scleras_path = osp.join(self.filepath,'Masks', f"{img_name_no_suffix}.png")
        
        masks = [scleras_path]

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # !!! 2RGB is the main diference from TF augmentation. And this messes up code clenliness.
        masks = [cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB) for mask in masks]


        
        img = smart_conversion(img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]



        save_img_ix = 0

        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_before_da")

        # Testing
        if show_testrun:

            img_test = img
            masks_test = masks
            
            py_log_always_on.log_time(MY_LOGGER, "test_da")

            py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); save_img_ix += 1; save_imgs_quick_figs([img_test, *masks_test], f"{idx}_{save_img_ix}_original_image")
            # py_log_always_on.log_time(MY_LOGGER, "test_da")
            img_test, masks_test = random_horizontal_flip(img_test, masks_test, prob=1.0)
            py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); save_img_ix += 1; save_imgs_quick_figs([img_test, *masks_test], f"{idx}_{save_img_ix}_horizontal_flip")
            py_log_always_on.log_time(MY_LOGGER, "test_da")


            ksize = (img_test.shape[0]//20)
            ksize = ksize+1 if ksize % 2 == 0 else ksize
            # py_log_always_on.log_time(MY_LOGGER, "test_da")
            img_test = random_gaussian_blur(img_test, possible_sigma_vals_list=np.linspace(1, 10, 50), ker_size=ksize, prob=1.0)
            py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); save_img_ix += 1; save_imgs_quick_figs([img_test, *masks_test], f"{idx}_{save_img_ix}_gaussian blur")
            py_log_always_on.log_time(MY_LOGGER, "test_da")

            # py_log_always_on.log_time(MY_LOGGER, "test_da")
            img_test, masks_test = random_rotation(img_test, masks_test, max_angle_diff=15, prob=1.0)
            py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); save_img_ix += 1; save_imgs_quick_figs([img_test, *masks_test], f"{idx}_{save_img_ix}_rotation")
            py_log_always_on.log_time(MY_LOGGER, "test_da")

            # py_log_always_on.log_time(MY_LOGGER, "test_da")
            img_test, masks_test = zoom_in_somewhere(img_test, masks_test, max_scale_percent=0.5, prob=1.0)
            py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); save_img_ix += 1; save_imgs_quick_figs([img_test, *masks_test], f"{idx}_{save_img_ix}_zoom_blur")
            py_log_always_on.log_time(MY_LOGGER, "test_da")

            # py_log_always_on.log_time(MY_LOGGER, "test_da")

            if True:
                img = img_test
                masks = masks_test




        

        if not show_testrun and self.split == 'train':

            img, masks = random_horizontal_flip(img, masks, prob=0.5)

            ksize = (img.shape[0]//20)
            ksize = ksize+1 if ksize % 2 == 0 else ksize
            img = random_gaussian_blur(img, possible_sigma_vals_list=np.linspace(1, 10, 50), ker_size=ksize, prob=0.1)

            img, masks = random_rotation(img, masks, max_angle_diff=30, prob=0.5)

            img, masks = zoom_in_somewhere(img, masks, max_scale_percent=0.3, prob=0.7)





        #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da")






        # Converting img and masks to correct dimensions, binarization, types, and removing unnecessary dimension
        img = smart_conversion(img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]

        masks = [cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) for mask in masks]



        #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da_converted")

        return img, masks, img_name_no_suffix



    def pass_through(self, idx, rand_id, show_testrun=False):


        # py_log_always_on.log_manual(MY_LOGGER, img_name_no_suffix=self.img_names_no_suffix, idx=idx)

        # img_name_no_suffix = self.img_names_no_suffix[idx]

        # image_path = osp.join(self.filepath,'Images',f'{img_name_no_suffix}.jpg')
        # scleras_path = osp.join(self.filepath,'Masks', f"{img_name_no_suffix}.png")
        
        # masks = [scleras_path]

        # img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # # !!! 2RGB is the main diference from TF augmentation. And this messes up code clenliness.
        # masks = [cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY) for mask in masks]


        # # Converting img and masks to correct dimensions, binarization, types, and removing unnecessary dimension
        # img = smart_conversion(img, 'ndarray', 'uint8')
        # masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]




        # #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da_converted")

        # return img, masks, img_name_no_suffix






        py_log_always_on.log_manual(MY_LOGGER, img_name_no_suffix=self.img_names_no_suffix, idx=idx)

        img_name_no_suffix = self.img_names_no_suffix[idx]

        image_path = osp.join(self.filepath,'Images',f'{img_name_no_suffix}.jpg')
        scleras_path = osp.join(self.filepath,'Masks', f"{img_name_no_suffix}.png")
        
        masks = [scleras_path]

        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # !!! 2RGB is the main diference from TF augmentation. And this messes up code clenliness.
        masks = [cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2RGB) for mask in masks]


        
        img = smart_conversion(img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]




        if not show_testrun and self.split == 'train':

            img, masks = random_horizontal_flip(img, masks, prob=0.5)


        #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da")




        # Converting img and masks to correct dimensions, binarization, types, and removing unnecessary dimension
        img = smart_conversion(img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]

        masks = [cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY) for mask in masks]



        #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da_converted")

        return img, masks, img_name_no_suffix





    @py_log.autolog(passed_logger=MY_LOGGER)
    def __getitem__(self, idx):

        try:
            



            show_testrun = False
            rand_id = random.randint(0, 1000)

            if self.aug_type == 'tf':
                img, masks, img_name_no_suffix = self.tf_aug(idx, rand_id, show_testrun)
            elif self.aug_type == 'np':
                img, masks, img_name_no_suffix = self.np_aug(idx, rand_id, show_testrun)
            elif self.aug_type == 'pass':
                img, masks, img_name_no_suffix = self.pass_through(idx, rand_id, show_testrun)
            else:
                raise ValueError(f"Augmentation type not recognized. Given value: {self.aug_type}")






            # Converting img and masks to correct dimensions, binarization, types, and removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')
            masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]



            # #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LANCZOS4)
            masks = [cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LANCZOS4) for mask in masks]

            if show_testrun:
                py_log_always_on.log_time(MY_LOGGER, "test_da")
                save_imgs_quick_figs([img, *masks], f"{idx}_{rand_id}_resize_over")


            
            sclera = masks[0]

            sclera[sclera <= 127] = 0
            sclera[sclera > 127] = 1
            sclera = np.expand_dims(sclera, axis=2)


            # This is a workaround for the pretraining of multi
            # There we have a model that will train for sclera and veins at the same time, and will use these extra channels as input later.
            # But we want to first only train on the larger dataset of sclera annotations.
            # So there, we neet to add these fake channels to the image so we can use the model with these dimensions later.
            zero_channels_to_add = 0
            if self.add_bcosfire_to_img:
                zero_channels_to_add += 1
            if self.add_coye_to_img:
                zero_channels_to_add += 1

            
            if zero_channels_to_add > 0:
                to_concat = np.zeros((self.input_height, self.input_width, zero_channels_to_add), dtype=np.uint8)
                img = np.concatenate([img, to_concat], axis=2)



            # Conversion to standard types that pytorch can work with.
            img = smart_conversion(img, "tensor", "float32") # converts to float32

            # target tensor is long, because it is a classification task (in regression it would also be float32)
            sclera = smart_conversion(sclera, 'tensor', "uint8").long() # converts to int64


            sclera_for_show = sclera.clone() * 255
            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_imgs_quick_figs([img, sclera_for_show], f"{idx}_{save_img_ix}_final")

            # sclera mustn't have channels. It is a target, not an image.
            # And since the output of our network is (batch_size, n_classes, height, width), our target has to be (batch_size, height, width).
            # So here we need to return (height, width) veins, not (height, width, 1) veins.
            sclera = sclera.squeeze() # This function removes all dimensions of size 1 from the tensor

            if show_testrun:
                py_log_always_on.log_manual(MY_LOGGER, img=img, sclera=sclera)
                save_imgs_quick_figs([img, sclera_for_show], f"{idx}_{rand_id}_final")



            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])



            # while True:
            #     plt.pause(0.1)

            # Make them nicely concatable in custom_collate_fn
            img = img.unsqueeze(0)
            sclera = sclera.unsqueeze(0).unsqueeze(0) # It was only 2d before
            img_name_no_suffix = [img_name_no_suffix]

            if self.patchify:
                # these become a tensor like [num_of_patches_from_img, channels, height, width]
                img, sclera = get_random_patches([img, sclera], patch_shape=self.patch_shape, num_of_patches_from_img=self.num_of_patches_from_img, 
                                                        prob_zero_patch_resample=None, resample_gt=None)
                
                old_img_name_no_suffix = img_name_no_suffix[0]
                img_name_no_suffix = [f"{old_img_name_no_suffix}_{i}" for i in range(self.num_of_patches_from_img)]

            sclera = sclera.squeeze(1) # PyTorch needs img to be 4d, and veins to be 3d. Respectively: [batch_size, channels, height, width] and [batch_size, height, width]
            # So we need to remove the channel dimension from veins, because it is a target, not an image.



            return {'images': img, "scleras": sclera , 'img_names': img_name_no_suffix}
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e



def custom_collate_fn(batch):
    
    images = torch.concat([item['images'] for item in batch], dim=0)
    scleras = torch.concat([item['scleras'] for item in batch], dim=0)

    img_names = []
    for item in batch:
        img_names.extend(item['img_names']) 
    return {'images': images, "scleras" : scleras, 'img_names': img_names}


if __name__ == "__main__":
    
    # def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, clipLimit=1.5, **kwargs):

    # self.output_width = kwargs['output_width']
    # self.output_height = kwargs['output_height']

    # self.testrun_length = kwargs['testrun_size']


    python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
    handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)


    dataloading_args = {


        "testrun" : False,
        "testrun_size" : 30,
    

        "input_width" : 3000,
        "input_height" : 1500,
        "output_width" : 3000,
        "output_height" : 1500,

        # "input_width" : 256,
        # "input_height" : 256,
        # "output_width" : 256,
        # "output_height" : 256,
        
        "transform" : None,
        "n_classes" : 2,
        "aug_type" : "pass",


        "patchify" : False,
        "patch_shape" : (256, 256),
        "num_of_patches_from_img" : 4,
        "prob_zero_patch_resample" : 0.8 #1.0

    }


    
    data_path = "Data/sclera_data"

    train_dataset = IrisDataset(filepath=data_path, split='train', **dataloading_args)

    for i in range(10):
        res = train_dataset[0]
        py_log_always_on.log_manual(MY_LOGGER, img=res['images'], sclera=res['scleras'], names=res['img_names'])
        save_img_quick_figs(res['images'][0, :, :, :], f"test_{i}.png")
        save_img_quick_figs(res['scleras'][0]*255, f"test_{i}_scleras.png")
        print(f"{i} finished")

    # res = train_dataset[0]
    # py_log_always_on.log_manual(MY_LOGGER, img=res['images'], sclera=res['scleras'], names=res['img_names'])
    # for i in range(dataloading_args["num_of_patches_from_img"]):
    #     save_img_quick_figs(res['images'][i, :, :, :], f"test_{i}.png")
    #     save_img_quick_figs(res['scleras'][i]*255, f"test_{i}_scleras.png")

    # for patches:
    # res = train_dataset[0]
    # py_log_always_on.log_manual(MY_LOGGER, img=res['images'], sclera=res['scleras'], names=res['img_names'])
    # for i in range(dataloading_args["num_of_patches_from_img"]):
    #     save_img_quick_figs(res['images'][i, :, :, :], f"test_{i}.png")
    #     save_img_quick_figs(res['scleras'][i]*255, f"test_{i}_scleras.png")

    # res = train_dataset[0]
    # py_log_always_on.log_manual(MY_LOGGER, img=res['images'], sclera=res['scleras'], names=res['img_names'])
    # for i in range(dataloading_args["num_of_patches_from_img"]):
    #     save_img_quick_figs(res['images'][i, :, :, :], f"test_{i}.png")
    #     save_img_quick_figs(res['scleras'][i]*255, f"test_{i}_scleras.png")


# %%
