



import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open("active_logging_config.txt", 'r') as f:
    yaml_path = f.read()

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





from img_augments import random_rotation, zoom_in_somewhere, random_gaussian_blur, random_horizontal_flip


from helper_img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs, save_imgs_quick_figs

from helper_patchification import get_random_patches

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

import matplotlib.pyplot as plt
import os.path as osp
# from utils import one_hot2dist

np.random.seed(7)




# transform = transforms.Compose(
#     [
#      transforms.Normalize([0.5], [0.5])
#     ])










class SavePredsDataset(Dataset):
    def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, clipLimit=1.5, **kwargs):
        
        self.transform = transform
        self.filepath= osp.join(filepath, split)
        
        self.input_width = kwargs['input_width']
        self.input_height = kwargs['input_height']
        self.output_width = kwargs['output_width']
        self.output_height = kwargs['output_height']

        self.testrun_length = kwargs['testrun_size']

        self.aug_type = kwargs.get('aug_type', 'tf')


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
        



        for img_name in image_names:
            
            img_name_without_suffix = img_name.strip(".jpg")

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
    











    @py_log.autolog(passed_logger=MY_LOGGER)
    def __getitem__(self, idx):

        try:
            





            img_name_no_suffix = self.img_names_no_suffix[idx]

            image_path = osp.join(self.filepath,'Images',f'{img_name_no_suffix}.jpg')
            
            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            
            img = smart_conversion(img, 'ndarray', 'uint8')





            # Converting img and masks to correct dimensions, binarization, types, and removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')


            # #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LANCZOS4)

            


            # Conversion to standard types that pytorch can work with.
            img = smart_conversion(img, "tensor", "float32") # converts to float32


            # Make them nicely concatable in custom_collate_fn
            img = img.unsqueeze(0)
            img_name_no_suffix = [img_name_no_suffix]

            if self.patchify:
                # these become a tensor like [num_of_patches_from_img, channels, height, width]
                img = get_random_patches([img], patch_shape=self.patch_shape, num_of_patches_from_img=self.num_of_patches_from_img, 
                                                        prob_zero_patch_resample=None, resample_gt=None)
                
                img = img[0]   # so it stops bein a list with one element

                old_img_name_no_suffix = img_name_no_suffix[0]
                img_name_no_suffix = [f"{old_img_name_no_suffix}_{i}" for i in range(self.num_of_patches_from_img)]




            return {'images': img, 'img_names': img_name_no_suffix}
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e



def save_preds_collate_fn(batch):
    
    images = torch.concat([item['images'] for item in batch], dim=0)

    img_names = []
    for item in batch:
        img_names.extend(item['img_names']) 
    return {'images': images, 'img_names': img_names}


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

    train_dataset = SavePredsDataset(filepath=data_path, split='save_preds', **dataloading_args)

    for i in range(10):
        res = train_dataset[0]
        py_log_always_on.log_manual(MY_LOGGER, img=res['images'], names=res['img_names'])
        save_img_quick_figs(res['images'][0, :, :, :], f"test_{i}.png")
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
