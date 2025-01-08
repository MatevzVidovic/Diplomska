



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

import numpy as np
import torch
from torch.utils.data import Dataset
import os
# import random
from PIL import Image
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











class IrisDataset(Dataset):
    def __init__(self, filepath, split='train', transform=None, n_classes=4, testrun=False, clipLimit=1.5, **kwargs):
        
        self.transform = transform
        self.filepath= osp.join(filepath, split)
        
        self.input_width = kwargs['input_width']
        self.input_height = kwargs['input_height']
        self.output_width = kwargs['output_width']
        self.output_height = kwargs['output_height']

        self.testrun_length = kwargs['testrun_size']
        
        self.split = split
        self.classes = n_classes

        self.images_without_mask = []
        

        images_with_masks = []
        
        image_names = os.listdir(osp.join(self.filepath,'Images'))
        image_names.sort()
        
        to_veins = osp.join(self.filepath,'Veins')
        to_scleras = osp.join(self.filepath,'Scleras')



        for img_name in image_names:
            
            img_name_without_suffix = img_name.strip(".jpg")

            if self.masks_exist(to_veins, to_scleras, img_name_without_suffix):
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
    
    def masks_exist(self, veins_folder_path, scleras_folder_path, img_name_name_no_suffix):
        vein_mask_filename = osp.join(veins_folder_path, img_name_name_no_suffix + '.png')
        sclera_mask_filename = osp.join(scleras_folder_path, img_name_name_no_suffix + '_sclera.png')
        masks_exist = osp.exists(vein_mask_filename) and osp.exists(sclera_mask_filename)
        return masks_exist



    @py_log.autolog(passed_logger=MY_LOGGER)
    def __getitem__(self, idx):

        try:
            
            img_name = self.img_names_no_suffix[idx]
            image_path = osp.join(self.filepath,'Images',f'{img_name}.jpg')

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            veins_path = osp.join(self.filepath,'Veins')
            img_name_no_suffix = self.img_names_no_suffix[idx]
            veins_filename = osp.join(veins_path, img_name_no_suffix + '.png')
            veins = cv2.imread(veins_filename)
            veins = cv2.cvtColor(veins, cv2.COLOR_BGR2RGB) # have to be this so that transformations are easier - the same as for the original image


            scleras_path = osp.join(self.filepath,'Scleras')
            img_name_no_suffix = self.img_names_no_suffix[idx]
            scleras_filename = osp.join(scleras_path, img_name_no_suffix + '_sclera.png')
            scleras = cv2.imread(scleras_filename)
            scleras = cv2.cvtColor(scleras, cv2.COLOR_BGR2RGB) # have to be this so that transformations are easier


            img = smart_conversion(img, 'ndarray', 'uint8')
            veins = smart_conversion(veins, 'ndarray', 'uint8')
            scleras = smart_conversion(scleras, 'ndarray', 'uint8')

            masks = [veins, scleras]


            save_img_ix = 0

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_before_da")

            show_testrun = False

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


            #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da_converted")


            # #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LANCZOS4)
            masks = [cv2.resize(mask, (self.input_width, self.input_height), interpolation=cv2.INTER_LANCZOS4) for mask in masks]

            if show_testrun:
                py_log_always_on.log_time(MY_LOGGER, "test_da")


            
            veins, sclera = masks

            veins = cv2.cvtColor(veins, cv2.COLOR_RGB2GRAY)
            veins[veins <= 127] = 0
            veins[veins > 127] = 1

            sclera = cv2.cvtColor(sclera, cv2.COLOR_RGB2GRAY)
            sclera[sclera <= 127] = 0
            sclera[sclera > 127] = 255
            

            #py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); save_img_ix += 1; save_imgs_quick_figs([img, *masks], f"{idx}_{save_img_ix}_after_da_resized")

            # stack img and sclera to get a 4-channel image
            sclera = np.expand_dims(sclera, axis=2)
            # img = np.concatenate([img, sclera], axis=2)
            img = img * sclera
            # Conversion to standard types that pytorch can work with.
            img = smart_conversion(img, "tensor", "float32") # converts to float32
            sclera = smart_conversion(sclera, 'tensor', "float32") # converts to float32

            # target tensor is long, because it is a classification task (in regression it would also be float32)
            veins = smart_conversion(veins, 'tensor', "uint8").long() # converts to int64
            
            veins_for_show = veins.clone() * 255
            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); save_imgs_quick_figs([img, veins_for_show, sclera], f"{idx}_{save_img_ix}_final")

            # veins mustn't have channels. It is a target, not an image.
            # And since the output of our network is (batch_size, n_classes, height, width), our target has to be (batch_size, height, width).
            # So here we need to return (height, width) veins, not (height, width, 1) veins.
            veins = veins.squeeze() # This function removes all dimensions of size 1 from the tensor

            if show_testrun:
                py_log_always_on.log_time(MY_LOGGER, "test_da")


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])



            # while True:
            #     plt.pause(0.1)



            return {'images': img, 'masks': veins, "scleras": sclera , 'img_names': img_name}
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e



def custom_collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    masks = torch.stack([item['masks'] for item in batch])
    scleras = torch.stack([item['scleras'] for item in batch])
    img_names = [item['img_names'] for item in batch]  # Collect image names into a list
    return {'images': images, 'masks': masks, "scleras" : scleras, 'img_names': img_names}



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
    

        "input_width" : 2048,
        "input_height" : 1024,
        "output_width" : 2048,
        "output_height" : 1024,

        # "input_width" : 256,
        # "input_height" : 256,
        # "output_width" : 256,
        # "output_height" : 256,
        
        "transform" : None,
        "n_classes" : 2,

    }


    
    data_path = "vein_sclera_data"

    train_dataset = IrisDataset(filepath=data_path, split='train', **dataloading_args)
#    for i in range(1000):
    img, mask = train_dataset[0]
    # show_image([img, mask])
    save_img_quick_figs(img, "ds_img.png")
    save_img_quick_figs(mask, "ds_mask.png")
    print(img.shape)
    print(mask.shape)


# %%
