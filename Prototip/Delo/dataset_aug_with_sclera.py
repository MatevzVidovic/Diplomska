



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


from helper_img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs

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

        self.images_without_mask = [] # seznam slik, ki nimajo maske
        

        images_with_masks = []
        
        all_images = os.listdir(osp.join(self.filepath,'Images'))
        masks_folder = osp.join(self.filepath,'Masks')
        all_images.sort()



        for file in all_images:
            
            file_without_suffix = file.strip(".jpg")

            if self.mask_exists(masks_folder, file_without_suffix):
                images_with_masks.append(file_without_suffix)


        self.list_files = images_with_masks
        self.testrun = testrun

        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8,8))

        #summary
        print('summary for ' + str(split))
        print('valid images: ' + str(len(self.list_files)))


    def __len__(self):
        real_size = len(self.list_files)
        if self.testrun:
            return min(self.testrun_length, real_size)
        return real_size
    
    def mask_exists(self, mask_folder_path, file_name_no_suffix):
        mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
        return osp.exists(mask_filename)

    def get_mask(self, mask_folder_path, file_name_no_suffix) -> np.array:

        mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
        try:
            mask = Image.open(mask_filename).convert("RGB")
            return mask

        except FileNotFoundError:
            print('file not found: ' + str(mask_filename))
            self.images_without_mask.append(file_name_no_suffix)
            return None

    @py_log.autolog(passed_logger=MY_LOGGER)
    def __getitem__(self, idx):

        try:
            
            img_name = self.list_files[idx]
            image_path = osp.join(self.filepath,'Images',f'{img_name}.jpg')

            img = Image.open(image_path).convert("RGB")

            middle_resize = False
            if middle_resize:
                preprocessing_resize_size = (1024, 1024) # img.size # or just (1024, 1024)

                img = img.resize(preprocessing_resize_size, Image.LANCZOS) #ali Image.BICUBIC ali 



            # Since our data augmentation should simulate errors and changes in how the image is taken, we should do gamma correction and clahe after the augmentation.
            # Because we should first simulate the errors, that is then what our taken img would have been,
            # and then we do our preprocessing (correction) from there.
            # We pretend that the data augmentations were actually pictures that we took.




            mask_path = osp.join(self.filepath,'Masks')
            file_name_no_suffix = self.list_files[idx]
            mask = self.get_mask(mask_path, file_name_no_suffix) # is of type Image.Image
            
            if middle_resize:
                mask = mask.resize(preprocessing_resize_size, Image.LANCZOS)


            img = smart_conversion(img, 'Image', 'uint8')
            mask = smart_conversion(mask, 'Image', 'uint8')


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])



            show_testrun = False

            # Testing
            if show_testrun:

                img_test = img
                mask_test = mask
                
                py_log_always_on.log_time(MY_LOGGER, "test_da")

                # py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Original image")
                # py_log_always_on.log_time(MY_LOGGER, "test_da")
                img_test, mask_test = random_horizontal_flip(img_test, mask_test, prob=1.0)
                # py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Horizontal flip")
                py_log_always_on.log_time(MY_LOGGER, "test_da")

                # Has to be here, because at this point the img is surely still a PIL image, so .size() is correct
                ksize = (img_test.size[0]//20)
                ksize = ksize+1 if ksize % 2 == 0 else ksize
                # py_log_always_on.log_time(MY_LOGGER, "test_da")
                img_test = random_gaussian_blur(img_test, possible_sigma_vals_list=np.linspace(1, 10, 50), ker_size=ksize, prob=1.0)
                # py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Gaussian blur")
                py_log_always_on.log_time(MY_LOGGER, "test_da")

                # py_log_always_on.log_time(MY_LOGGER, "test_da")
                img_test, mask_test = random_rotation(img_test, mask_test, max_angle_diff=15, prob=1.0)
                # py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Rotation")
                py_log_always_on.log_time(MY_LOGGER, "test_da")

                # py_log_always_on.log_time(MY_LOGGER, "test_da")
                img_test, mask_test = zoom_in_somewhere(img_test, mask_test, max_scale_percent=0.5, prob=1.0)
                # py_log_always_on.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img_test, mask_test], title="Zoom blur")
                py_log_always_on.log_time(MY_LOGGER, "test_da")

                # py_log_always_on.log_time(MY_LOGGER, "test_da")

                if True:
                    img = img_test
                    mask = mask_test





            

            if not show_testrun and self.split == 'train':

                img, mask = random_horizontal_flip(img, mask, prob=0.5)

                # Has to be here, because at this point the img is surely still a PIL image, so .size() is correct
                ksize = (img.size[0]//20)
                ksize = ksize+1 if ksize % 2 == 0 else ksize
                img = random_gaussian_blur(img, possible_sigma_vals_list=np.linspace(1, 10, 50), ker_size=ksize, prob=0.1)

                img, mask = random_rotation(img, mask, max_angle_diff=30, prob=0.5)

                img, mask = zoom_in_somewhere(img, mask, max_scale_percent=0.3, prob=0.7)





            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); show_image([img, mask])






            # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
            img = smart_conversion(img, 'ndarray', 'uint8')
            mask = smart_conversion(mask, 'ndarray', 'uint8')


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); show_image([img, mask])


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

            # performing the necessary resizing
            img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (self.output_width, self.output_height), interpolation=cv2.INTER_LANCZOS4)

            if show_testrun:
                py_log_always_on.log_time(MY_LOGGER, "test_da")
            
            # Making the mask binary, as it is meant to be.
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            mask[mask < 127] = 0
            mask[mask >= 127] = 1

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])

            # Conversion to standard types that pytorch can work with.
            # target tensor is long, because it is a classification task (in regression it would also be float32)
            img = smart_conversion(img, "tensor", "float32") # converts to float32
            mask = smart_conversion(mask, 'tensor', "uint8").long() # converts to int64

            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math", "hist"]); show_image([img, mask])


            # mask mustn't have channels. It is a target, not an image.
            # And since the output of our network is (batch_size, n_classes, height, width), our target has to be (batch_size, height, width).
            # So here we need to return (height, width) mask, not (height, width, 1) mask.
            mask = mask.squeeze() # This function removes all dimensions of size 1 from the tensor

            if show_testrun:
                py_log_always_on.log_time(MY_LOGGER, "test_da")


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])



            # while True:
            #     plt.pause(0.1)



            return {'images': img, 'masks': mask, 'img_names': img_name}
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER)
            raise e



def custom_collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    masks = torch.stack([item['masks'] for item in batch])
    img_names = [item['img_names'] for item in batch]  # Collect image names into a list
    return {'images': images, 'masks': masks, 'img_names': img_names}



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
