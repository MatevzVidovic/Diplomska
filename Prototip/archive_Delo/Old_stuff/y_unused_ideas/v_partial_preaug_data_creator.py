
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


python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024, var_blacklist=["tree_ix_2_module", "mask_path"])
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path)



from img_augments import horizontal_flip, rotation, gaussian_blur


from helper_img_and_fig_tools import smart_conversion, show_image, save_img_quick_figs, save_img

from enum import Enum

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




transform = transforms.Compose(
    [
     transforms.Normalize([0.5], [0.5])
    ])







def mask_exists(mask_folder_path, file_name_no_suffix):
    mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')
    return osp.exists(mask_filename)

def get_mask(mask_folder_path, file_name_no_suffix) -> np.array:

    mask_filename = osp.join(mask_folder_path, file_name_no_suffix + '.png')

    mask = Image.open(mask_filename).convert("RGB")
    return mask






import argparse

if __name__ == "__main__":


    # Suggested plan:
    # In my dataset script, 40% of the time I do a zoom between 0.0 and 0.2.
    # This means that I take 1.0 to 0.8 of the image in each dimension.

    # I would like it that when I take the image from my partially augmented dataset, and then I do a zoom,
    # and then I resize to the final dimensions, that this final resizing isn't trying to increase the size of the image.
    # Because resizing in the increasing direction is bad. It tries to make up information.

    # So, since the maximum retention is 0.8, the minimum of the dimension after zoom is  x*0.8.
    # I have decided to have the final dimensions that the model uses to be 2048x1024
    # So the partial augmentation width should be:    x * 0.8 = 2480 ==>   2048 / 0.8 = 2560
    # And for height: 1024 / 0.8 = 1280
    

    # cores probs help because of C vectorization in numpy operations
    # srun -c 30 --gpus=A100  python3 v_partial_preaug_data_creator.py --ow 1500 --oh 1000

    parser = argparse.ArgumentParser(description='Data Augmentation for Vein Sclera Segmentation')
    parser.add_argument('--ow', type=int, default=2560, help='Output width of the image')
    parser.add_argument('--oh', type=int, default=1280, help='Output height of the image')
    parser.add_argument('--fp', type=str, default='./vein_sclera_data', help='Folder path of the dataset')
    parser.add_argument('--dafp', type=str, default='./vein_sclera_data_partial_preaug', help='Augmented data folderpath. To this name the ow and oh get appended.')
    parser.add_argument('--split', type=str, default='train', help='Split of the dataset')

    args = parser.parse_args()
    
    folderpath = args.fp
    da_folderpath = args.dafp
    split = args.split
    
    output_width = args.ow
    output_height = args.oh

    da_folderpath = f"{da_folderpath}_{output_width}x{output_height}"

        
    images_without_mask = []
    images_with_masks = []
    
    imgs_path = osp.join(folderpath, split, 'Images')
    masks_path = osp.join(folderpath, split, 'Masks')
    all_images = os.listdir(imgs_path)
    all_images.sort()



    for file in all_images:
        
        file_without_suffix = file.strip(".jpg")

        if mask_exists(masks_path, file_without_suffix):
            images_with_masks.append(file_without_suffix)
        else:
            images_without_mask.append(file_without_suffix)


    #summary
    print('summary for ' + str(split))
    print('valid images: ' + str(len(images_with_masks)))


    try:
        for ix, img_name in enumerate(images_with_masks):


        

            image_path = osp.join(imgs_path, f"{img_name}.jpg")

            img = Image.open(image_path).convert("RGB")



            # Since our data augmentation should simulate errors and changes in how the image is taken, we should do gamma correction and clahe after the augmentation.
            # Because we should first simulate the errors, that is then what our taken img would have been,
            # and then we do our preprocessing (correction) from there.
            # We pretend that the data augmentations were actually pictures that we took.




            file_name_no_suffix = img_name
            mask = get_mask(masks_path, file_name_no_suffix) # is of type Image.Image


            base_img = smart_conversion(img, 'Image', 'uint8')
            base_mask = smart_conversion(mask, 'Image', 'uint8')


            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])


            # Idea:
            # On one side we have on-the-fly augmentations, which are done during training.
            # But augmenting a 3000x2000 image is slow. 3 ops of 1.5 secs.
            # So what if we augment the data beforehand, and then just load it in during training?
            # But then:
            #    - The augmentations are static (instead of always getting a slightly differently augmented image, we only have one random choice of augmentation saved)
            #   - It's hard to keep impact of non-augmented images high: We have an original image. We tak 3 rotated ones from it. For each one of them we do 9 different zooms.
            #     So we have 27 images from one original. If we have 100 originals, we have 2700 images. But only 100 of them are original. So the impact of the original is low.
            #     And this is a problem if the augmented images are not perfectly what we want to train on. Eg we are interested at a specific scale, but we do zooms because our dataset is too small.
            #     So this helps, but if we have such an imbalance of actual correct scale (original) and incorrect scale (augmented), then we will barely be learning the correct scale.
            #     So we need to keep the impact of the original high. And partial augmentation does that, see for yourself.

            # So we want to do preaugmentations only for some augmentations that are still quite close to the original.


            # Pipeline of partial augmentations:
            # - mirror or dont
            # - slight gauss or dont
            # - if gauss and otherwise also very augmented, randomly abort 90% of the time, so I don't have that many gaussed up images.
            # - rotate for different fixed angles
            # - then save the image and mask

            # And on the fly we will only sometimes be doing zooms.
            # And only sometimes be doing smaller random rotations.


            # (1.25 is a guesstimate of how often we actually do gauss. If we do it 0.25, then we get 1.0 for not gauss, and so togehter it's 1.25)
            # For each original image, we will have 2*(2*1.25)*3 = 15 images.
            # So if original is 0.5GB, we will have 7.5GB of data.
            # We had 89 imgs, now we have 1335 imgs. This is a nice number.



            for ix_0, is_mirror in enumerate([False, True]):
                for ix_1, is_gauss in enumerate([False, True]):
                    for ix_2, rotation_angle in  enumerate([None, 10, 20]):

                        img = base_img.copy()
                        mask = base_mask.copy()


                        # If the img is already augmented with both rotation and zoom, make it very rare to be gaussed also. Those imgs are already very different,
                        # and we don't want too much gaussing, because too different from our original dataset.
                        if is_gauss and (rotation_angle is not None) and np.random.random() < 0.9:
                            continue


                        if is_mirror:

                            img, mask = horizontal_flip(img, mask)

                        if is_gauss:

                            # Has to be here, because at this point the img is surely still a PIL image, so .size() is correct
                            ksize = (img.size[0]//20)
                            ksize = ksize+1 if ksize % 2 == 0 else ksize
                            img = gaussian_blur(img, possible_sigma_vals_list=np.linspace(1, 5, 50), ker_size=ksize)


                        if rotation_angle is not None:
                            img, mask = rotation(img, mask, angle= ((2*rotation_angle)*np.random.random() - rotation_angle), rotate_type="shrink")
                        

                        # Converting img and mask to correct dimensions, binarization, types, and removing unnecessary dimension
                        img = smart_conversion(img, 'ndarray', 'uint8')
                        mask = smart_conversion(mask, 'ndarray', 'uint8')

                        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

                        # performing the necessary resizing
                        img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                        mask = cv2.resize(mask, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                        
                        # Making the mask binary, as it is meant to be.
                        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

                        mask[mask < 127] = 0
                        mask[mask >= 127] = 255




                        # da_suffix = f"_{is_mirror}{is_gauss}{rotation_angle}{zoom_level}{zoom_position}"
                        da_suffix = f"_{ix_0}{ix_1}{ix_2}"
                        new_img_name = f"{img_name}{da_suffix}"
                        # Mask has same name as image, just different file extension.
                        new_folderpath = osp.join(da_folderpath, split)
                        new_img_path = osp.join(new_folderpath, 'Images')
                        new_mask_path = osp.join(new_folderpath, 'Masks')

                        os.makedirs(new_folderpath, exist_ok=True)
                        os.makedirs(osp.join(new_folderpath, 'Images'), exist_ok=True)
                        os.makedirs(osp.join(new_folderpath, 'Masks'), exist_ok=True)

                        # # Saving the image and mask
                        # cv2.imwrite(new_img_path, img)
                        # cv2.imwrite(new_mask_path, mask)
                        
                        img = smart_conversion(img, 'Image', 'uint8')
                        mask = smart_conversion(mask, 'Image', 'uint8')
                        

                        save_img(img, new_img_path, new_img_name + '.jpg')
                        save_img(mask, new_mask_path, new_img_name + '.png')

                        # input("Press Enter to continue...")

                        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])



                        # while True:
                        #     plt.pause(0.1)


        
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e
