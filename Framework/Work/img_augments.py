




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



from y_helpers.helper_img_and_fig_tools import smart_conversion, show_image

import numpy as np
import os
from PIL import Image
import cv2

np.random.seed(7)






# Horizontal flip and gaussian blur funtions:

def random_horizontal_flip(img, masks, prob=0.5):
    # Takes PIL img as input, returns PIL img.
    # If input not PIL img, automatically transforms it to PIL img.

    try:
        
        if np.random.random() > prob:
            return img, masks


        img = smart_conversion(img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]

        # aug_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # aug_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        aug_img = cv2.flip(img, 1)
        aug_masks = [cv2.flip(mask, 1) for mask in masks]

        #py_log_always_on.log_time(MY_LOGGER, "test_da")
        return aug_img, aug_masks
    
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e

def horizontal_flip(img, masks):
    aug_img, aug_masks = random_horizontal_flip(img, masks, prob=1.1)
    return aug_img, aug_masks


def random_gaussian_blur(img, possible_sigma_vals_list=range(2, 7), ker_size=7, prob=0.2):
    # Takes np.array img as input. Returns np.array.
    # If input not np.array, automatically transforms it to np.array.

    try:
            
        if np.random.random() > prob:
            return img
        
        img = smart_conversion(img, 'ndarray', 'uint8')

        sigma_ix = np.random.randint(len(possible_sigma_vals_list))
        sigma_value = possible_sigma_vals_list[sigma_ix]

        aug_img = cv2.GaussianBlur(img, (ker_size, ker_size), sigma_value)

        #py_log_always_on.log_time(MY_LOGGER, "test_da")
        return aug_img
    
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e
    

def gaussian_blur(img, possible_sigma_vals_list=range(2, 7), ker_size=7):
    aug_img = random_gaussian_blur(img, possible_sigma_vals_list, ker_size, prob=1.1)
    return aug_img







# Random rotation block of functions:


def maxHist(row):
    # Create an empty stack. The stack holds
    # indexes of hist array / The bars stored
    # in stack are always in increasing order
    # of their heights.

    try:
        stack = []

        # Top of stack
        top_val = 0

        # Initialize max area in current
        max_area = 0
        height_of_max_area = 0
        left_index = 0
        right_index = 0
        # row (or histogram)

        area = 0  # Initialize area with current top

        # Run through all bars of given
        # histogram (or row)
        i = 0
        while (i < len(row)):

            # If this bar is higher than the
            # bar on top stack, push it to stack
            if (len(stack) == 0) or (row[stack[-1]] <= row[i]):
                stack.append(i)
                i += 1

            # This will pop stuff from stack until we get a bar that is lower than the currently next bar.
            else:

                # If this bar is lower than top of stack,
                # then calculate area of rectangle with
                # stack top as the smallest (or minimum
                # height) bar. 'i' is 'right index' for
                # the top and element before top in stack
                # is 'left index'
                top_val = row[stack.pop()]
                area = top_val * i

                if len(stack) > 0:
                    # i is the ix of the curr next bar
                    # stack[-1] + 1 is the ix of the currently popped bar
                    area = top_val * (i - (stack[-1] + 1))

                if area > max_area:
                    max_area = area
                    height_of_max_area = top_val
                    left_index = stack[-1] + 1 if stack else 0
                    right_index = i

        # Now pop the remaining bars from stack
        # and calculate area with every popped
        # bar as the smallest bar
        while (len(stack)):
            top_val = row[stack.pop()]
            area = top_val * i
            if (len(stack)):
                area = top_val * (i - stack[-1] - 1)

            if area > max_area:
                max_area = area
                height_of_max_area = top_val
                left_index = stack[-1] + 1 if stack else 0
                right_index = i - 1

        return max_area, left_index, right_index, height_of_max_area
    
    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


def max_histogram_area_with_indices(heights):

    try:
        stack = []
        max_area = 0
        left_index = 0
        right_index = 0
        height_index = 0
        index = 0

        while index < len(heights):
            if len(stack) == 0 or heights[stack[-1]] <= heights[index]:
                stack.append(index)
                index += 1
            else:
                top_of_stack = stack.pop()
                area = (heights[top_of_stack] *
                        ((index - stack[-1] - 1) if stack else index))
                if area > max_area:
                    max_area = area
                    height_index = top_of_stack
                    left_index = stack[-1] + 1 if stack else 0
                    right_index = index - 1

        while stack:
            top_of_stack = stack.pop()
            area = (heights[top_of_stack] *
                    ((index - stack[-1] - 1) if stack else index))
            if area > max_area:
                max_area = area
                height_index = top_of_stack
                left_index = stack[-1] + 1 if stack else 0
                right_index = index - 1

        return max_area, left_index, right_index, height_index

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


def maximal_rectangle_with_indices(matrix):
    
    try:
        max_area = 0
        max_coords = None
        heights = [0] * len(matrix[0])

        for row_index, row in enumerate(matrix):
            for i in range(len(row)):
                heights[i] = heights[i] + 1 if row[i] == 1 else 0

            # area, left, right, height = max_histogram_area_with_indices(heights)
            area, left, right, height = maxHist(heights)
            if area > max_area:
                max_area = area
                max_coords = (row_index - height + 1, left, row_index, right)

        return max_area, max_coords

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e

def crop_to_nonzero_in_fourth_channel(img, masks, crop="all_zero"):

    # crop can be "all_zero" or "hug_nonzero"
    

    # If you want to crop the image to the non-zero values in the fourth channel of the image and mask.
    # Zero values still remain, the frame of img just hugs the non-zero values now.

    try:

        if crop == "hug_nonzero":
        
            # Assuming the last channel is the one added with ones
            non_zero_indices = np.where(img[..., -1] != 0)
            # Note: img[..., -1] is the same as img[:, :, -1]
            
            min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
            min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
            
            # Crop the image and mask
            cropped_img = img[min_y:max_y+1, min_x:max_x+1]
            cropped_masks = [mask[min_y:max_y+1, min_x:max_x+1] for mask in masks]
        


        # This squeezes the frame until there are no zero values in the fourth channel anywhere:
        elif crop == "all_zero":
            # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img[...,:-1], mask])


            # This doesn't give you the biggest AREA!!!
            # At least not in general.
            # It just works well enough.
            # And in our case of a rotated rect, it might even work.

            method = 1
            if method == 0:

                # Find the rect that is inside the area of 1s in the fourth channel
                # that has the maximum area.

                # Since this is a rotated rectangle, this should work:
                # Go from the right. Keep checking the pixel in the top row and the bottom row.
                # As soon as you one of them is 1, crop to that distance.
                # Now do the same from the left.
                # Now you have a new cropped image, which was cropped the minimum amount that you would have to anyway from both sides.

                # Now on the new cropped image go from top down and bottom up, but both pixels have to be 1 and that's when you stop.

                y_min = 0
                y_max = img.shape[0] - 1
                x_min = 0
                x_max = img.shape[1] - 1
                
                # for fraction_of_img == 100,
                # if the img dim  size is <= 99, launch step is 1, 
                # so it can't possibly skip an area of 1s, and it is performance neutral.
                # if the img dim size is 1000, launch step is 11, so it most likely won't skip over an area of 1s.
                # It will, however, make the code 10x faster.
                # So even if we did skip an area of 1s, we didn't waste much time.

                # fraction_of_img == 100 might even be too conservative.
                # The thing is, the middle pixel in the uppermost row is bound to be 1. And a bunch of pixels around it too.

                # We don't want fraction_of_img to be too small from the speedup perspective.
                # Because then we are doing a lot of if-checks for nothing because we are constantly going into 1s.

                # fraction_of_img = 100
                fraction_of_img = 50 # 20x speedup

                # ensure launch step is at least 1, so we aren't doing these if-checks
                # for nothing - at worst it's neutral (1 if-check for movement of 1)
                launch_step = img.shape[0] // fraction_of_img + 1 # small enough that it doesn't skip over the 1s to the other side of 0s
                while img[y_min, x_min, -1] == 0 and img[y_max, x_min, -1] == 0:
                    x_min += 1
                    if img[y_min, x_min + launch_step, -1] == 0 and img[y_max, x_min + launch_step, -1] == 0:
                        x_min += launch_step
                    else:
                        launch_step = (launch_step // 2) + 1 # so it is always at least 1 and the if check isn't for nothing
                
                launch_step = img.shape[0] // fraction_of_img + 1
                while img[y_min, x_max, -1] == 0 and img[y_max, x_max, -1] == 0:
                    x_max -= 1
                    if img[y_min, x_max - launch_step, -1] == 0 and img[y_max, x_max - launch_step, -1] == 0:
                        x_max -= launch_step
                    else:
                        launch_step = (launch_step // 2) + 1



                # Now we have the new leftmost and rightmost column. It is as if we cropped it from the left and right.
                
                launch_step = img.shape[1] // fraction_of_img + 1
                while img[y_min, x_min, -1] == 0 or img[y_min, x_max, -1] == 0:
                    y_min += 1
                    if img[y_min + launch_step, x_min, -1] == 0 or img[y_min + launch_step, x_max, -1] == 0:
                        y_min += launch_step
                    else:
                        launch_step = (launch_step // 2) + 1
                
                launch_step = img.shape[1] // fraction_of_img + 1
                while img[y_max, x_min, -1] == 0 or img[y_max, x_max, -1] == 0:
                    y_max -= 1
                    if img[y_max - launch_step, x_min, -1] == 0 or img[y_max - launch_step, x_max, -1] == 0:
                        y_max -= launch_step
                    else:
                        launch_step = (launch_step // 2) + 1
            


                # If the area is too small, we probably made a mistake with the launch_step.
                # (We stepped over the 1s to the other side of the 0s.)
                # So we just do it the slow way.
                a = (y_max - y_min)
                b = (x_max - x_min)
                if  a < 0 or b < 0 or a*b < 0.5 * img.shape[0] * img.shape[1]:

                    y_min = 0
                    y_max = img.shape[0] - 1
                    x_min = 0
                    x_max = img.shape[1] - 1
                    
                    while img[y_min, x_min, -1] == 0 and img[y_max, x_min, -1] == 0:
                        x_min += 1
                    
                    while img[y_min, x_max, -1] == 0 and img[y_max, x_max, -1] == 0:
                        x_max -= 1

                    # Now we have the new leftmost and rightmost column. It is as if we cropped it from the left and right.
                    
                    while img[y_min, x_min, -1] == 0 or img[y_min, x_max, -1] == 0:
                        y_min += 1
                    
                    while img[y_max, x_min, -1] == 0 or img[y_max, x_max, -1] == 0:
                        y_max -= 1
            

            elif method == 1:
                
                try:
                    y_min = 0
                    y_max = img.shape[0] - 1
                    x_min = 0
                    x_max = img.shape[1] - 1
                    
                    while img[y_min, x_min, -1] == 0 and img[y_max, x_min, -1] == 0:
                        x_min += 1
                    
                    while img[y_min, x_max, -1] == 0 and img[y_max, x_max, -1] == 0:
                        x_max -= 1

                    # Now we have the new leftmost and rightmost column. It is as if we cropped it from the left and right.
                    
                    while img[y_min, x_min, -1] == 0 or img[y_min, x_max, -1] == 0:
                        y_min += 1
                    
                    while img[y_max, x_min, -1] == 0 or img[y_max, x_max, -1] == 0:
                        y_max -= 1
                except:
                    # An error in line 395 (the y_min += 1) would cause an error here. I guess sometimes it just doesn't go through nicely.
                    print("Error in cropping the image to the non-zero values in the fourth channel. Returned the original image.")
                    return img, masks, True

            # Works but uses more python so it's probably slower:
            else:


                # We use the maximum rectangle algorithm for this, which uses the maximum histogram algorithm.
                # https://www.geeksforgeeks.org/maximum-size-rectangle-binary-sub-matrix-1s/
                # https://www.youtube.com/watch?v=zx5Sw9130L0 !!!!!!!!
                # https://www.youtube.com/watch?v=ZmnqCZp9bBs
                # https://www.youtube.com/watch?v=g8bSdXCG-lA 

                # I tried with my own algorithm, but this one is faster.

                _, (y_min, x_min, y_max, x_max) = maximal_rectangle_with_indices(img[..., -1])





            cropped_img = img[y_min:y_max+1, x_min:x_max+1]
            cropped_masks = [mask[y_min:y_max+1, x_min:x_max+1] for mask in masks]
            


        return cropped_img, cropped_masks, False

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e
    

def random_rotation(inp_img, inp_masks, max_angle_diff=15, mean_angle=0, rotate_type="shrink", prob=0.2):
    
    # THE SIZE OF THE IMAGE MIGHT BE SMALLER AFTER THIS FUNCTION!!!!!

    # IF NOT, IT INTRODUCES BLACK PIXELS!!!!!

    # CREATES BLURRYNESS!!!!! (because of interpolation)

    # max_angle is actually how much the randomness of tha angle deviates from the mean_angle.


    # The size of the returned image MAY NOT BE THE SAME as the input image.
    # If using "shrink", the size of the returned image will be smaller.
    
    # Rotating the image around its center introduces black pixels.
    # The size of the image remains the same in the method we use
    # (could also get bigger if we wanted to retain all pixels of the original img).
    # If "shrink" we shrink the frame of the image from each direction until there are no newly introduced black pixels.
    # (Don't worry, it doesn't actually look at blackness of pixels to crop, it looks at the fourth channel we added.
    # So if your image is mostly black at the edges, don't worry, we won't crop into your img.)

    try:

        if np.random.random() > prob:
            return inp_img, inp_masks
        
        # These give coppies anyway, so don't worry about changing them.
        img = smart_conversion(inp_img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in inp_masks]
        
        
        # Add a channel with ones
        ones_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype)
        img = np.concatenate((img, ones_channel), axis=-1)
        # mask = np.concatenate((mask, ones_channel), axis=-1)
        

        # Randomly choose an angle between -max_angle and max_angle
        angle = np.random.uniform(-max_angle_diff, max_angle_diff) + mean_angle

        # Get the image center
        center = (img.shape[1] // 2, img.shape[0] // 2)

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Perform the rotation
        aug_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        aug_masks = [cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)) for mask in masks]

        #py_log_always_on.log_time(MY_LOGGER, "test_da")

        # with np.printoptions(threshold=np.inf):
        #     # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([aug_img[...,:-1], aug_mask])

        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([aug_img[...,:-1], aug_mask])

        if rotate_type == "shrink":
            aug_img, aug_masks, error = crop_to_nonzero_in_fourth_channel(aug_img, aug_masks, crop="all_zero")
        
        if error:
            return inp_img, inp_masks
        

        # Remove the last channel
        aug_img = aug_img[..., :-1]


        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])

        #py_log_always_on.log_time(MY_LOGGER, "test_da")
        return aug_img, aug_masks

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e
    

def rotation(img, masks, angle, rotate_type="shrink"):
    aug_img, aug_masks = random_rotation(img, masks, max_angle_diff=0, mean_angle=angle, rotate_type=rotate_type, prob=1.1)
    return aug_img, aug_masks










def zoom_and_offset(img, masks, scale_percent, offset_percent_y=0.5, offset_percent_x=0.5):

    try:
        # This function zooms in on a random part of the image.


        # Takes np.array img as input. Returns np.array.
        # If input not np.array, automatically transforms it to np.array.

        img = smart_conversion(img, 'ndarray', 'uint8')
        masks = [smart_conversion(mask, 'ndarray', 'uint8') for mask in masks]

        remain_percent = 1 - scale_percent
        
        

        # How many picels we keep in each direction:
        vert_pix_num = int(img.shape[0] * remain_percent)
        horiz_pix_num = int(img.shape[1] * remain_percent)

        # In the very extreme case, we should keep at least one pixel:
        if vert_pix_num <= 0:
            vert_pix_num = 1
        if horiz_pix_num <= 0:
            horiz_pix_num = 1
        
        max_offset_vert = img.shape[0] - vert_pix_num
        max_offset_horiz = img.shape[1] - horiz_pix_num

        offset_vert = int(offset_percent_y * max_offset_vert)
        offset_horiz = int(offset_percent_x * max_offset_horiz)
        
        aug_img = img[offset_vert:offset_vert+vert_pix_num, offset_horiz:offset_horiz+horiz_pix_num, :]
        aug_masks = [mask[offset_vert:offset_vert+vert_pix_num, offset_horiz:offset_horiz+horiz_pix_num, :] for mask in masks]

        # aug_img = img[vert_pix_num:-vert_pix_num, horiz_pix_num:-horiz_pix_num, :]
        # aug_mask = mask[vert_pix_num:-vert_pix_num, horiz_pix_num:-horiz_pix_num, :]

        # py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"])


        #py_log_always_on.log_time(MY_LOGGER, "test_da")
        return aug_img, aug_masks

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


def zoom_in_somewhere(img, masks, max_scale_percent=0.2, prob=0.2):

    # This function zooms in on a random part of the image.


    # Takes np.array img as input. Returns np.array.
    # If input not np.array, automatically transforms it to np.array.

    if np.random.random() > prob:
        return img, masks
    

    scale_percent = np.random.uniform(0, max_scale_percent)

    offset_percent_x = np.random.uniform(0, 1)
    offset_percent_y = np.random.uniform(0, 1)

    aug_img, aug_masks = zoom_and_offset(img, masks, scale_percent, offset_percent_x, offset_percent_y)

    return aug_img, aug_masks
