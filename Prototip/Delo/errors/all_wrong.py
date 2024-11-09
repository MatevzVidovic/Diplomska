


def crop_to_nonzero_in_fourth_channel(img, mask, crop="all_zero"):

    # crop can be "all_zero" or "hug_nonzero"
    

    # If you want to crop the image to the non-zero values in the fourth channel of the image and mask.
    # Zero values still remain, the frame of img just hugs the non-zero values now.

    if crop == "hug_nonzero":
    
        # Assuming the last channel is the one added with ones
        non_zero_indices = np.where(img[..., -1] != 0)
        # Note: img[..., -1] is the same as img[:, :, -1]
        
        min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
        min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
        
        # Crop the image and mask
        cropped_img = img[min_y:max_y+1, min_x:max_x+1]
        cropped_mask = mask[min_y:max_y+1, min_x:max_x+1]
    


    # This squeezes the frame until there are no zero values in the fourth channel anywhere:
    elif crop == "all_zero":
        py_log.log_locals(MY_LOGGER, attr_sets=["size", "math"]); show_image([img, mask])


        # New approach - find the rect that is inside the area of 1s in the fourth channel
        # that has the maximum area.

        # We start at y=0. We will go over each y:
        # We shoot a beam in the horizontal direction and find the first and last column that have 1s in the fourth channel.
        # We go down those columns as long as there are 1s in the fourth channel in both of them.
        # When one of them fails, we stop and calculate the area of the rect, and remember at which row we stopped.
        
        # We could also do binary search when trying to find where the 1s end in both columns.
        # And then just take the smaller of the two values.
        
        # But even better:
        # For each row we find the first and last column that have 1s in the fourth channel.
        boundaries = []
        for y in range(img.shape[0]):
            first_col = None
            last_col = None
            for x in range(img.shape[1]):
                if img[y, x, -1] != 0:
                    first_col = x
                    break
            for x in range(img.shape[1]-1, -1, -1):
                if img[y, x, -1] != 0:
                    last_col = x
                    break

            if first_col is not None and last_col is not None:
                boundaries.append((y, first_col, last_col))


        # Then we get a new sorted list, sorted primarily by first_col, secondarily by y, thirdly by last_col.
        # We find all the lines that have the same first_col as our CURR_POINT.
        # Since this is a convex shape, there should either only be 2, (our CURR_POINT and the lowest point LP),
        # or there should be more, but all are directly below CURR_POINT or directly above LP.
        # Otherwise that means the contour to the left would have to have hills, not one circle-like / one-jill-like shape,
        # and that is not convex.
        


        # All a bit complicated and actually misguided:
        """
        # Then we get a new sorted list, sorted primarily by first_col, secondarily by y, thirdly by last_col.
        # (boundaries is sorted by y) - we keep that.
        # For each y, we get the current line, and we go and find the line that has the same first_col.
        # (This is like shooting a beam vertically down to see where we can get matches.)
        # There might be more such lines (think of rasterization of a slanted rect - the edge will be stairs).
        # Among them, we first go to the rightmost one - the one with the highest y (to maximize area).

        # If that line has a last_col that is greater than the last_col of the current line, 
        # and y bigger than the current line, (has to be below it in the image, because we are shooting down)
        # we have our match.

        # If not, then we simply go left in our list - this is like following the outermost contour of the 1s in the fourth channel:
        # - if there is a pixel with the same first_col and a smaller y, we go one up. (that's what going left is)
        #  (That pixel has only black pixels to the left of it - otherwise it wouldn't be in the list)
        # - if in going up we found no success, we will basically go k left and one up (that's what going left is)
        #    (Because before we were going up the leftmost contour (those pixels had all black to the left of them).

        # The above description is only true if in the locality, the shape is leftward slanted (the top is more left than the bottom).
        # You could also say, in this pixel, the derivative of the shape contour was negative (picture it in your head).

        # BUT leftard slantedness is the only posibility.
        # Because the pixel we have (CURR_PIX) is the leftmost pixel in its row (CURR_ROW).
        # And we also know that it is lower than another pixel vertically above it (ABOVE_PIX).
        # So if we can go down and LEFT and find another pixel (LEFT_DOWN_PIX),
        # then we know, that the line connecting (LEFT_DOWN_PIX) and (ABOVE_PIX) when intersecting (CURR_ROW)
        # has to go to the left of (CURR_PIX), so through an area of 0s.
        # And this violates the fact that our shape is convex.
        # So we have reached a contradiction.

        # This also means that when we go left in the list,
        # (our first_col is smaller apriori due to sort)
        # if our y is bigger, there will be no match from there on so we can stop.
        

        # Because if the derivative wasn't negative,
        # that would mean it is positive and that means there are 1s to the left somewhere.

        # Ecvept if this is literally just one line of pixels, which goes

        # But if the shape is rightward slanted (derivative is positive), we will go k left and one down instead.
        # And this is still perectly following the contour of the 1s in the fourth channel.

        # So we will literally only be following the contour of the 1s in the fourth channel.
        
        # 
        # all those have first_col equal or smaller, so they match in that respect.
        # When we get to the first line that has the last_col equal or greater than the last_col of the current line, 
        # and has its y bigger than the current line, we have our match.
        # That is the match with the highest area.
        # This should work for inscribing the *rect with the biggest area into any convex shape. 
        # * - rect whose lines are parallel to the img frame.

        """





        # Naive approach - crops way too much because it basically makes th frame that is inside the
        # cross where there are no zeros in the fourth channel.
        # With cross I mean as if you extended the inner rect that is kept in all 4 directions.
        """
        min_y, max_y = 0, img.shape[0]
        min_x, max_x = 0, img.shape[1]
        
        while np.any(img[min_y, :, -1] == 0):
            min_y += 1
        
        while np.any(img[max_y-1, :, -1] == 0):
            max_y -= 1

        while np.any(img[:, min_x, -1] == 0):
            min_x += 1

        while np.any(img[:, max_x-1, -1] == 0):
            max_x -= 1
        
        # Crop the image and mask
        cropped_img = img[min_y:max_y, min_x:max_x, :]
        cropped_mask = mask[min_y:max_y, min_x:max_x, :]
        """



    return cropped_img, cropped_mask
