

import os

import cv2
import numpy as np
import y_helpers.shared as shared
if not shared.PLT_SHOW: # For more info, see shared.py
    import matplotlib
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt



all_imgs = os.listdir('./')
all_imgs.sort()


basenames = []

for img in all_imgs:
    img_name  = img.removesuffix('.png')
    if img_name.endswith('_img'):
        img_name = img_name.removesuffix('_img')
        basenames.append(img_name)


img_ix = 0
inp = ""
g = True
y = True
r = True
s = False

if shared.PLT_SHOW:
    plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()

while True:
    img_name = basenames[img_ix] + '_img.png'
    colormap_name = basenames[img_ix] + '_colormap.png'
    sclera_name = basenames[img_ix] + '_sclera.png'

    print(f"Showing {img_name} with colormap {colormap_name} and sclera {sclera_name}")

    img = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    img = img[:,:,:3]
    colormap = cv2.imread(colormap_name)
    sclera = cv2.imread(sclera_name)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    sclera = cv2.cvtColor(sclera, cv2.COLOR_BGR2RGB)


    print(f"Image shape: {img.shape}")
    print(f"Colormap shape: {colormap.shape}")
    print(f"Sclera shape: {sclera.shape}")

    # is_red = colormap == [255, 0, 0]
    # print(is_red.shape)
    # print(is_red)
    # print(np.all(is_red, axis=2))

    where_colormap_is_green = np.where(np.all(colormap == [0, 255, 0], axis=2))
    where_colormap_is_red = np.where(np.all(colormap == [255, 0, 0], axis=2))
    where_colormap_is_yellow = np.where(np.all(colormap == [255, 255, 0], axis=2))


    where_sclera = np.where(np.all(sclera == [255, 255, 255], axis=2))

    showing_img = img.copy()

    if s:
        showing_img[where_sclera] = [255, 255, 255]
    if g:
        showing_img[where_colormap_is_green] = [0, 255, 0]
    if r:
        showing_img[where_colormap_is_red] = [255, 0, 0]
    if y:
        showing_img[where_colormap_is_yellow] = [255, 255, 0]

    print(f"State of toggles: g={g}, y={y}, r={r}, s={s}")
        

    ax.cla()  # Clear the current axes
    ax.imshow(showing_img)  # Display the image
    plt.draw()  # Update the figure
    plt.pause(0.1)  # Pause for a short duration

    # plt.ioff()  # Turn off interactive mode
    if shared.PLT_SHOW:
        plt.show()  # Keep the window open after the loop

    inp = input(f"""Enter num to look at img with that ix. There are {len(basenames)} images.
                Enter 'g' to toggle green.
                Enter 'y' to toggle yellow.
                Enter 'r' to toggle red.
                Enter 'f' to toggle FP and FN (yellow and red).
                Enter 'x' to reset all toggles to original state.
                Enter 's' to toggle sclera.
                Enter 'q' to quit.
                """)

    if inp == 'q':
        break
    elif inp == 'g':
        g = not g
    elif inp == 'y':
        y = not y
    elif inp == 'r':
        r = not r
    elif inp == 'f':
        y = not y
        r = not r
    elif inp == 's':
        s = not s
    elif inp == 'x':
        g = True
        y = True
        r = True
        s = False
    else:
        try:
            img_ix = int(inp)
            if img_ix < 0 or img_ix >= len(basenames):
                print("Invalid index. Try again.")
                continue
        except:
            print("Invalid input. Try again.")
            continue




    
