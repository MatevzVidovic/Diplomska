

import os
import os.path as osp
import numpy as np
import cv2

import shutil as sh

from PIL import Image

import sys

import matplotlib.pyplot as plt


vasd = osp.join('vein_and_sclera_data')
os.makedirs(vasd, exist_ok=True)

sd = osp.join('sclera_data')
vsd = osp.join('vein_sclera_data')


# make tiff? make in np format? Make 4chan png?

# with cv2 show first 3 chans separately and the last chan separately

sclera_all_masks_path = osp.join('sclera_all_masks')
os.makedirs(sclera_all_masks_path, exist_ok=True)


paths = os.listdir(sd) # ["test", "train", "val"]
for p in paths:
    curr_p = osp.join(sd, p, "Masks")
    sh.copytree(curr_p, sclera_all_masks_path, dirs_exist_ok=True)



sh.copytree(vsd, vasd, dirs_exist_ok=True)

for folder in os.listdir(vasd):
    goal_f = osp.join(vasd, folder, "Scleras")
    os.makedirs(goal_f, exist_ok=True)
    source = osp.join(vsd, folder, "Images")
    for img_name in os.listdir(source):
        sclera_name = img_name.removesuffix(".jpg") + ".png"
        new_sclera_name = img_name.removesuffix(".jpg") + "_sclera.png"
        curr_source = osp.join(source, img_name)
        curr_dest = osp.join(goal_f, new_sclera_name)
        sh.copy(curr_source, curr_dest)
        




sys.exit(0)
