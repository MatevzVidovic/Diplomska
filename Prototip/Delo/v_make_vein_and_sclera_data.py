

import os
import os.path as osp
import numpy as np
import cv2

from PIL import Image

import matplotlib.pyplot as plt


vasd = osp.join('vein_and_sclera_data')

sd = osp.join('sclera_data')
vsd = osp.join('vein_sclera_data')


# make tiff? make in np format? Make 4chan png?

# with cv2 show first 3 chans separately and the last chan separately