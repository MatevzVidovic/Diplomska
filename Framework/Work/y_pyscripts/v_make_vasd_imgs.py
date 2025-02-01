

import os
import os.path as osp

import shutil as sh



# srun python3 v_make_vask_imgs.py




vasd = osp.join("Data", 'vein_and_sclera_data')

vasd_just_imgs = osp.join("Data", 'vasd_just_imgs')
os.makedirs(vasd_just_imgs, exist_ok=True)



sd = osp.join("Data", 'sclera_data')
vsd = osp.join("Data", 'vein_sclera_data')


splits = ["train", "val", "test"]

for split in splits:
    curr_path = osp.join(vasd, split, "Images")
    for img_name in os.listdir(curr_path):
        img_path = osp.join(curr_path, img_name)
        sh.copy(img_path, vasd_just_imgs)
