

import os
import os.path as osp
import shutil as sh
import sys





vasd = osp.join("Data", 'vein_and_sclera_data')
sh.rmtree(vasd, ignore_errors=True)
os.makedirs(vasd, exist_ok=True)

sd = osp.join("Data", 'sclera_data')
vsd = osp.join("Data", 'vein_sclera_data')


# make tiff? make in np format? Make 4chan png?

# with cv2 show first 3 chans separately and the last chan separately

sclera_all_masks_path = osp.join('sclera_all_masks')
os.makedirs(sclera_all_masks_path, exist_ok=True)


paths = os.listdir(sd) # ["test", "train", "val"]
for p in paths:
    curr_p = osp.join(sd, p, "Masks")
    sh.copytree(curr_p, sclera_all_masks_path, dirs_exist_ok=True)





sh.copytree(vsd, vasd, dirs_exist_ok=True)

# rename Masks to Veins
paths = os.listdir(vasd) # ["test", "train", "val"]
for p in paths:
    curr_p = osp.join(vasd, p, "Masks")
    new_p = osp.join(vasd, p, "Veins")
    os.rename(curr_p, new_p)


for folder in os.listdir(vasd):
    goal_f = osp.join(vasd, folder, "Scleras")
    os.makedirs(goal_f, exist_ok=True)
    source = osp.join(sclera_all_masks_path)

    relevant_imgs = osp.join(vasd, folder, "Images")

    for img_name in os.listdir(relevant_imgs):
        sclera_name = img_name.removesuffix(".jpg") + ".png"
        new_sclera_name = img_name.removesuffix(".jpg") + "_sclera.png"
        curr_source = osp.join(source, sclera_name)
        curr_dest = osp.join(goal_f, new_sclera_name)
        sh.copy(curr_source, curr_dest)
        




sys.exit(0)
