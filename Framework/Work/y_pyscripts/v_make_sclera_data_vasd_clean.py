

import os
import os.path as osp

import shutil as sh




# srun python3 v_make_sclera_data_vasd_clean.py




sd = osp.join("Data", 'sclera_data')

sdvc = osp.join("Data", 'sclera_data_vasd_clean')
sh.rmtree(sdvc, ignore_errors=True)
os.makedirs(sdvc, exist_ok=True)

splits = ["train", "val", "test"]

for split in splits:
    sh.copytree(osp.join(sd, split), osp.join(sdvc, split), dirs_exist_ok=True)



vasd_just_imgs = osp.join("Data", 'vasd_just_imgs')
vasd_image_names = {img.removesuffix(".jpg") for img in os.listdir(vasd_just_imgs)}


num_removed = 0
for split in splits:
    print(f"In split {split}")
    curr_path = osp.join(sdvc, split, "Images")
    for img_name in os.listdir(curr_path):
        img_name_no_suffix = img_name.removesuffix(".jpg")
        if img_name_no_suffix in vasd_image_names:
            print(f"Removing {img_name}")
            os.remove(osp.join(curr_path, img_name))
            mask_path = osp.join(sdvc, split, "Masks", f"{img_name_no_suffix}.png")
            os.remove(mask_path)
            num_removed += 1
print(f"Removed {num_removed} images")


os.makedirs(osp.join(sdvc, "save_preds", "Images"), exist_ok=True)

sh.copytree(vasd_just_imgs, osp.join(sdvc, "save_preds", "Images"), dirs_exist_ok=True)
# sh.move(osp.join(sdvc, "save_preds", "vasd_just_imgs"), osp.join(sdvc, "save_preds", "Images"))
