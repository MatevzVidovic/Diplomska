

import os
import os.path as osp

import shutil as sh


import sys


# srun python3 v_make_scleras_preds_in_vasd.py




preds= osp.join("unet_small_sclera_recall_train", "saved_main", "121_save_preds")

vasd = osp.join("Data", 'vein_and_sclera_data')

splits = ["train", "val", "test"]

for split in splits:
    p = osp.join(vasd, split, "Scleras")
    sh.rmtree(p, ignore_errors=True)
    os.makedirs(p, exist_ok=True)


for split in splits:
    p = osp.join(vasd, split, "Images")
    imgs = os.listdir(p)
    imgs_no_suffix = [img.removesuffix(".jpg") for img in imgs]
    for img in imgs_no_suffix:
        sh.copy(osp.join(preds, f"{img}_pred.png"), osp.join(vasd, split, "Scleras", f"{img}.png"))
