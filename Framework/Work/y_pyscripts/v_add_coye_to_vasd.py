

import os
import os.path as osp
import shutil as sh
import sys





vasd = osp.join("Data", 'vein_and_sclera_data')

bcosfire = osp.join("Data", 'coye2')






for folder in os.listdir(vasd):
    goal_f = osp.join(vasd, folder, "Coye")
    os.makedirs(goal_f, exist_ok=True)

    relevant_imgs = osp.join(vasd, folder, "Images")

    for img_name in os.listdir(relevant_imgs):
        bcosfire_name = img_name.removesuffix(".jpg") + ".png"
        curr_source = osp.join(bcosfire, bcosfire_name)
        curr_dest = osp.join(goal_f, bcosfire_name)
        try:
            sh.copy(curr_source, curr_dest)
        except:
            print(f"Error with {curr_source} and {curr_dest}")    




sys.exit(0)
