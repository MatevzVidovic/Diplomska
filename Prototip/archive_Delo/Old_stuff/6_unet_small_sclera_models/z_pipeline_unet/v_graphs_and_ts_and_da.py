

import os
import os.path as osp



# srun --pty -p dev -c 7 --gpus=A100 python3 v_graphs_and_ts_and_da.py

# or run me through multipurpose.sbatch



folder_structure = {
    "" : [ "unet_train2" ],
    "pruned_models_unet_IPAD" : [ "unet_IPAD_490_after_pruning", "unet_IPAD_740_after_pruning", "unet_IPAD_940_after_pruning" ],
    "pruned_models_unet_random" : [ "unet_random_490_after_pruning", "unet_random_740_after_pruning", "unet_random_940_after_pruning" ],
    "pruned_models_unet_uniform" : [ "unet_uniform_490_after_pruning", "unet_uniform_740_after_pruning", "unet_uniform_940_after_pruning" ],
}


# # overwrite for testing
folder_structure = {
    "" : [ "unet_train2" ],
}

get_graphs_sh = osp.join("z_pipeline_unet", "z0_get_graphs.sh")
get_da_sh = osp.join("z_pipeline_unet", "z0_get_da.sh")
get_ts_sh = osp.join("z_pipeline_unet", "z0_get_ts.sh")


for base_f, main_fs in folder_structure.items():

    for main_f in main_fs:
        sd = osp.join(base_f, main_f)
        os.system(f"bash {get_ts_sh} {sd}")
        # os.system(f"bash {get_da_sh} {sd}")
        # os.system(f"bash {get_graphs_sh} {sd}")





        # Fixing the mistake of not coppying stuff okay:

        # init_res_calc = osp.join(sd, "saved_model_wrapper", "initial_conv_resource_calc.pkl")
        # os.system(f"rm {init_res_calc}")

        # real_res_calc = osp.join("unet_prune_IPAD", "saved_model_wrapper", "initial_conv_resource_calc.pkl")
        # os.system(f"cp {real_res_calc} {init_res_calc}")