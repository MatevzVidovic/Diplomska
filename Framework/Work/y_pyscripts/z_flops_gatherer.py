

import os
import os.path as osp


# !!! NO SRUN SO IT CAN RUN SBATCH !!!
# python3 -m y_pyscripts.z_flops_gatherer


paths = {
    "zzz_flops_gathering" : {
        "z_pipeline_segnet_sclera",
        "z_pipeline_segnet_veins",
        "z_pipeline_unet_sclera",
        "z_pipeline_unet_veins",
    }
}

for base_dir in paths:
    for pipeline in paths[base_dir]:
        path = f"{base_dir}/{pipeline}"
        id = pipeline.split("z_pipeline_")[-1]

        save_dir = f"b_flops_gathering/{id}_fw"
        os.makedirs(save_dir, exist_ok=True)

        yaml = id.split("_")[-1]

        print(path)
        print(id)
        print(save_dir)

        sbatch_path = osp.join(path, "standalone_scripts", "z_get_fw.sbatch")

        os.system(f"sbatch {sbatch_path} {yaml} {save_dir}")