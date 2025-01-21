#!/bin/bash


#SBATCH --job-name=ut
#SBATCH --time=0-5:00:00

#SBATCH -p frida
#SBATCH -c 64
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G 


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.



source z_pipeline_unet/z0_main_name.sh



id=small_sclera_recall_0
mti=20
python3 ${main_name} --mti ${mti} --sd unet_small_sclera_recall_train --yaml z_pipeline_unet/unet_${id}.yaml  > x_unet_${id}_train.txt

id=small_sclera_recall_1
mti=20
python3 ${main_name} --mti ${mti} --sd unet_small_sclera_recall_train --yaml z_pipeline_unet/unet_${id}.yaml  > x_unet_${id}_train.txt

id=small_sclera_recall_2
mti=20
python3 ${main_name} --mti ${mti} --sd unet_small_sclera_recall_train --yaml z_pipeline_unet/unet_${id}.yaml  > x_unet_${id}_train.txt


id=small_sclera_recall_3
mti=1
python3 ${main_name} --mti ${mti} --sd unet_small_sclera_recall_train --yaml z_pipeline_unet/unet_${id}.yaml  > x_unet_${id}_train.txt

srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_sp.sh small_sclera_recall_2 unet_small_sclera_recall_train > x_sp_small_sclera_recall.txt

srun -c 10 --gpus=A100:1 bash z_pipeline_unet/z0_get_ts.sh small_sclera_recall_2 unet_small_sclera_recall_train > x_ts_small_sclera_recall.txt





