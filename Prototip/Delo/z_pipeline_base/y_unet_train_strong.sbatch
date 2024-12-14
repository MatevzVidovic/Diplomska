#!/bin/bash
#SBATCH --job-name=uts
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100
#SBATCH --output=x_unet_train_strong.txt


# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5
# ptd=$6




# These experiments were on ./sclera_data (because vein_sclera_data only has 89 training examples, so bs can at most be that)

# Using torch.optim.SGD (uses less params than ADAM)
# -c 64 pri teh testih, ampak pomoje to nic ne spremeni
# grafa:2 ne pomaga. Bi moral pomoje nekaj s pytorchom dodatno naredit, da bi 2 naenkrat uporabljal.


# Na ana A100_80GB dela bs 250. Z 320 ne dela. 

# Na aga1 na navadnem A100 dela bs 100 (tudi pruning). Z 125 je delalo pa tudi že ni delalo. Z 140 ne dela.

# Za pruning se uporablja vein, ki itak ne preseže 90 primerkov,torej nas ne rabi skrbet pruning del.

# srun bash z_pipeline_base/z1_test_main.sh unet_main.py test_UNet_main 105 64 80





srun bash z_pipeline_base/z1_strong_training_main.sh unet_main.py unet_train_strong 90 48 80 ./vein_sclera_data


