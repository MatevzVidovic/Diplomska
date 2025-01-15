#!/bin/bash
#SBATCH --job-name=stsf
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 48
#SBATCH --gpus=A100
#SBATCH --output=x_segnet_train_strong_fast.txt



# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5
# lr=$6
# ptd=$7



# These experiments were on ./sclera_data (because vein_sclera_data only has 89 training examples, so bs can at most be that)

# Using torch.optim.SGD (uses less params than ADAM)
# -c 64 pri teh testih, ampak pomoje to nic ne spremeni
# grafa:2 ne pomaga. Bi moral pomoje nekaj s pytorchom dodatno naredit, da bi 2 naenkrat uporabljal.

# Na ana A100_80GB dela bs 240 (tudi pruning). 250 že ne dela več.

# Na aga1 na navadnem A100 dela bs 120 (tudi pruning). Z 130 ne dela.

# Te rezultati zelo smiselni, ker A100 ima 40GB VRAMa

# Za pruning se uporablja vein, ki itak ne preseže 90 primerkov,torej nas ne rabi skrbet pruning del.




# srun bash z_pipeline_base/z1_test_main.sh segnet_main.py test_SegNet_main 210 64 80





srun bash z_pipeline_base/z1_strong_fast_training_main.sh segnet_main.py segnet_train_strong_fast 90 48 80 1e2  ./vein_sclera_data

