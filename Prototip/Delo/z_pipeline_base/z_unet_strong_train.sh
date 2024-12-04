#!/bin/bash


# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5






# Using torch.optim.SGD (uses less params than ADAM)
# -c 64 pri teh testih, ampak pomoje to nic ne spremeni
# grafa:2 ne pomaga. Bi moral pomoje nekaj s pytorchom dodatno naredit, da bi 2 naenkrat uporabljal.


# Na ana A100_80GB dela bs 250. Z 320 ne dela. 

# Na aga1 na navadnem A100 dela bs 100 (tudi pruning). Z 125 je delalo pa tudi že ni delalo. Z 140 ne dela.

# Za pruning se uporablja vein, ki itak ne preseže 90 primerkov,torej nas ne rabi skrbet pruning del.

# bash z_pipeline_base/z1_test_main.sh main.py test_UNet_main 105 64 80





bash z_pipeline_base/z1_strong_training_main.sh main.py UNet_main 90 48 80


