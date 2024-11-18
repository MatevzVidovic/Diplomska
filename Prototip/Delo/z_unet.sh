#!/bin/bash


# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5


# grafa:2 ne pomaga. Bi moral pomoje nekaj s pytorchom dodatno naredit, da bi 2 naenkrat uporabljal.


# Na ana A100_80GB dela bs 250. Z 320 ne dela. 

# Na aga1 na navadnem A100 dela bs 100 (tudi pruning). Z 125 dela. Z 140 ne dela.

# Za pruning se uporablja vein, ki itak ne preseže 90 primerkov,torej nas ne rabi skrbet pruning del.

# source z1_test_main.sh main.py test_UNet_main 125 64 80





source z1_standard_main.sh main.py UNet_main 105 64 80


