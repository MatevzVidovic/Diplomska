#!/bin/bash



# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5




# grafa:2 ne pomaga. Bi moral pomoje nekaj s pytorchom dodatno naredit, da bi 2 naenkrat uporabljal.

# Na ana A100_80GB dela bs 240 (tudi pruning). 250 že ne dela več.

# Na aga1 na navadnem A100 dela bs 120 (tudi pruning). Z 130 ne dela.

# Te rezultati zelo smiselni, ker A100 ima 40GB VRAMa

# Za pruning se uporablja vein, ki itak ne preseže 90 primerkov,torej nas ne rabi skrbet pruning del.




# bash z1_test_main.sh segnet_main.py test_SegNet_main 210 64 80





bash z1_standard_main.sh segnet_main.py SegNet_main 210 64 80

