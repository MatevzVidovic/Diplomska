#!/bin/bash


#SBATCH --job-name=utp
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 30
#SBATCH --gpus=A100

#SBATCH --mem-per-cpu=8G 



# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.

# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5
# lr=$6
# ptd=$7
# iw=$8
# ih=$9
# model_name=${10}
# tesl=${11}
# mti=${12}
# ds=${13}

# bs 4 gre, bs 6 ne gre 
# bs 5 gre


srun --output=x_unet_train_patch${1}.txt  bash z_pipeline_base/z1_main.sh unet_main.py unet_train_patch${1} ${1} 30 80 1e-4 ./vein_sclera_data_patchified_256x256 256 256 UNet_original_64_2_4 100 ${2} pass_through










# inp size se v x in y razpolovi z vsakim nivojem.

# Velikost aktivacij je torej:
# (inp_y * inp_x) * s_ker + (inp_y * inp_x * 1/4 * s_ker) * exp + (inp_y * inp_x * 1/16 * s_ker * exp ) * exp
# Za i-ti nivo torej:   (inp_y * inp_x * (1/4)**i) * s_ker * exp^(i) = (inp_y * inp_x * s_ker)  * (exp/4)**i

 #   S_n = a \frac{1 - r^n}{1 - r} ]
#Where:
  #  (a) is the first term.
 #   (r) is the common ratio.
#    (n) is the number of terms.
# 1 2 4   a=1  r=2   n=3
# 1*(1-2^3)/(1-2) = 1*(1-8)/(-1) = 1* 7 = 7

# Ce sestejemo 5 nivojev, od 0 do 4, je to skupaj:
# (inp_y * inp_x * s_ker)  *   1*(1-(exp/4)**5)/(1-(exp/4))