#!/bin/bash


#SBATCH --job-name=ut
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100
#SBATCH --mem-per-cpu=8G 


# #SBATCH -w ana


# --job-name sadly cannot be changed with the passed id.
# You can't give it as an srun param, that doesn't work.
# But also, substituting it into SBATCH commands doesn't work.


# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# lr=$5
# ptd=$6
# iw=$7
# ih=$8
# model_name=$9
# tesl=${10}
# mti=${11}


# bs 4 gre, bs 6 ne gre 
# bs 5 gre


srun --output=x_unet_train${1}.txt  bash z_pipeline_base/z1_main.sh unet_main.py unet_train${1} ${1} 7 5e-5 ./vein_sclera_data 2048 1024 UNet_64_2_6 400 ${2}










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