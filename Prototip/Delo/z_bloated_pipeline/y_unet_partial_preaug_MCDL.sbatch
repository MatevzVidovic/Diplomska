#!/bin/bash


#SBATCH --job-name=uppM
#SBATCH --time=0-12:00:00

#SBATCH -p frida
#SBATCH -c 7
#SBATCH --gpus=A100_80GB

#SBATCH -w ana
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
# ds=${10}
# model_name=${11}
# tesl=${12}

# Pri 3000 2000, UNet_3000x2000, starting kernels 5, expansion 1.3,   je maksimalen bs 17

# Pri 3000 2000, UNet_3000x2000_input_skip_conn, starting kernels 5, expansion 1.3,   je maksimalen bs 13




# Pri 1500 1000, UNet_1500x1000, !!!na A100_80GB!!!  starting kernels 8, expansion 1.5,   je maksimalen bs  med 80 in 85

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
# (1500*1000*8)  *   1*(1-(1.5/4)**5)/(1-(1.5/4))
# 19057617.1875
# exp = 2      23250000
# exp = 3      36609375

# If we were to go and plot this *(1-(exp/4)**5)/(1-(exp/4)) in desmos,
# we see its minimum is in when exp is negative, so ot applicable to us.
# Then it is just growing. When exp is 0, the term is 1. And then it is just going up.
# we can see that his should behave like a polynomial of the 4th degree (5th degreee polynomial divided by 1st degree polynomial)

# When exp is 4, the term is undefined. But the limit is 5. Makes sense, because we have 5 nivojev.
# Exp, term size pairs:  [(0, 1), (1, 1.33), (1.5, 1.58) (2, 1.93), (3, 3), (4, 5), (5, 8), (6, 13)]


# So, for 1500x1000, 8, we can maybe do expansion being 3 we can do bs 40.



# Imamo 5 nivojev.



                                                                     # !!!!!!!!!!!!! CURRENTLY NOT PARTIALLY PREAUGING!!!!
srun --output=x_unet_partial_preaug_MCDL${1}.txt  bash z_pipeline_base/z1_main_MCDL.sh unet_main.py unet_partial_preaug_MCDL${1} 2 7 80 1e-5 ./vein_sclera_data 2048 1024 augment UNet_original_64_2_6 400


