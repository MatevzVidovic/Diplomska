#!/bin/bash

source z_pipeline_SegNet_random_main/z0_bash_saver.sh z_pipeline_SegNet_random_main

# Creates temp files save_and_stop, results_and_stop, and graph_and_stop
source z_pipeline_SegNet_random_main/z0_temp_inputs.sh

# Sets vars batch_size, nodw
source z_pipeline_SegNet_random_main/SegNet_random_main_params.sh

mti=60


python3 segnet_main.py --ips 0 --bs ${batch_size} --nodw $nodw --sd SegNet_random_main_80_after_pruning --ptd ./vein_sclera_data --mti ${mti}          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


# python3 segnet_main.py --ips 999999 --bs ${batch_size} --nodw $nodw --sd SegNet_random_main_80_after_pruning --ptd ./vein_sclera_data --mti ${mti}          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
