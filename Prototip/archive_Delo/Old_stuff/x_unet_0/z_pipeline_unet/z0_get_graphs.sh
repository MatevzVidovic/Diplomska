



# srun -c 2 --gpus=A100:1 bash z_pipeline_unet/z0_get_graphs.sh name_of_sd > x_get_graphs.txt

sd_name=$1

echo "sd_name: $sd_name"


source z0_sh_help.sh

python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$graph_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$results_and_stop"
python3 unet_original_main.py --ips 0 --ptd ./vein_sclera_data --sd $sd_name --yaml z_pipeline_unet/unet_original_0.yaml    < "$resource_graph_and_stop"