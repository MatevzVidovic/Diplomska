
#!/bin/bash



# python3 main_resnet.py 2 10 999999 ResNet ./sclera_data -t --mti 1 > resnet_sclera_50.txt
# python3 main_resnet.py 2 10 999999 ResNet ./vein_sclera_data -t --mti 1 > resnet_vein_sclera_150.txt
# python3 main_resnet.py 2 10 999999 ResNet ./vein_sclera_data -t --pruning_phase --pbop --map 2 --pnkao 20 --rn flops_num --ptp 0.01 > resnet_vein_sclera_75_20_flops_num_0.01.txt

python3 main_resnet.py 2 10 999999 ResNet ./sclera_data --mti 50 > resnet_sclera_50.txt
python3 main_resnet.py 2 10 999999 ResNet ./vein_sclera_data --mti 150 > resnet_vein_sclera_150.txt
python3 main_resnet.py 2 10 999999 ResNet ./vein_sclera_data --pruning_phase --pbop --map 75 --pnkao 20 --rn flops_num --ptp 0.01 > resnet_vein_sclera_75_20_flops_num_0.01.txt


python3 main.py 4 10 999999 UNet ./sclera_data --mti 50 > unet_sclera_50.txt
python3 main.py 4 10 999999 UNet ./vein_sclera_data --mti 150 > unet_vein_sclera_150.txt
python3 main.py 4 10 999999 UNet ./vein_sclera_data --pruning_phase --pbop --map 75 --pnkao 20 --rn flops_num --ptp 0.01 > unet_vein_sclera_75_20_flops_num_0.01.txt




    # # Add the optional positional arguments
    # parser.add_argument("BATCH_SIZE", nargs='?', type=int, default=1)
    # parser.add_argument("NUM_OF_DATALOADER_WORKERS", nargs='?', type=int, default=1)
    
    # parser.add_argument('iter_possible_stop', nargs='?', type=int, default=1e9,
    #                     help='An optional positional argument with a default value of 1e9')

    # parser.add_argument("SAVE_DIR", nargs='?', type=str, default="ResNet")
    # parser.add_argument("PATH_TO_DATA", nargs='?', type=str, default="./vein_sclera_data")
    
    # # These store True if the flag is present and False otherwise.
    # # Watch out with argparse and bool fields - they are always True if you give the arg a nonempty string.
    # # So --pbop False would still give True to the pbop field.
    # # This is why they are implemented this way now.
    # parser.add_argument("-t", "--is_test_run", action='store_true',
    #                     help='If present, enables test run')
    # parser.add_argument('-p', '--pruning_phase', action='store_true',
    #                     help='If present, enables pruning phase (automatic pruning)')
    # parser.add_argument('--pbop', action='store_true',
    #                     help='Prune by original percent, otherwise by number of filters')
    

    # # Add the optional arguments
    # # setting error_ix: ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)
    # parser.add_argument('--e_ix', type=int, default=3,
    #                     help='ix of the loss you want in the tuple: (test_loss, IoU, F1, IoU_as_avg_on_matrixes)')
    # parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations')
    # parser.add_argument('--map', type=int, default=1e9, help='Max auto prunings')
    # parser.add_argument('--nept', type=int, default=1,
    #                     help='Number of epochs per training iteration')
    # parser.add_argument('--pnkao', type=int, default=20, help="""Prune n kernels at once - in one pruning iteration, we:
    #                     1. calculate the importance of all kernels
    #                     2. prune n kernels based on these importances
    #                     3. calculate the importances based on the new pruned model
    #                     4. prune n kernels based on these new importances
    #                     5. ...
    #                     Repeat until we have pruned the desired amount of kernels.

    #                     Then we go back to training the model until it is time for another pruning iteration.


    #                     In theory, it would be best to have --pnkao at 1, because we get the most accurate importance values.
    #                     However, this is very slow. And also it doesn't make that much of a difference in quality.
    #                     (would be better to do an actual retraining between the prunings then, 
    #                     since we are doing so many epoch passes it is basically computationally worse than retraining).

    #                     Also, you can do epoch_pass() on the validation set, not the training set, because it is faster.

    #                     If you are not using --pbop, then you can set --pnkao to 1e9.
    #                     Because the number of kernels we actually prune is capped by --nftp.
    #                     It will look like this:
    #                     1. calculate the importance of all kernels
    #                     2. prune --nftp kernels based on these importances
    #                     Done.

    #                     But if you are using --pbop, then you should set --pnkao to a number that is not too high.
    #                     Because we actually prune --pnkao kernels at once. And then we check if now we meet our resource goals.
    #                     So if you set it to 1e9, it will simply prune the whole model in one go.

    #                     """)
    # parser.add_argument('--nftp', type=int, default=1,
    #                     help='Number of filters to prune in one pruning')
    # parser.add_argument('--rn', type=str, default="flops_num", help='Resource name to prune by')
    # parser.add_argument('--ptp', type=float, default=0.01, help='Proportion of original {resource_name} to prune')
