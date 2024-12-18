#!/bin/bash


# The script is meant for automatic storing of outputs of programs.

# This script creates bash_results/curr/ and bash_results/older
# It moves what is in curr into older.
# Then it writes into curr the copy of the current .sh file,
# and the time at the start of it's execution.

# It then enables you to save the output of your commands by simply pasting this line:
# # [command]    2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
# This will then write to a file in bash_results/curr, while also printing to stdout.

# For this, please do not use:
# command_num=0
# obn=${output_basic_name}
# rfn=${results_folder_name}
# cbi=${curr_bash_ix}
# cn=${command_num}

source z_pipeline_base/z0_bash_saver.sh


# Creates temp files save_and_stop, results_and_stop, and graph_and_stop
source z_pipeline_base/z0_temp_inputs.sh





# main_name=$1
# folder_name=$2
# bs=$3
# nodw=$4
# pnkao=$5


# param_num=5

# if [[ $# -ne $param_num ]]; then
#     echo "Error: The number of parameters is not correct."
#     exit 1
# fi







# # additional use of z_bash_saver.sh
# program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
# cat "${main_name}" > "${program_content_file}"















# # test main training (fast execution):
# # (and for testing --bs and --pnkao and such:
# # --bs - how much it can take
# # --pnkao - how many "Curr resource value:" prints there are. It's safest to have sth like 3 pruning rounds
# # (at the point that if you decreased --pnkao by like 10, youd start getting a bounch of 4 iteration prunings - this means the 3rd iteration is mostly not overshooting)

# # folder_name="test_SegNet_main"

# python3 ${main_name} --ips 999999 --bs ${bs} --nodw ${nodw} --ntibp 1 --sd ${folder_name} --ptd ./sclera_data --mti 1          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# # te so itak hitri, ker majhen train set
# python3 ${main_name} --ips 999999 --bs ${bs} --nodw ${nodw} --ntibp 1 --sd ${folder_name} --ptd ./vein_sclera_data --mti 2          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# # torej (2 tr + k epoch passov) * 15 = 45 epochov
# # 4 tr * 15 = 60 trainingov
# # vsak pruning pa ima vsakih 100 kernelov en epoch pass
# # Rezimo torej še 7 epoch passov
# python3 ${main_name} --ips 999999 --bs ${bs} --nodw ${nodw} --ntibp 1 --sd ${folder_name} --ptd ./vein_sclera_data --pruning_phase --pbop --map 2 --pnkao ${pnkao} --rn flops_num --ptp 0.05          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))









# main_name=$1
# bs=$2
# nodw=$3
# iw=$4
# ih=$5
# model_name=$6

main_name=$1
bs=$2
nodw=$3
iw=$4
ih=$5
model_name=$6
param_num=6

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

echo $@


# additional use of z_bash_saver.sh
program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"






# Function to delete a folder if it exists and then create it empty
create_empty_folder() {
    local folder_name="$1"  # Access the first parameter

    if [ -d "$folder_name" ]; then
        rm -r "$folder_name"
    fi

    mkdir "$folder_name"
}




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
# tras=${12}

# make it 1.5 of the batch size, so we do 2 iterations per epoch and also test out if things work with a not-full batch size
tesl=$((bs + bs / 2))
tras=$((bs * 2))

create_empty_folder test_main_vein
python3 ${main_name} --bs ${bs} --nodw ${nodw} --sd test_main_vein --ptd ./vein_sclera_data --mti 2 --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras}            2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

create_empty_folder test_main_sclera
python3 ${main_name} --bs ${bs} --nodw ${nodw} --sd test_main_sclera --ptd ./sclera_data --mti 2 --lr 1e-4 --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name} --tras ${tras}            2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))








    # # Model specific arguments (because of default being different between different models you can't just copy them between models)

    # parser.add_argument("-m", "--model", type=str, default="UNet_256x256", help='Model to use. Options: UNet_256x256, UNet_3000x2000')
    # parser.add_argument("--bs", type=int, default=4, help='BATCH_SIZE')
    # parser.add_argument("--nodw", type=int, default=4, help='NUM_OF_DATALOADER_WORKERS')
    # parser.add_argument("--sd", type=str, default="UNet", help='SAVE_DIR')
    # parser.add_argument("--ptd", type=str, default="./sclera_data", help='PATH_TO_DATA')
    # parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    # parser.add_argument('--iw', type=int, default=256, help='Input width')
    # parser.add_argument('--ih', type=int, default=256, help='Input height')
    # parser.add_argument('--ic', type=int, default=3, help='Input channels')
    # parser.add_argument('--oc', type=int, default=2, help='Output channels')
    # parser.add_argument('--ds', type=str, default="augment", help='Dataset option. Options: augment, pass_through, partially_preaugmented.')
    # parser.add_argument('--tesl', type=int, default=1e9, help=f"""TRAIN_EPOCH_SIZE_LIMIT. If we have 1500 images in the training set, and we set this to 1000, 
    #                     we will stop the epoch as we have trained on >= 1000 images.
    #                     We should watch out to use shuffle=True in the DataLoader, because otherwise we will always only train on the first 1000 images in the Dataset's ordering.

    #                     This is useful if we want to do rapid prototyping, and we don't want to wait for the whole epoch to finish.
    #                     Or if one epoch is too huge so it just makes more sense to train on a few smaller ones.
    #                     """)


    # # General arguments

    # parser.add_argument('--ips', type=int, default=1e9,
    #                     help='iter_possible_stop An optional positional argument with a default value of 1e9')
    # parser.add_argument("--ntibp", type=int, default=10, help='NUM_TRAIN_ITERS_BETWEEN_PRUNINGS')

    
    # # With action='store_true' these args store True if the flag is present and False otherwise.
    # # Watch out with argparse and bool fields - they are always True if you give the arg a nonempty string.
    # # So --pbop False would still give True to the pbop field.
    # # This is why they are implemented this way now.
    # parser.add_argument("-t", "--is_test_run", action='store_true',
    #                     help='If present, enables test run')
    # parser.add_argument('-p', '--pruning_phase', action='store_true',
    #                     help='If present, enables pruning phase (automatic pruning)')
        

    # # Add the optional arguments
    # # setting error_ix: ix of the loss you want in the tuple: (CE_loss, approx_IoU, F1, IoU)
    # # CE_loss only really makes sense - because if the model is a bit better, we are sure it goes down
    # # (IoU and F1 are based on binary classification, so a slightly better model might still do the same predictions, so the loss would be the same - and so you can't choose which to clean away)
    # parser.add_argument('--ce_ix', type=int, default=0,
    #                     help='cleaning error ix. We takeix of the loss you want in the tuple: (CE_loss, IoU, F1, IoU_as_avg_on_matrixes)')
    # parser.add_argument("--ck", type=int, default=3, help="Cleanup k. When saving, how many models to keep. (k+1 models are kept if the current model is not in the best k - because we have to keep it to train the next one.)")
    # parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations')
    # parser.add_argument('--map', type=int, default=1e9, help='Max auto prunings')
    # parser.add_argument('--nept', type=int, default=1,
    #                     help='Number of epochs per training iteration')
    
    # parser.add_argument('--pbop', action='store_true',
    #                     help='Prune by original percent, otherwise by number of filters')
    # parser.add_argument('--nftp', type=int, default=1,
    #                     help="""
    #                     !!! ONLY APPLIES IF --pbop IS FALSE !!!
    #                     Number of filters to prune in one pruning""")
    
    # parser.add_argument('--pnkao', type=int, default=20, help="""
    #                     !!! THIS IS OVERRIDDEN IF --ifn IS 0 OR 1 !!!
    #                     It becomes 1. Because we need the new importances to be calculated after every pruning.

    #                     Prune n kernels at once - in one pruning iteration, we:
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
    # parser.add_argument('--rn', type=str, default="flops_num", help='Resource name to prune by')
    # parser.add_argument('--ptp', type=float, default=0.01, help="""Proportion of original {resource_name} to prune - actually, we don't just prune by this percent, because that get's us bad results.
    #                     Every time we prune, we prune e.g. 1 percent. Because of pnkao we overshoot by a little. So next time, if we prune by 1 percent again, we will overshoot by a little again, and the overshoots compound.
    #                     So we will instead prune in this way: get in which bracket of this percent we are so far (eg, we have 79.9 percent of original weights), then we will prune to 79 percent and pnkao will overshoot a little.
    #                     """)


    
    # def custom_type_conversion(value):
    #     try:
    #         # Try to convert to int
    #         return int(value)
    #     except ValueError:
    #         try:
    #             # Try to convert to tuple using ast.literal_eval
    #             return ast.literal_eval(value)
    #         except (ValueError, SyntaxError):
    #             raise argparse.ArgumentTypeError(f"Invalid value: {value}")
    # parser.add_argument("--ifn", type=custom_type_conversion, default=(0.5, 0.5, 0.5), help="Importance func. If 0, random pruning. If 1, uniform pruning. If tuple, we get IPAD with those 3 alphas.")

    # parser.add_argument("--tp", action="store_true", help="""test pruning. 
    #                     This makes it so all the conv layers have 0.999999 as their limit for weights. 
    #                     MIND THAT input slice pruning also affects the weights - so this generally means each layer will get one kernel OR one input slice pruned.
    #                     The uniform pruning starts at CURR_PRUNING_IX - so if you want the other half of layers to have their kernels pruned, just change that to 1.""")
    # parser.add_argument('--optim', type=str, default="Adam", help='Optimizer used. Adam, SGD, LBFGS')
    # parser.add_argument("--loss_fn_name", type=str, default="Default", help="""Loss function used. Default (MCDL), 
    #                     MCDL for MultiClassDiceLoss, CE for CrossEntropyLoss,
    #                     CEHW for CrossEntropyLoss_hardcode_weighted,
    #                     MCDL_CEHW_W for MultiClassDiceLoss and CrossEntropyLoss_hardcode_weighted in a weighted pairing of both losses,
    #                     MCDLW for MultiClassDiceLoss with background adjustment,
    #                     """)

    # def isfloat(s):
    #     """
    #     Checks if a string represents a valid float number.
        
    #     Args:
    #         s (str): The input string to check.
        
    #     Returns:
    #         bool: True if the string represents a valid float, False otherwise.
    #     """
    #     # Remove leading and trailing whitespace
    #     s = s.strip()
        
    #     # Handle empty string
    #     if not s:
    #         return False
        
    #     # Allow for negative sign
    #     if s.startswith('-'):
    #         s = s[1:]
        
    #     # Check if the remaining part is either a digit or a single decimal point
    #     return s.replace('.', '', 1).isdigit()

    # def list_conversion(list_str):
    #     if isfloat(list_str):
    #         return [float(list_str)]
    #     return ast.literal_eval(list_str)
    
    # parser.add_argument("--alphas", type=list_conversion, default=[], help="Alphas used in loss_fn. Currently only one is used. If there is just one alpha, you can just pass a float as an arg, like: 0.8.")

