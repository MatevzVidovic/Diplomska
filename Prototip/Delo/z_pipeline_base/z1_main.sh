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
# lr=$5
# ptd=$6
# iw=$7
# ih=$8
# model_name=$9
# tesl=${10}
# mti=${11}
# tras=${12}

main_name=$1
folder_name=$2
bs=$3
nodw=$4
lr=$5
ptd=$6
iw=$7
ih=$8
model_name=$9
tesl=${10}
mti=${11}
param_num=11

if [[ $# -ne $param_num ]]; then
    echo "Error: The number of parameters is not correct. Expected $param_num, given $#. Given params: $@"
    exit 1
fi

echo $@


# additional use of z_bash_saver.sh
program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"












# Main training:

python3 ${main_name} --bs ${bs} --nodw ${nodw} --sd ${folder_name} --ptd ${ptd} --mti ${mti} --lr ${lr} --tesl ${tesl} --iw ${iw} --ih ${ih} -m ${model_name}            2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))



    # # Model specific arguments (because of default being different between different models you can't just copy them between models)

    # parser.add_argument("--bs", type=int, default=16, help='BATCH_SIZE')
    # parser.add_argument("--nodw", type=int, default=10, help='NUM_OF_DATALOADER_WORKERS')
    # parser.add_argument("--sd", type=str, default="SegNet", help='SAVE_DIR')
    # parser.add_argument("--ptd", type=str, default="./sclera_data", help='PATH_TO_DATA')
    # parser.add_argument("--lr", type=str, help="Learning rate", default=1e-3)



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

