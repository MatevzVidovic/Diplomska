#!/bin/bash


output_basic_name="main"

results_folder_name="bash_results"

# Step 1: Create the "bash_results" directory if it doesn't exist
mkdir -p ${results_folder_name}

# Create /older and /curr directories inside bash_results if they don't exist
mkdir -p "${results_folder_name}/older"
mkdir -p "${results_folder_name}/curr"






# Step 2: Read and update the current bash index
prev_bash_ix_file="${results_folder_name}/prev_bash_ix.txt"
if [[ -f $prev_bash_ix_file ]]; then
    curr_bash_ix=$(<"$prev_bash_ix_file")
    curr_bash_ix=$((curr_bash_ix + 1))
else
    curr_bash_ix=0
fi
echo "$curr_bash_ix" > "$prev_bash_ix_file"





# Move contents from /curr to /older
# (the 2>/dev/null is to suppress the error message if the folder is empty)
mkdir -p "${results_folder_name}/older/moved_in_${curr_bash_ix}"
mv "${results_folder_name}/curr/"* "${results_folder_name}/older/" 2>/dev/null







# Step 3: Save the script content into a file
script_content_file="${results_folder_name}/curr/bash_code_${curr_bash_ix}.sh"
cat "$0" > "$script_content_file"

# Save the starting time of the script
start_time=$(date)
time_file="${results_folder_name}/curr/time_${curr_bash_ix}.txt"
echo "$start_time" > "$time_file"

# Step 4: Execute commands and save outputs
command_num=0
obn=${output_basic_name}
rfn=${results_folder_name}
cbi=${curr_bash_ix}
cn=${command_num}








save_and_stop=$(mktemp)
printf "s\nstop\n" > "$save_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


results_and_stop=$(mktemp)
printf "r\nstop\n" > "$results_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$results_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

graph_and_stop=$(mktemp)
printf "g\nstop\n" > "$graph_and_stop"
# python3 ${main_file} --ips 0 --sd ${folder_name}  -t < "$graph_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


# Writing to a file and a terminal:
# [command]    2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
# Explanation:
# 2>&1 - redirect stderr to stdout 
# | - pipe the output to the next command
# tee - read from stdin and write to stdout and files (stdin is here the output of the previous command)


# This didn't work so far:
# {
# We want another file, which will collect all errors from all the commands.
# So we can see all the errors that happened fast. To locate them after, we just ctrl+f them in the actual output files.
# We do this like so:

# Saving std error to a file while we are also doing the above saving of stdout and stderr:
# 2>> "${rfn}/curr/stderr_log_{cbi}.txt" 2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
# The first part redirects stderr to this file (>> appends, > overwrites)
# So somehow 2 goes both to the file and to 1. Don't ask me how exactly this works.
# }





main_name="segnet_main.py"


program_content_file="${results_folder_name}/curr/program_code_${curr_bash_ix}.py"
cat "${main_name}" > "${program_content_file}"



# test main training (fast execution):
# (and for testing --bs and --pnkao and such:
# --bs - how much it can take
# -- pnkao - how many "Curr resource value:" prints there are. It's safest to have sth like 3 pruning rounds
# (at the point that if you decreased --pnkao by like 10, youd start getting a bounch of 4 iteration prunings - this means the 3rd iteration is mostly not overshooting)

# folder_name="test_SegNet_main"

# python3 ${main_name} --ips 999999 --bs 10 --nodw 10 --ntibp 1 --sd ${folder_name} --ptd ./sclera_data --mti 1          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# te so itak hitri, ker majhen train set
# python3 ${main_name} --ips 999999 --bs 10 --nodw 10 --ntibp 1 --sd ${folder_name} --ptd ./vein_sclera_data --mti 2          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# torej (2 tr + k epoch passov) * 15 = 45 epochov
# 4 tr * 15 = 60 trainingov
# vsak pruning pa ima vsakih 100 kernelov en epoch pass
# Rezimo torej še 7 epoch passov
# python3 ${main_name} --ips 999999 --bs 10 --nodw 10 --ntibp 1 --sd ${folder_name} --ptd ./vein_sclera_data --pruning_phase --pbop --map 2 --pnkao 80 --rn flops_num --ptp 0.05          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))




# Main training:

folder_name="SegNet"

# ogromen dataset, ne rabis veliko trainingov
python3 ${main_name} --ips 999999 --bs 10 --nodw 10 --ntibp 1 --sd ${folder_name} --ptd ./sclera_data --mti 10          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# te so itak hitri, ker majhen train set
python3 ${main_name} --ips 999999 --bs 10 --nodw 10 --ntibp 1 --sd ${folder_name} --ptd ./vein_sclera_data --mti 50          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# torej (2 tr + k epoch passov) * 15 = 45 epochov
# 4 tr * 15 = 60 trainingov
# vsak pruning pa ima vsakih 100 kernelov en epoch pass
# Rezimo torej še 7 epoch passov
python3 ${main_name} --ips 999999 --bs 10 --nodw 10 --ntibp 4 --sd ${folder_name} --ptd ./vein_sclera_data --pruning_phase --pbop --map 15 --pnkao 80 --rn flops_num --ptp 0.05          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))









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
    # parser.add_argument('--nftp', type=int, default=1,
    #                     help='Number of filters to prune in one pruning')
    # parser.add_argument('--rn', type=str, default="flops_num", help='Resource name to prune by')
    # parser.add_argument('--ptp', type=float, default=0.01, help='Proportion of original {resource_name} to prune')

    # parser.add_argument("--tp", action="store_true", help="test pruning. This makes it so all the conv layers have 0.999999 as their limit for flops. This generally means each one only gets one kernel to prune.",)
    

    
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


