#!/bin/bash


output_basic_name="main"

results_folder_name="bash_results"

# Step 1: Create the "bash_results" directory if it doesn't exist
mkdir -p ${results_folder_name}

# Create /older and /curr directories inside bash_results if they don't exist
mkdir -p "${results_folder_name}/older"
mkdir -p "${results_folder_name}/curr"

# Move contents from /curr to /older
# (the 2>/dev/null is to suppress the error message if the folder is empty)
mv "${results_folder_name}/curr/"* "${results_folder_name}/older/" 2>/dev/null




# Step 2: Read and update the current bash index
prev_bash_ix_file="${results_folder_name}/prev_bash_ix.txt"
if [[ -f $prev_bash_ix_file ]]; then
    curr_bash_ix=$(<"$prev_bash_ix_file")
    curr_bash_ix=$((curr_bash_ix + 1))
else
    curr_bash_ix=0
fi
echo "$curr_bash_ix" > "$prev_bash_ix_file"

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






# Testing if this works:
# - save
# - train for 2 iterations
# - save
# - prune (2 prunings by 0.02 percent, between them 2 train iters)
# - save
# - prune (2 prunings by 0.02 percent, before each 2 train iters)
# - save
# - migrate to new dataset (from sclera to vein_sclera)
# - prune (2 prunings by 0.02 percent, before each 2 train iters)
# - save
# - prune (2 prunings by 0.02 percent, before each 2 train iters)
# - save
# - train for 2 iterations
# - save


    # parser.add_argument('--ips', type=int, default=1e9,
    #                     help='iter_possible_stop An optional positional argument with a default value of 1e9')

    # parser.add_argument("--bs", type=int, default=16, help='BATCH_SIZE')
    # parser.add_argument("--nodw", type=int, default=1, help='NUM_OF_DATALOADER_WORKERS')
    # parser.add_argument("--ntibp", type=int, default=10, help='NUM_TRAIN_ITERS_BETWEEN_PRUNINGS')

    # parser.add_argument("--sd", type=str, default="SegNet", help='SAVE_DIR')
    # parser.add_argument("--ptd", type=str, default="./sclera_data", help='PATH_TO_DATA')

python3 segnet_main.py --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

echo "Here"

python3 segnet_main.py --ips 999999 --bs 4 --nodw 10 --sd SegNet --ptd ./sclera_data -t --mti 2           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 segnet_main.py --ips 0 --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


python3 segnet_main.py --ips 999999 --bs 4 --nodw 10 --ntibp 2 --sd SegNet --ptd ./sclera_data -t --pruning_phase --pbop --map 2 --pnkao 50 --rn flops_num --ptp 0.02           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 segnet_main.py --ips 0 --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


python3 segnet_main.py --ips 999999 --bs 4 --nodw 10 --ntibp 2 --sd SegNet --ptd ./sclera_data -t --pruning_phase --pbop --map 2 --pnkao 50 --rn flops_num --ptp 0.02           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 segnet_main.py --ips 0 --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))




python3 segnet_main.py --ips 999999 --bs 4 --nodw 10 --ntibp 2 --sd SegNet --ptd ./vein_sclera_data -t --pruning_phase --pbop --map 2 --pnkao 50 --rn flops_num --ptp 0.02           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 segnet_main.py --ips 0 --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


python3 segnet_main.py --ips 999999 --bs 4 --nodw 10 --ntibp 2 --sd SegNet --ptd ./vein_sclera_data -t --pruning_phase --pbop --map 2 --pnkao 50 --rn flops_num --ptp 0.02           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 segnet_main.py --ips 0 --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))


python3 segnet_main.py --ips 999999 --bs 4 --nodw 10 --sd SegNet --ptd ./vein_sclera_data -t --mti 2           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))
python3 segnet_main.py --ips 0 --sd SegNet --ptd ./sclera_data -t < "$save_and_stop"           2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))














# Main training:

# python3 segnet_main.py 999999 4 10 1 SegNet ./sclera_data --mti 20          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# python3 segnet_main.py 999999 4 10 1 SegNet ./vein_sclera_data --mti 50          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))

# python3 segnet_main.py 999999 4 10 1 SegNet ./vein_sclera_data --pruning_phase --pbop --map 15 --pnkao 50 --rn flops_num --ptp 0.05          2>&1 | tee "${rfn}/curr/${obn}_${cbi}_${cn}.txt"; cn=$((cn + 1))












    # # Mandatory args:


    # parser.add_argument('--ips', type=int, default=1e9,
    #                     help='iter_possible_stop An optional positional argument with a default value of 1e9')

    # parser.add_argument("--bs", type=int, default=16, help='BATCH_SIZE')
    # parser.add_argument("--nodw", type=int, default=1, help='NUM_OF_DATALOADER_WORKERS')
    # parser.add_argument("--ntibp", type=int, default=10, help='NUM_TRAIN_ITERS_BETWEEN_PRUNINGS')

    # parser.add_argument("--sd", type=str, default="SegNet", help='SAVE_DIR')
    # parser.add_argument("--ptd", type=str, default="./sclera_data", help='PATH_TO_DATA')


    # parser.add_argument("--ips", type=int, default=1e9, help='MAX_TRAIN_ITERS')





    
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
