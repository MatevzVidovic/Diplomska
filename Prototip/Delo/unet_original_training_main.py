

import os
import os.path as osp
import logging
import python_logger.log_helper_off as py_log
import python_logger.log_helper as py_log_always_on


MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)


python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)




import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse

from min_resource_percentage import MinResourcePercentage
from model_wrapper import ModelWrapper

from training_support import *
from losses import MultiClassDiceLoss, WeightedLosses

import ast





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process arguments that can change between trainings.")
    



    # Model specific arguments (because of default being different between different models you can't just copy them between models)

    parser.add_argument("-m", "--model", type=str, default="64_2_6", help='Model to use. Options: 64_2_6, 64_2_4')
    parser.add_argument("--bs", type=int, default=4, help='BATCH_SIZE')
    parser.add_argument("--nodw", type=int, default=4, help='NUM_OF_DATALOADER_WORKERS')
    parser.add_argument("--sd", type=str, default="UNet", help='SAVE_DIR')
    parser.add_argument("--ptd", type=str, default="./sclera_data", help='PATH_TO_DATA')
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument('--iw', type=int, default=256, help='Input width')
    parser.add_argument('--ih', type=int, default=256, help='Input height')
    parser.add_argument('--ic', type=int, default=3, help='Input channels')
    parser.add_argument('--oc', type=int, default=2, help='Output channels')
    parser.add_argument('--ds', type=str, default="augment", help='Dataset option. Options: augment, pass_through, partially_preaugmented.')
    parser.add_argument('--tesl', type=int, default=1e9, help=f"""TRAIN_EPOCH_SIZE_LIMIT. If we have 1500 images in the training set, and we set this to 1000, 
                        we will stop the epoch as we have trained on >= 1000 images.
                        We should watch out to use shuffle=True in the DataLoader, because otherwise we will always only train on the first 1000 images in the Dataset's ordering.

                        This is useful if we want to do rapid prototyping, and we don't want to wait for the whole epoch to finish.
                        Or if one epoch is too huge so it just makes more sense to train on a few smaller ones.
                        """)


    # General arguments

    parser.add_argument('--ips', type=int, default=1e9,
                        help='iter_possible_stop An optional positional argument with a default value of 1e9')
    parser.add_argument("--ntibp", type=int, default=10, help='NUM_TRAIN_ITERS_BETWEEN_PRUNINGS')
    parser.add_argument("--tras", type=int, default=-1, help="""Test run and size. If you pass an int, that will be the size of the dataset, and it will be in testrun. 
                        If -1 (default), then it is not a test run.""")

    
    # With action='store_true' these args store True if the flag is present and False otherwise.
    # Watch out with argparse and bool fields - they are always True if you give the arg a nonempty string.
    # So --pbop False would still give True to the pbop field.
    # This is why they are implemented this way now.
    parser.add_argument('-p', '--pruning_phase', action='store_true',
                        help='If present, enables pruning phase (automatic pruning)')
        

    # Add the optional arguments
    # setting error_ix: ix of the loss you want in the tuple: (CE_loss, approx_IoU, F1, IoU)
    # CE_loss only really makes sense - because if the model is a bit better, we are sure it goes down
    # (IoU and F1 are based on binary classification, so a slightly better model might still do the same predictions, so the loss would be the same - and so you can't choose which to clean away)
    parser.add_argument('--ce_ix', type=int, default=0,
                        help='cleaning error ix. We takeix of the loss you want in the tuple: (CE_loss, IoU, F1, IoU_as_avg_on_matrixes)')
    parser.add_argument("--ck", type=int, default=3, help="Cleanup k. When saving, how many models to keep. (k+1 models are kept if the current model is not in the best k - because we have to keep it to train the next one.)")
    parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations')
    parser.add_argument('--map', type=int, default=1e9, help='Max auto prunings')
    parser.add_argument('--nept', type=int, default=1,
                        help='Number of epochs per training iteration')
    
    parser.add_argument('--pbop', action='store_true',
                        help='Prune by original percent, otherwise by number of filters')
    parser.add_argument('--nftp', type=int, default=1,
                        help="""
                        !!! ONLY APPLIES IF --pbop IS FALSE !!!
                        Number of filters to prune in one pruning""")
    
    parser.add_argument('--pnkao', type=int, default=20, help="""
                        !!! THIS IS OVERRIDDEN IF --ifn IS 0 OR 1 !!!
                        It becomes 1. Because we need the new importances to be calculated after every pruning.

                        Prune n kernels at once - in one pruning iteration, we:
                        1. calculate the importance of all kernels
                        2. prune n kernels based on these importances
                        3. calculate the importances based on the new pruned model
                        4. prune n kernels based on these new importances
                        5. ...
                        Repeat until we have pruned the desired amount of kernels.

                        Then we go back to training the model until it is time for another pruning iteration.


                        In theory, it would be best to have --pnkao at 1, because we get the most accurate importance values.
                        However, this is very slow. And also it doesn't make that much of a difference in quality.
                        (would be better to do an actual retraining between the prunings then, 
                        since we are doing so many epoch passes it is basically computationally worse than retraining).

                        Also, you can do epoch_pass() on the validation set, not the training set, because it is faster.

                        If you are not using --pbop, then you can set --pnkao to 1e9.
                        Because the number of kernels we actually prune is capped by --nftp.
                        It will look like this:
                        1. calculate the importance of all kernels
                        2. prune --nftp kernels based on these importances
                        Done.

                        But if you are using --pbop, then you should set --pnkao to a number that is not too high.
                        Because we actually prune --pnkao kernels at once. And then we check if now we meet our resource goals.
                        So if you set it to 1e9, it will simply prune the whole model in one go.

                        """)
    parser.add_argument('--rn', type=str, default="flops_num", help='Resource name to prune by')
    parser.add_argument('--ptp', type=float, default=0.01, help="""Proportion of original {resource_name} to prune - actually, we don't just prune by this percent, because that get's us bad results.
                        Every time we prune, we prune e.g. 1 percent. Because of pnkao we overshoot by a little. So next time, if we prune by 1 percent again, we will overshoot by a little again, and the overshoots compound.
                        So we will instead prune in this way: get in which bracket of this percent we are so far (eg, we have 79.9 percent of original weights), then we will prune to 79 percent and pnkao will overshoot a little.
                        """)


    
    def custom_type_conversion(value):
        try:
            # Try to convert to int
            return int(value)
        except ValueError:
            try:
                # Try to convert to tuple using ast.literal_eval
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                raise argparse.ArgumentTypeError(f"Invalid value: {value}")
    parser.add_argument("--ifn", type=custom_type_conversion, default=(0.5, 0.5, 0.5), help="Importance func. If 0, random pruning. If 1, uniform pruning. If tuple, we get IPAD with those 3 alphas.")

    parser.add_argument("--tp", action="store_true", help="""test pruning. 
                        This makes it so all the conv layers have 0.999999 as their limit for weights. 
                        MIND THAT input slice pruning also affects the weights - so this generally means each layer will get one kernel OR one input slice pruned.
                        The uniform pruning starts at CURR_PRUNING_IX - so if you want the other half of layers to have their kernels pruned, just change that to 1.""")
    parser.add_argument('--optim', type=str, default="Adam", help='Optimizer used. Adam, SGD, LBFGS')
    parser.add_argument("--loss_fn_name", type=str, default="Default", help="""Loss function used. Default (MCDL), 
                        MCDL for MultiClassDiceLoss, CE for CrossEntropyLoss,
                        CEHW for CrossEntropyLoss_hardcode_weighted,
                        MCDL_CEHW_W for MultiClassDiceLoss and CrossEntropyLoss_hardcode_weighted in a weighted pairing of both losses,
                        MCDLW for MultiClassDiceLoss with background adjustment,
                        """)

    def isfloat(s):
        """
        Checks if a string represents a valid float number.
        
        Args:
            s (str): The input string to check.
        
        Returns:
            bool: True if the string represents a valid float, False otherwise.
        """
        # Remove leading and trailing whitespace
        s = s.strip()
        
        # Handle empty string
        if not s:
            return False
        
        # Allow for negative sign
        if s.startswith('-'):
            s = s[1:]
        
        # Check if the remaining part is either a digit or a single decimal point
        return s.replace('.', '', 1).isdigit()

    def list_conversion(list_str):
        if isfloat(list_str):
            return [float(list_str)]
        return ast.literal_eval(list_str)
    
    parser.add_argument("--alphas", type=list_conversion, default=[], help="Alphas used in loss_fn. Currently only one is used. If there is just one alpha, you can just pass a float as an arg, like: 0.8.")


    args = parser.parse_args()

    MODEL = args.model
    BATCH_SIZE = args.bs
    NUM_OF_DATALOADER_WORKERS = args.nodw
    SAVE_DIR = args.sd
    PATH_TO_DATA = args.ptd
    LEARNING_RATE = args.lr
    INPUT_WIDTH = args.iw
    INPUT_HEIGHT = args.ih
    INPUT_CHANNELS = args.ic
    OUTPUT_CHANNELS = args.oc
    DATASET = args.ds
    TRAIN_EPOCH_SIZE_LIMIT = args.tesl
    

    iter_possible_stop = args.ips
    NUM_TRAIN_ITERS_BETWEEN_PRUNINGS = args.ntibp
    TEST_RUN_AND_SIZE = args.tras
    IS_TEST_RUN = TEST_RUN_AND_SIZE != -1

    is_pruning_ph = args.pruning_phase
    prune_by_original_percent = args.pbop

    cleaning_err_ix = args.ce_ix
    cleanup_k = args.ck
    max_train_iters = args.mti
    max_auto_prunings = args.map
    num_ep_per_iter = args.nept

    prune_n_kernels_at_once = args.pnkao
    num_to_prune = args.nftp
    resource_name = args.rn
    proportion_to_prune = args.ptp

    TEST_PRUNING = args.tp
    IMPORTANCE_FN_DEFINER = args.ifn

    optimizer = args.optim
    loss_fn_name = args.loss_fn_name
    alphas = args.alphas


    if IMPORTANCE_FN_DEFINER == 0 or IMPORTANCE_FN_DEFINER == 1:
        prune_n_kernels_at_once = 1




    if DATASET == "partially_preaugmented":
        raise NotImplementedError("partially_preaugmented dataset not in use anymore.")
        from dataset_partial_preaug import IrisDataset, transform
    elif DATASET == "augment":
        from dataset_aug import IrisDataset, transform
    elif DATASET == "pass_through":
        raise NotImplementedError("pass_through dataset not in use anymore.")
        from dataset_pass_through import IrisDataset, transform
    else:
        raise ValueError(f"DATASET not recognized: {DATASET}.")




    # print("Currently disregarding the args. They are hardcoded in the script.")

    # is_pruning_ph = IS_PRUNING_PH
    # prune_n_kernels_at_once = PRUNE_N_KERNELS_AT_ONCE
    # prune_by_original_percent = PRUNE_BY_ORIGINAL_PERCENT
    # max_train_iters = MAX_TRAIN_ITERS
    # max_auto_prunings = MAX_AUTO_PRUNINGS
    # err_ix = ERR_IX
    # resource_name = RESOURCE_NAME
    # proportion_to_prune = PROPORTION_TO_PRUNE





    pruning_kwargs = {
        "prune_by_original_percent": prune_by_original_percent,
        "prune_n_kernels_at_once": prune_n_kernels_at_once,
        "num_of_prunes": num_to_prune,
        "resource_name": resource_name,
        "original_proportion_to_prune": proportion_to_prune
    }

    print(f"Validation phase: {is_pruning_ph}")
    print(args)



# # main changable parameters between trainings:
# SAVE_DIR = "UNet"
# IS_TEST_RUN = True
# PATH_TO_DATA = "./vein_sclera_data"
# NUM_OF_DATALOADER_WORKERS = 1
# BATCH_SIZE = 4


# program args that could be hardcoded here:

# IS_PRUNING_PH = True
# MAX_TRAIN_ITERS = 100 # change this to 10e9 when doing the pruning phase
# MAX_AUTO_PRUNINGS = 70 # to get to 70 percent of the starting flops

# # IS_PRUNING_PH to PROPORTION_TO_PRUNE are actually args you can pass.
# # But they are hardcoded here for the sake of the experiment. Just makes it less room for error.

# # Main parameeters you don't change after training.

# PRUNE_BY_ORIGINAL_PERCENT = True
# ERR_IX = 3
# RESOURCE_NAME = "flops_num"
# PRUNE_N_KERNELS_AT_ONCE = 20
# PROPORTION_TO_PRUNE = 0.01









# save_path = osp.join(osp.dirname(__file__), "UNet")
save_path = osp.join(".", SAVE_DIR)

main_save_path = osp.join(save_path, "saved_main")


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# self.device = "cpu" # for debugging purposes
print(f"Device: {device}")








# CE_weights = torch.tensor([0.1, 6.5])
CE_weights = torch.tensor([0.1, 50.0])
CE_weights = CE_weights / CE_weights.mean() # to make the loss more interpretable and graphable
CE_weights = CE_weights.to(device)



if loss_fn_name == "Default":
    loss_fn_name = "MCDL"

if loss_fn_name == "MCDL":
    loss_fn = MultiClassDiceLoss()
elif loss_fn_name == "CE":
    loss_fn = nn.CrossEntropyLoss()

elif loss_fn_name == "CEHW":

    # There are 65000 pixels in the image. Only like 700 are 1s. 
    # So to make 0s and 1s equally important, we would have to roughly [0.1, 6.5]
    # And even then - we should give more priority to the 1s, because they are more important. 
    # And we really want to increase how much we decide on 1s.

    # The mean of the weights should be 1.0, so that loss remains more interpretable and graphable
    # CE loss is between 0 and 1 in the 2 class case if the model is remotely okay. 
    # The weights multiply the error for each class, so if the mean is 1, i think the loss is between 0 and 1.

    loss_fn = nn.CrossEntropyLoss(weight=CE_weights)

elif loss_fn_name == "MCDL_CEHW_W":
    losses = [MultiClassDiceLoss(), nn.CrossEntropyLoss(weight=CE_weights)]
    weights = [0.5, 0.5]
    loss_fn = WeightedLosses(losses, weights)

elif loss_fn_name == "MCDLW":
    loss_fn = MultiClassDiceLoss(background_adjustment=alphas[0])

else:
    raise ValueError("Loss function not recognized.")





if optimizer == "Adam":
    optimizer = torch.optim.Adam
elif optimizer == "SGD":
    optimizer = torch.optim.SGD
elif optimizer == "LBFGS":
    raise NotImplementedError("LBFGS is not implemented.")
    optimizer = torch.optim.LBFGS
else:
    raise ValueError("Optimizer not recognized.")


learning_parameters = {
    "learning_rate" : LEARNING_RATE,
    "loss_fn" : loss_fn,
    "optimizer_class" : optimizer,
    "train_epoch_size_limit" : TRAIN_EPOCH_SIZE_LIMIT
}


# In our UNet implementation the dims can be whatever you want.
# You could even change them between training iterations - but it might be a bad idea because all the weights had been learnt at the scale of the previous dims.
INPUT_DIMS = {
    "width" : INPUT_WIDTH,
    "height" : INPUT_HEIGHT,
    "channels" : INPUT_CHANNELS
}

# In our UNet the output width and height have to be the same as the input width and height. 
OUTPUT_DIMS = {
    "width" : INPUT_DIMS["width"],
    "height" : INPUT_DIMS["height"],
    "channels" : OUTPUT_CHANNELS
}


dataloading_args = {


    "testrun" : IS_TEST_RUN,
    "testrun_size" : TEST_RUN_AND_SIZE,
   

    "input_width" : INPUT_DIMS["width"],
    "input_height" : INPUT_DIMS["height"],
    "output_width" : OUTPUT_DIMS["width"],
    "output_height" : OUTPUT_DIMS["height"],
    
    # iris dataset params
    "path_to_sclera_data" : PATH_TO_DATA,
    "transform" : transform,
    "n_classes" : OUTPUT_DIMS["channels"],

    # DataLoader params
    # Could have separate "train_batch_size" and "eval_batch_size" (for val and test)
    #  since val and test use torch.no_grad() and therefore use less memory. 
    "batch_size" : BATCH_SIZE,
    "shuffle" : False, # TODO shuffle??
    "num_workers" : NUM_OF_DATALOADER_WORKERS,
}


def get_data_loaders(**dataloading_args):
    
    data_path = dataloading_args["path_to_sclera_data"]
    # n_classes = 4 if 'sip' in args.dataset.lower() else 2

    print('path to file: ' + str(data_path))

    train_dataset = IrisDataset(filepath=data_path, split='train', **dataloading_args)
    valid_dataset = IrisDataset(filepath=data_path, split='val', **dataloading_args)
    test_dataset = IrisDataset(filepath=data_path, split='test', **dataloading_args)

    trainloader = DataLoader(train_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    validloader = DataLoader(valid_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # I'm not sure why we're dropping last, but okay.

    # Actually, no. Let's not drop last.
    # it makes no sense. i think this might have been done because the IPAD fn was done only on the last batch, and so that
    # batch needed to be big.


    print('train dataset len: ' + str(train_dataset.__len__()))
    print('val dataset len: ' + str(valid_dataset.__len__()))
    print('test dataset len: ' + str(test_dataset.__len__()))

    print('train dataloader num of batches: ' + str(trainloader.__len__()))
    print('val dataloader num of batches: ' + str(validloader.__len__()))
    print('test dataloader num of batches: ' + str(testloader.__len__()))

    
    return trainloader, validloader, testloader






train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(**dataloading_args)# 

dataloader_dict = {
    "train" : train_dataloader,
    "validation" : valid_dataloader,
    "test" : test_dataloader,
}






from unet_original import UNet

if MODEL == "64_2_4":

    model_parameters = {
        # layer sizes
        "output_y" : OUTPUT_DIMS["height"],
        "output_x" : OUTPUT_DIMS["width"],
        "n_channels" : INPUT_DIMS["channels"],
        "n_classes" : OUTPUT_DIMS["channels"],
        "starting_kernels" : 64,
        "expansion" : 2,
        "depth" : 4,
    }

elif MODEL == "64_2_6":

    model_parameters = {
        # layer sizes
        "output_y" : OUTPUT_DIMS["height"],
        "output_x" : OUTPUT_DIMS["width"],
        "n_channels" : INPUT_DIMS["channels"],
        "n_classes" : OUTPUT_DIMS["channels"],
        "starting_kernels" : 64,
        "expansion" : 2,
        "depth" : 6,
    }

else:
    raise ValueError(f"MODEL not recognized: {MODEL}.")



INPUT_EXAMPLE = torch.randn(1, INPUT_DIMS["channels"], INPUT_DIMS["height"], INPUT_DIMS["width"])
















if __name__ == "__main__":

    
    model_wrapper = ModelWrapper(UNet, model_parameters, dataloader_dict, learning_parameters, INPUT_EXAMPLE, save_path, device)








    tree_ix_2_name = model_wrapper.get_tree_ix_2_name()


    # If you change FLOPS_min_res_percents and weights_min_res_percents 
    # or other disallowments
    # between runnings of main, 
    # the new onew will be used. So you can have an effect on your training by doing this.

    







    def validation_stop(training_logs: TrainingLogs, pruning_logs: PruningLogs, curr_train_iter, initial_train_iter):
        # returns True when you should stop

        # initial_train_iter is the train_iter when we ran the program this time around
        # - so if you perhaps wanted at least 3 train iters every time you run the program, you would do:
        # if curr_train_iter - initial_train_iter < 3:
        #    return False


        # val_errors = [item[0] for item in training_logs.errors]

        # If there have been no prunings, prune.
        if len(pruning_logs.pruning_logs) == 0:
            return True

        last_pruning_train_iter = pruning_logs.pruning_logs[-1][0]



        # This only makes sense for how we are designing our experiment. This is the only way we can compare methods for pruning kernel-selection.
        # Our idea is: you train the model, then you want it to have 25% the amount of flops.
        # But how do you choose which 75% of filters to prune?
        # Well, you prune 1%, then retrain, then prune 1%, then retrain, and so on until you get to 75%.
        # How you choose the 1% is the question. We are comparing different methods of choosing the 1%.
        
        # And since we are comparing different methods, we want to compare them on the same number of train iters between prunings.

        if (curr_train_iter - last_pruning_train_iter) >= NUM_TRAIN_ITERS_BETWEEN_PRUNINGS:
            return True
        
        return False



        # Older idea of dynamic decision of when to prune:
        """
        if len(val_errors) < 3:
            return False
        
        if len(val_errors) >= 25:
            return True
        
        returner = val_errors[-1] > val_errors[-2] and val_errors[-1] > val_errors[-3]

        # if previous metric doesn't say we should return, we also go check another metric:
        # if the current validation error is higher than either of the 4. and 5. back
        # we should stop. Because it means we are not improving.
        if not returner and len(val_errors) >= 5:
            returner = val_errors[-1] > val_errors[-4] or val_errors[-1] > val_errors[-5]
            
        return returner
        """





    
    train_automatically(model_wrapper, main_save_path, validation_stop, max_training_iters=max_train_iters, max_auto_prunings=max_auto_prunings, train_iter_possible_stop=iter_possible_stop, pruning_phase=is_pruning_ph, cleaning_err_ix=cleaning_err_ix, cleanup_k=cleanup_k,
                         num_of_epochs_per_training=num_ep_per_iter, pruning_kwargs_dict=pruning_kwargs)








