

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

from training_support import train_automatically, TrainingLogs, PruningLogs
from losses import MultiClassDiceLoss, WeightedLosses

import ast





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process arguments that can change between trainings.")
    



    # Model specific arguments (because of default being different between different models you can't just copy them between models)

    parser.add_argument("-m", "--model", type=str, default="SegNet_256x256", help='Model to use. Options: SegNet_256x256, SegNet_3000x2000')
    parser.add_argument("--bs", type=int, default=16, help='BATCH_SIZE')
    parser.add_argument("--nodw", type=int, default=10, help='NUM_OF_DATALOADER_WORKERS')
    parser.add_argument("--sd", type=str, default="SegNet", help='SAVE_DIR')
    parser.add_argument("--ptd", type=str, default="./sclera_data", help='PATH_TO_DATA')
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    parser.add_argument('--iw', type=int, default=256, help='Input width')
    parser.add_argument('--ih', type=int, default=256, help='Input height')
    parser.add_argument('--ic', type=int, default=3, help='Input channels')
    parser.add_argument('--oc', type=int, default=2, help='Output channels')
    parser.add_argument('--ds', type=str, default="standard", help='Dataset option. Options: augment, preaugmented, partially_preaugmented.')
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

    
    # With action='store_true' these args store True if the flag is present and False otherwise.
    # Watch out with argparse and bool fields - they are always True if you give the arg a nonempty string.
    # So --pbop False would still give True to the pbop field.
    # This is why they are implemented this way now.
    parser.add_argument("-t", "--is_test_run", action='store_true',
                        help='If present, enables test run')
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

    IS_TEST_RUN = args.is_test_run
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
        from dataset_partial_preaug import IrisDataset, transform
    elif DATASET == "augment":
        from dataset_aug import IrisDataset, transform
    elif DATASET == "preaugmented":
        from dataset_preaug import IrisDataset, transform
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


# In our SegNet implementation the dims can be whatever you want.
# You could even change them between training iterations - but it might be a bad idea because all the weights had been learnt at the scale of the previous dims.
INPUT_DIMS = {
    "width" : INPUT_WIDTH,
    "height" : INPUT_HEIGHT,
    "channels" : INPUT_CHANNELS
}

# In our SegNet the output width and height have to be the same as the input width and height. 
OUTPUT_DIMS = {
    "width" : INPUT_DIMS["width"],
    "height" : INPUT_DIMS["height"],
    "channels" : OUTPUT_CHANNELS
}


dataloading_args = {


    "testrun" : IS_TEST_RUN,
    "testrun_size" : 30,
   

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






if MODEL == "SegNet_256x256":
    from segnet import SegNet

    model_parameters = {
        "in_chn" : INPUT_DIMS["channels"],
        "out_chn" : OUTPUT_DIMS["channels"],
    }

elif MODEL == "SegNet_3000x2000":
    from segnet import SegNet

    model_parameters = {
        # layer sizes
        "output_y" : OUTPUT_DIMS["height"],
        "output_x" : OUTPUT_DIMS["width"],
        "expansion" : 1.3,
        "starting_kernels" : 5,
        "in_chn" : INPUT_DIMS["channels"],
        "out_chn" : OUTPUT_DIMS["channels"],
    }

else:
    raise ValueError(f"MODEL not recognized: {MODEL}.")



INPUT_EXAMPLE = torch.randn(1, INPUT_DIMS["channels"], INPUT_DIMS["height"], INPUT_DIMS["width"])








"""
THIS HERE IS THE START OF BUILDING A CONNECTION fn
based on the _get_next_conv_id_list_recursive()
It is very early stage.
"""






def segnet_input_slice_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, initial_kernel_ix) -> [(goal_tree_ix_1, goal_initial_input_slice_ix_1), (goal_tree_ix_2, goal_initial_input_slice_ix_2),...]

    # TL;DR -  for skip connections, where the input channels need to be pruned, because the output channels of this layer were pruned
    
    # This functions takes the tree_ix and the ix of where the kernel we are concerned with was in the model initially (before pruning).
    # And it returns a list of tuples giving the following modules tree_ixs and the input_slice_ix
    # (where the effect of the above-mentioned kernel is in the input tensor) in the initial model (before pruning).


    conn_destinations = []

    # we kind of only care about convolutional modules.
    # We just need to prune there (and possibly something with the batch norm layer)
    # So it would make sense to transform the tree_ix to the ordinal number of 
    # the convolutional module, and work with that ix instead.



    # doesn't have skip connections, so we only need to prunte the input slice of the following layer

    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)
        conn_destinations.append((conv_tree_ixs[conv_ix+1], kernel_ix))

    
    
    return conn_destinations


    



def segnet_kernel_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
    # f(tree_ix, real_kernel_ix) -> [(goal_tree_ix_1, goal_real_kernel_ix_1), (goal_tree_ix_2, goal_real_kernel_ix_2),...]
    
    # This functions takes the tree_ix and the ix of where the kernel we are concerned with was in the model RIGHT NOW, NOT INITIALLY.
    # And it returns a list of tuples giving the tree_ixs and "kernel_ixs" in the model RIGHT NOW, NOT INITIALLY.
    # for layers which are inextricably linked with the convolutional layer.

    # Meant for batchnorm and special cases.

    # Inextricably linked are in direct connection with the conv's current (not intitial) kernel_ix, so they don't need the more complex fn.
    # We could have treated them in the regular way (through initial ixs), but this way is better,
    # because, in the pruner, we don't need to keep track of the initial ixs (although we do anyways for accounting reasons).
    # Also it's simpler and conceptually makes more sense - which is the main reason.

    # The batchnorm is such a layer - for it, the "kernel_ix" isn't really a kernel ix.
    # It is, however, the position we need to affect due to pruning the kernel_ix in the convolutional layer.
    # There are possibly more such layers and more types of such layers, so we made this function more general.



    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # This is the additional way that SegNet needs to be pruned besides batchnorms.

    # CONV DE 12 JE TREBA INEEXTRICABLY PRUNEAT, če prunaš 0-ti conv v networku.
    # Pač ta princip je, da mora izhod tega dela potem bit enak, da lahko unpool dela s tem.
    # Ker unpool tu dela z incices. Ne postavi vseh zadev equally spaced in samo dela bilinear interpolacijo.
    # Ampak postavi številke na iste pozicije kot so bile v poolingu. In potem naredi interpolacijo.

    # In saj v height in width smereh mi nič ne prunamo. Tako da sam mehanizem tega nas ne rabi skrbet.
    # Samo pač input poolinga in output poolinga imata neko število kernelov. In tako število kernelov imajo tudi indices.
    # In zato mora vhod v unpool tudi imeti toliko kernelov.

    # In zato se zgodi ta inextricable connection.



    # So when we prune the layer right before a pooling, we have to prune the layer right before the corresonding unpoolong.

    # Pairs of conv ixs:
    # 1 23
    # 3 21
    # 6 18
    # 9 15










    # doesn't have skip connections
    # Only prune batchnorm
    
    conn_destinations = []

    LLM_ix = None
    if tree_ix in lowest_level_modules:
        LLM_ix = lowest_level_modules.index(tree_ix)


    # conv_ix = None
    if tree_ix in conv_tree_ixs:

        conv_ix = conv_tree_ixs.index(tree_ix)

        # # OUTC MUSTN'T BE PRUNED ANYWAY!!!!!!!!, BECAUSE IT IS THE OUTPUT OF THE NETWORK
        # # out.conv doesn't have a batchnorm after it.
        # if conv_ix < 18:

        conn_destinations.append((lowest_level_modules[LLM_ix+1], kernel_ix))

        if conv_ix == 1:
            conn_destinations.append((conv_tree_ixs[23], kernel_ix))
        elif conv_ix == 3:
            conn_destinations.append((conv_tree_ixs[21], kernel_ix))
        elif conv_ix == 6:
            conn_destinations.append((conv_tree_ixs[18], kernel_ix))
        elif conv_ix == 9:
            conn_destinations.append((conv_tree_ixs[15], kernel_ix))



    # for batchnorm, conn_destinations is simply empty
    

    
    return conn_destinations




# When each batch is processed, the averaging_objects function is called.
# Here you define how you would like to create your averaging objects through one epoch of training.
# This function shows how we would like to update our average of the activations (outputs)
# for the convolutional layers (because in the background this is only set for convolutional layers).
# At each iteration the mean is corrects so far. So at the end the mean is also correct.
# It is better to train with larger batch sizes so numerical errors of the iterative mean calculation are smaller.

# Proof:
# The first mean is correct so far. It is avg_0 = \sum x_i / n_0 where n_0 is the number of elements of the 0-th iteration.
# by the same logic, avg_1 is also correct (the average of just the next batch).
# The second mean avg_{1,2} is (n_0 * avg _0 + n_1 * avg_1) / (n_0 + n_1) = 
# (n_0 * (\sum x_i / n_0) + n_1 * (\sum x_j / n_1)) / (n_0 + n_1) =
# ( \sum x_i + \sum x_j ) / (n_0 + n_1)
# # Which is the correct mean of all the elements. By induction, the same logic applies to all iterations.  

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# IF USING INPUT OR MODULE WEIGHTS, YOU HAVE TO DETACH THEM!!!!!
# Also, input is a tuple, so you have to figure out what it really is first - I haven't looked into it.
# The output has already been detached, so we don't need to worry about backpropagation.
# You can do .detach() again, which won't change anything, it's idempotent.
# If they weren't detached, they remain in the computational graph and keep being in the gradient calculation during loss.backward().
# Because of pruning, this shows an error like so:
#  File "/home/matevzvidovic/Desktop/Diplomska/Prototip/Delo/TrainingWrapper.py", line 424, in train
#     loss.backward()
#   File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/_tensor.py", line 522, in backward
#     torch.autograd.backward(
#   File "/home/matevzvidovic/.local/lib/python3.10/site-packages/torch/autograd/__init__.py", line 266, in backward
#     Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# RuntimeError: Function ConvolutionBackward0 returned an invalid gradient at index 1 - got [128, 255, 3, 3] but expected shape compatible with [128, 256, 3, 3]

# We would like to also use weights in our importance calculation.
# The easiest and conceptually best place to put them is in the averaging function (outside of making their own function).
# It doesn't make sense to average them, so we would just save them when the first average is made.


INITIAL_AVG_OBJECT = (0, None, None)
def averaging_function(module, input, output, prev_avg_object):
    
    batch_size = output.shape[0]
    batch_mean = output.mean(dim=(0))

    if prev_avg_object[1] is None:
        new_avg_object = (batch_size, batch_mean, module.weight.data.detach().clone())
        return new_avg_object

    new_avg_object = (prev_avg_object[0] + batch_size, 
                      (prev_avg_object[0] * prev_avg_object[1] + batch_size * batch_mean) / (prev_avg_object[0] + batch_size),
                        prev_avg_object[2])

    return new_avg_object 


# averaging_mechanism = {
#     "initial_averaging_object" : INITIAL_AVG_OBJECT,
#     "averaging_function" : averaging_function
# }



# An additional function could be applied in between the averaging function and the importance function.
# If we were, for example, interested in a specific interaction between the active outputs (not the averaged ones)
# with our averaging object. For example, to calculate the correlation between the output and our average activations.
# Then the averaging function would be applied in the first pass through the network and we would make our averaging objects.
# Then this middle function would be used and we would calculate our batch_importances (correlations) for each batch.
# Then the final importance function we see below us would only be used to combine these batch_importances.
# For example to average them or to sum them.
# But this is not currently implemented.





def IPAD_kernel_importance_fn_generator(L1_ADC_weight):
    assert L1_ADC_weight > 0 and L1_ADC_weight < 1, "L1_ADC_weight must be between 0 and 1."
    
    
    def IPAD_kernel_importance_fn(averaging_objects: dict, conv_tree_ixs):
        # Returns dict tree_ix_2_list_of_kernel_importances
        # The ix-th importance is for the kernel currently on the ix-th place.
        # To convert this ix to the initial unpruned models kernel ix, use the pruner's
        # state of active kernels.

        tree_ix_2_kernels_importances = {}
        for tree_ix in conv_tree_ixs:

            kernels_average_activation = averaging_objects[tree_ix][1]
            # print(kernels_average_activation.shape)
            # print(kernels_average_activation)
            overall_average_activation = kernels_average_activation.mean(dim=(0))
            # print(overall_average_activation)
            # print(overall_average_activation.shape)
            # print(overall_average_activation)
            h = kernels_average_activation.shape[1]
            w = kernels_average_activation.shape[2]
            diff = kernels_average_activation - overall_average_activation
            L1_ADC = torch.abs(diff).sum(dim=(1,2)) / (h*w)
            L2_ADC = (diff).pow(2).sum(dim=(1,2)).sqrt() / (h*w)
            kernels_importances = L1_ADC_weight * L1_ADC + (1 - L1_ADC_weight) * L2_ADC
            # print(f"L1_ADC: {L1_ADC}")
            # print(f"L2_ADC: {L2_ADC}")
            # print(kernels_importances.shape)
            # print(kernels_importances)

            tree_ix_2_kernels_importances[tree_ix] = kernels_importances
        
        
        return tree_ix_2_kernels_importances
        
    
    return IPAD_kernel_importance_fn




def weights_importance_fn_generator(L1_over_L2_alpha):
    assert L1_over_L2_alpha > 0 and L1_over_L2_alpha < 1, "L1_over_L2_alpha must be between 0 and 1."
    
    def weights_importance_fn(averaging_objects: dict, conv_tree_ixs):
        # Returns dict tree_ix_2_list_of_kernel_importances
        # The ix-th importance is for the kernel currently on the ix-th place.
        # To convert this ix to the initial unpruned models kernel ix, use the pruner's
        # state of active kernels.

        tree_ix_2_kernels_importances = {}
        for tree_ix in conv_tree_ixs:
            
            # [num_of_kernels, depth, h, w]
            kernels_weights = averaging_objects[tree_ix][2]
            overall_weights = kernels_weights.mean(dim=(0))
            d = kernels_weights.shape[1]
            h = kernels_weights.shape[2]
            w = kernels_weights.shape[3]
            L1 = torch.abs(kernels_weights - overall_weights).sum(dim=(1,2,3)) / (d*h*w)
            L2 = (kernels_weights - overall_weights).pow(2).sum(dim=(1,2,3)).sqrt() / (d*h*w)
            kernels_importances = L1_over_L2_alpha * L1 + (1 - L1_over_L2_alpha) * L2

            tree_ix_2_kernels_importances[tree_ix] = kernels_importances
        
        
        return tree_ix_2_kernels_importances
        
    
    return weights_importance_fn



def IPAD_and_weights(IPAD_over_weights_alpha, IPAD_L1_ADC_weight, weights_L1_over_L2_alpha):
    assert IPAD_over_weights_alpha > 0 and IPAD_over_weights_alpha < 1, "IPAD_over_weights_alpha must be between 0 and 1."

    IPAD_fn = IPAD_kernel_importance_fn_generator(IPAD_L1_ADC_weight)
    weights_fn = weights_importance_fn_generator(weights_L1_over_L2_alpha)

    def joined_imporance_fn(averaging_objects: dict, conv_tree_ixs):
        IPAD_importances = IPAD_fn(averaging_objects, conv_tree_ixs)
        weights_importances = weights_fn(averaging_objects, conv_tree_ixs)

        joined_importances = {}
        for tree_ix in conv_tree_ixs:
            joined_importances[tree_ix] = IPAD_over_weights_alpha * IPAD_importances[tree_ix] + (1 - IPAD_over_weights_alpha) * weights_importances[tree_ix]

        return joined_importances

    return joined_imporance_fn



def random_pruning_importance_fn(averaging_objects: dict, conv_tree_ixs):
    tree_ix_2_kernel_importances = {}
    for tree_ix in conv_tree_ixs:
        num_of_kernels = averaging_objects[tree_ix][1].shape[0]
        kernel_importance = torch.rand(num_of_kernels)
        tree_ix_2_kernel_importances[tree_ix] = kernel_importance

    return tree_ix_2_kernel_importances




# Da imamo najmanjše importance v layerju, čigar curr_conv_ix (ix v conv_tree_ixs) je enak oziroma njabližje CURR_PRUNING_IX.
# Znotraj layerja pa imajo kernels v V shapeu - da se vedno na sredini prunea (saj uniform pruning bi bil, da vedno 0-tega prunaš.- Ampak mi ni všeč, da se vedno the edge one prunea. Raje da vedno the middle one.)
# Za posamezen layer določimo oddaljenost od trenutnega pruninga:
# curr_dist = abs(curr_conv_ix - CURR_PRUNING_IX)
# Naredi torej recimo, da kernel importances iz sredine proti robu rastejo med:
# curr_dist in curr_dist+1.

CURRENT_PRUNING_IX = 0
def uniform_random_pruning_importance_fn(averaging_objects: dict, conv_tree_ixs):

    global CURRENT_PRUNING_IX

    tree_ix_2_kernel_importances = {}
    for ix, tree_ix in enumerate(conv_tree_ixs):
        
        num_of_kernels = averaging_objects[tree_ix][1].shape[0]
        curr_dist = abs(ix - CURRENT_PRUNING_IX)

        middle_kernel_ix = num_of_kernels // 2
        ixs = torch.arange(num_of_kernels)
        kernel_distances = torch.abs(ixs - middle_kernel_ix)
        
        # should look sth like: [1.0, 0.97,...,0.0, 0.02, ... 1.0]
        base_importances = kernel_distances.float() / kernel_distances.max().float()
        # and now we put them in the right bracket based on distance of the layer from the current pruning ix
        final_importances = base_importances + curr_dist
        
        tree_ix_2_kernel_importances[tree_ix] = final_importances
    
    CURRENT_PRUNING_IX += 1
    if CURRENT_PRUNING_IX >= len(conv_tree_ixs):
        CURRENT_PRUNING_IX = 0


    return tree_ix_2_kernel_importances






if IMPORTANCE_FN_DEFINER == 0:
    IMPORTANCE_FN = random_pruning_importance_fn
elif IMPORTANCE_FN_DEFINER == 1:
    IMPORTANCE_FN = uniform_random_pruning_importance_fn
else:
    IMPORTANCE_FN = IPAD_and_weights(*IMPORTANCE_FN_DEFINER)










def set_averaging_objects_hooks(model_wrapper, initial_averaging_object, averaging_function, averaging_objects: dict, resource_calc, tree_ixs: list):
        
    
    def get_activation(tree_ix):
        
        def hook(module, input, output):
            
            detached_output = output.detach()

            if tree_ix not in averaging_objects:
                averaging_objects[tree_ix] = initial_averaging_object

            averaging_objects[tree_ix] = averaging_function(module, input, detached_output, averaging_objects[tree_ix])

        return hook

    tree_ix_2_hook_handle = {}
    for tree_ix in tree_ixs:
        module = resource_calc.module_tree_ix_2_module_itself[tree_ix]
        tree_ix_2_hook_handle[tree_ix] = module.register_forward_hook(get_activation(tree_ix))
    
    model_wrapper.tree_ix_2_hook_handle = tree_ix_2_hook_handle
    



def remove_hooks(model_wrapper):
    
    if model_wrapper.tree_ix_2_hook_handle is None:
        raise ValueError("In remove_hooks: model_wrapper.tree_ix_2_hook_handle is already None")
    
    for hook_handle in model_wrapper.tree_ix_2_hook_handle.values():
        hook_handle.remove()
    
    model_wrapper.tree_ix_2_hook_handle = None


def get_importance_dict(model_wrapper: ModelWrapper):

    model_wrapper.averaging_objects = {}
    set_averaging_objects_hooks(model_wrapper, INITIAL_AVG_OBJECT, averaging_function, model_wrapper.averaging_objects, model_wrapper.resource_calc, model_wrapper.conv_tree_ixs)

    model_wrapper.epoch_pass(dataloader_name="train")
    # maybe doing this on val, because it is faster and it kind of makes more sense
    # model_wrapper.epoch_pass(dataloader_name="validation")

    # pruner needs the current state of model resources to know which modules shouldn't be pruned anymore
    model_wrapper.resource_calc.calculate_resources(model_wrapper.input_example)

    importance_dict = IMPORTANCE_FN(model_wrapper.averaging_objects, model_wrapper.conv_tree_ixs)
    remove_hooks(model_wrapper)
    model_wrapper.averaging_objects = {}

    return importance_dict


def dummy_get_importance_dict(model_wrapper: ModelWrapper):
    # This is used for uniform and random pruning. Because we don't actually need to do the epoch pass.
    # And --pnkao needs to be 1 so we get the importance dict after pruning each kernel. So epoch pass would be too slow.
    
    fake_avg_objects = {}
    for tree_ix in model_wrapper.conv_tree_ixs:
        weight_dims = model_wrapper.resource_calc.module_tree_ix_2_module_itself[tree_ix].weight.data.detach().size()
        fake_avg_objects[tree_ix] = (None, torch.zeros(weight_dims), None)


    importance_dict = IMPORTANCE_FN(fake_avg_objects, model_wrapper.conv_tree_ixs)
    return importance_dict



GET_IMPORTANCE_DICT_FN = get_importance_dict
if IMPORTANCE_FN_DEFINER == 0 or IMPORTANCE_FN_DEFINER == 1:
    GET_IMPORTANCE_DICT_FN = dummy_get_importance_dict




if __name__ == "__main__":

    
    model_wrapper = ModelWrapper(SegNet, model_parameters, dataloader_dict, learning_parameters, INPUT_EXAMPLE, save_path, device)

    # Go see model graph to help you choose the right layers to prune.
    # model_wrapper.model_graph()







    tree_ix_2_name = model_wrapper.get_tree_ix_2_name()


    # If you change FLOPS_min_res_percents and weights_min_res_percents 
    # or other disallowments
    # between runnings of main, 
    # the new onew will be used. So you can have an effect on your training by doing this.

    


    


    # Here we abuse the min_res_percentage class to disallow certain prunings.
    # Both for general disallowments and for choice disallowments
    # (only disallowed to be chosen for pruning, but still allowed to be pruned as a consequence of another pruning (through the kernel_connection_fn)).

    # Important disallowing:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # OUTCONV HAS TO BE DISALLOWED FROM PRUNING!!!!!!!
    # Because otherwise your num of classes of the output (pred) will change.
    # Otherwise you get "../aten/src/ATen/native/cuda/NLLLoss2d.cu:104: nll_loss2d_forward_kernel: block: [0,0,0], thread: [154,0,0] Assertion `t >= 0 && t < n_classes` failed."
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    generally_disallowed = MinResourcePercentage(tree_ix_2_name)

    disallowed_dict = {
        model_wrapper.conv_tree_ixs[25] : 1.1
    }
    generally_disallowed.set_by_tree_ix_dict(disallowed_dict)




    # Choice disallowing:
    # (only disallowed to be chosen for pruning, but still allowed to be pruned as a consequence of another pruning (through the kernel_connection_fn)).
    conv_tree_ixs = model_wrapper.conv_tree_ixs
    CHOICE_DISALLOWED_CONV_IXS = [15, 18, 21, 23]
    # The reasoning for this choice comes from kernel_connection_fn:
    # Because this then means, that [15, 18, 21, 23] haveto be disallowed to be chosen for pruning.
    # Because the kernel nums must match.
    """
    # So when we prune the layer right before a pooling, we have to prune the layer right before the corresonding unpoolong.

    # Pairs of conv ixs:
    # 1 23
    # 3 21
    # 6 18
    # 9 15
    """
    choice_disallowed = MinResourcePercentage(tree_ix_2_name)

    for tree_ix in CHOICE_DISALLOWED_CONV_IXS:
        disallowed_dict[conv_tree_ixs[tree_ix]] = 1.1
    choice_disallowed.set_by_tree_ix_dict(disallowed_dict)

    
    









    FLOPS_min_res_percents = MinResourcePercentage(tree_ix_2_name)
    FLOPS_min_res_percents.set_by_name("Conv2d", 0.2)

    tree_ix_2_percentage_dict = {
        (0,) : 0.2    # This will obviously have no effect, since all convolutional layers are capped. It is simply to show an example.
    }
    FLOPS_min_res_percents.set_by_tree_ix_dict(tree_ix_2_percentage_dict)








    weights_min_res_percents = MinResourcePercentage(tree_ix_2_name)
    weights_min_res_percents.set_by_name("Conv2d", 0.2)

    if TEST_PRUNING:
        weights_min_res_percents.set_by_name("Conv2d", 0.999999)


    

    pruning_disallowments = {
        "general" : generally_disallowed.min_resource_percentage_dict,
        "choice" : choice_disallowed.min_resource_percentage_dict,
        "FLOPS" : FLOPS_min_res_percents.min_resource_percentage_dict,
        "weights" : weights_min_res_percents.min_resource_percentage_dict
    }

    model_wrapper.initialize_pruning(GET_IMPORTANCE_DICT_FN, segnet_input_slice_connection_fn, segnet_kernel_connection_fn, pruning_disallowments)



    # model_wrapper.training_wrapper.test_showcase()
    # input("Don't go past here.")







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








