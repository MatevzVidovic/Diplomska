

import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open("active_logging_config.txt", 'r') as f:
    yaml_path = f.read()

log_config_path = osp.join(yaml_path)
do_log = False
if osp.exists(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
        file_log_setting = config.get(osp.basename(__file__), False)
        if file_log_setting:
            do_log = True

print(f"{osp.basename(__file__)} do_log: {do_log}")
if do_log:
    import python_logger.log_helper as py_log
else:
    import python_logger.log_helper_off as py_log

MY_LOGGER = logging.getLogger("prototip") # or any string. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

python_logger_path = osp.join(osp.dirname(__file__), 'python_logger')
py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024, var_blacklist=["tree_ix_2_module", "mask_path"])
handlers = py_log_always_on.file_handler_setup(MY_LOGGER, python_logger_path)




import torch
from torch import nn
from torch.utils.data import DataLoader

import argparse

from min_resource_percentage import MinResourcePercentage
from model_wrapper import ModelWrapper

from training_support import *
from losses import MultiClassDiceLoss, WeightedLosses, JaccardLoss

import ast

import helper_yaml_handler as yh



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process arguments that can change between trainings.")
    


    # Working with more then 5 arguments is a headache. With each argument you add the headache increases quadratically.

    parser.add_argument("--sd", type=str, help='SAVE_DIR', required=True)

    # To easily get the num of trainings to some nice round number.
    parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations. After how many train iterations do we stop the program.')
    parser.add_argument('--mtti', type=int, default=1e9, help='max_total_train_iters. If the num of total train iters is this, we stop the program.')
    
    # To conduct pruning.
    parser.add_argument('-p', '--pruning_phase', action='store_true',
                        help='If present, enables pruning phase (automatic pruning)')
    parser.add_argument('--ifn', type=str, default="IPAD_eq", help='Importance function definer. Options: IPAD_eq, uniform, random.')
    

    # Set this to 0 when you want to simulate input with temp file, so you e.g. save the graph of the model, or results, or ...
    parser.add_argument('--ips', type=int, default=1e9,
                        help="""iter_possible_stop. After this num of iters we are prompted for keyboard input. 
                        Just use 0 if you want to simulate input with temp file. The default is 1e9 so that we never stop.""")
    
    # Great for writing tests and making them fast (the speed of your tests is how much hair remains on your head).
    parser.add_argument("--tras", type=int, default=-1, help="""Test run and size. If you pass an int, that will be the size of the dataset, and it will be in testrun. 
                        If -1 (default), then it is not a test run.""")
    parser.add_argument("--tp", action="store_true", help="""test pruning. 
                    This makes it so all the conv layers have 0.999999 as their limit for weights. 
                    MIND THAT input slice pruning also affects the weights - so this generally means each layer will get one kernel OR one input slice pruned.
                    The uniform pruning starts at CURR_PRUNING_IX - so if you want the other half of layers to have their kernels pruned, just change that to 1.""")


    # Main batch of parameters:
    parser.add_argument("--yaml", type=str, help="Path to YAML file with all the parameters.", required=True)

    # Overriding yaml files:
    parser.add_argument("--yo", type=str, nargs='*', help="""
                        yaml_overrides
                        You can pass multiple paths to yaml files. 
                        Each next yaml file will override/add the attributes it has to the dict of attributes.
                        This can add some flexibility, so not every set of params you need has to have it's own yaml file.
                        """)


    # Overriding the YAML parameters 
    # For 2 uses:
    # - tests (should be used only in tests)
    # - exploration - when trying to find the right parameter so you'd rather be setting it in the sbatch script
    # (so you can run multiple programs at once, and it's nice if you can have a bash argument that you just change and through the argument this value changes)
    # e.g. parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    # Testing ones:
    parser.add_argument("--ntibp", type=int, help="Number of training iterations between prunings.", default=None)
    parser.add_argument("--ptp", type=float, help="Proportion to prune.", default=None)
    parser.add_argument("--map", type=float, help="Max auto prunings.", default=None)






    args = parser.parse_args()
    print(f"Args: {args}")


    SAVE_DIR = args.sd
    max_train_iters = args.mti
    max_total_train_iters = args.mtti
    is_pruning_ph = args.pruning_phase
    IMPORTANCE_FN_DEFINER = args.ifn
    iter_possible_stop = args.ips

    TEST_RUN_AND_SIZE = args.tras
    IS_TEST_RUN = TEST_RUN_AND_SIZE != -1
    TEST_PRUNING = args.tp

    yaml_path = args.yaml

    yaml_overrides = args.yo


    yaml_dict = yh.read_yaml(yaml_path)

    if yaml_overrides is not None:
        for path in yaml_overrides:
            yo_dict = yh.read_yaml(path)
            for key, val in yo_dict.items():
                yaml_dict[key] = val


    print(f"YAML: {yaml_dict}")

    IS_PRUNING_READY = yaml_dict["is_pruning_ready"]
    PATH_TO_DATA = yaml_dict["path_to_data"]
    BATCH_SIZE = yaml_dict["batch_size"]
    LEARNING_RATE = yaml_dict["learning_rate"]
    NUM_OF_DATALOADER_WORKERS = yaml_dict["num_of_dataloader_workers"]
    TRAIN_EPOCH_SIZE_LIMIT = yaml_dict["train_epoch_size_limit"]

    num_ep_per_iter = yaml_dict["num_epochs_per_training_iteration"]
    cleanup_k = yaml_dict["cleanup_k"]
    optimizer = yaml_dict["optimizer_used"]
    ZERO_OUT_NON_SCLERA_ON_PREDICTIONS = yaml_dict["zero_out_non_sclera_on_predictions"]
    loss_fn_name = yaml_dict["loss_fn_name"]
    alphas = yaml_dict["alphas"]

    DATASET = yaml_dict["dataset_option"]
    zero_out_non_sclera = yaml_dict["zero_out_non_sclera"]
    add_sclera_to_img = yaml_dict["add_sclera_to_img"]
    add_bcosfire_to_img = yaml_dict["add_bcosfire_to_img"]
    add_coye_to_img = yaml_dict["add_coye_to_img"]

    MODEL = yaml_dict["model"]
    INPUT_WIDTH = yaml_dict["input_width"]
    INPUT_HEIGHT = yaml_dict["input_height"]
    INPUT_CHANNELS = yaml_dict["input_channels"]
    OUTPUT_CHANNELS = yaml_dict["output_channels"]

    have_patchification = yaml_dict["have_patchification"]
    patchification_params = yaml_dict["patchification_params"]


    NUM_TRAIN_ITERS_BETWEEN_PRUNINGS = yaml_dict["num_train_iters_between_prunings"]
    max_auto_prunings = yaml_dict["max_auto_prunings"]
    proportion_to_prune = yaml_dict["proportion_to_prune"]
    
    prune_by_original_percent = yaml_dict["prune_by_original_percent"]
    num_to_prune = yaml_dict["num_filters_to_prune"]
    prune_n_kernels_at_once = yaml_dict["prune_n_kernels_at_once"]
    resource_name = yaml_dict["resource_name_to_prune_by"]



    # Override yaml with args here if you want to.

    # For writing tests:
    if args.ntibp is not None:
        NUM_TRAIN_ITERS_BETWEEN_PRUNINGS = args.ntibp
    if args.ptp is not None:
        proportion_to_prune = args.ptp
    if args.map is not None:
        max_auto_prunings = args.map



    # Parameter changes to prevent wrongness.

    if IMPORTANCE_FN_DEFINER == "uniform" or IMPORTANCE_FN_DEFINER == "random":
        prune_n_kernels_at_once = 1
    















    # For pruning to work the functions need to be written to some specific model. We choose to make them after the model that proved to be successful in the training phase.
    # These are the specifications.
    # This is how we guard against wrong callings.

    sth_wrong = OUTPUT_CHANNELS != 2 or optimizer != "Adam" or alphas != []
    if sth_wrong:
        print(f"OUTPUT_CHANNELS: {OUTPUT_CHANNELS}, should be 2, optimizer: {optimizer}, should be Adam, alphas: {alphas}, should be [].")
        raise ValueError("Some of the parameters are hardcoded and can't be changed. Please check the script and set the parameters to the right values.")








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






if DATASET == "aug_tf":
    from dataset_aug_tf import IrisDataset, custom_collate_fn
elif DATASET == "aug_old":
    from dataset_aug_old import IrisDataset, custom_collate_fn
    
    

if loss_fn_name == "MCDL":
    loss_fn = MultiClassDiceLoss()
elif loss_fn_name == "JACCARD":
    loss_fn = JaccardLoss()
else:
    raise ValueError("Loss function not recognized.")

optimizer = torch.optim.Adam




learning_parameters = {
    "learning_rate" : LEARNING_RATE,
    "loss_fn" : loss_fn,
    "optimizer_class" : optimizer,
    "train_epoch_size_limit" : TRAIN_EPOCH_SIZE_LIMIT,
    "zero_out_non_sclera_on_predictions" : ZERO_OUT_NON_SCLERA_ON_PREDICTIONS,
    "have_patchification" : have_patchification,
    "patchification_params" : patchification_params
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
    "channels" : 2
}



from unet_original import UNet

# patchification changesthings only for the model
# The dataset is giving the same exact images as usual.
# It's just that we do special things in the TrainingWrapper,
# and consequently patches get fed into the model.

# in and out dims of the model are the same anyways, so i won't say in_x and out_x, but just dim_x.
dim_y = OUTPUT_DIMS["height"]
dim_x = OUTPUT_DIMS["width"]
if have_patchification:
    dim_y = patchification_params["patch_y"]
    dim_x = patchification_params["patch_x"]

if MODEL == "64_2_6":
    model_parameters = {
        # layer sizes
        "output_y" : dim_y,
        "output_x" : dim_x,
        "n_channels" : INPUT_DIMS["channels"],
        "n_classes" : OUTPUT_DIMS["channels"],
        "starting_kernels" : 64,
        "expansion" : 2,
        "depth" : 6,
        }
elif MODEL == "64_1_6":
    model_parameters = {
        # layer sizes
        "output_y" : dim_y,
        "output_x" : dim_x,
        "n_channels" : INPUT_DIMS["channels"],
        "n_classes" : OUTPUT_DIMS["channels"],
        "starting_kernels" : 64,
        "expansion" : 1,
        "depth" : 6,
        }


INPUT_EXAMPLE = torch.randn(1, INPUT_DIMS["channels"], dim_y, dim_x)







dataloading_args = {


    # DataLoader params
    # Could have separate "train_batch_size" and "eval_batch_size" (for val and test)
    #  since val and test use torch.no_grad() and therefore use less memory. 
    "batch_size" : BATCH_SIZE,
    "shuffle" : False, # TODO shuffle??
    "num_workers" : NUM_OF_DATALOADER_WORKERS,
}


dataset_args = {

    "testrun" : IS_TEST_RUN,
    "testrun_size" : TEST_RUN_AND_SIZE,
   

    "input_width" : INPUT_DIMS["width"],
    "input_height" : INPUT_DIMS["height"],
    "output_width" : OUTPUT_DIMS["width"],
    "output_height" : OUTPUT_DIMS["height"],
    
    # iris dataset params
    "path_to_sclera_data" : PATH_TO_DATA,
    # "transform" : transform,
    "n_classes" : OUTPUT_DIMS["channels"],

    "zero_out_non_sclera" : zero_out_non_sclera,
    "add_sclera_to_img" : add_sclera_to_img,
    "add_bcosfire_to_img" : add_bcosfire_to_img,
    "add_coye_to_img" : add_coye_to_img

}



def get_data_loaders(**dataloading_args):
    
    data_path = dataset_args["path_to_sclera_data"]
    # n_classes = 4 if 'sip' in args.dataset.lower() else 2

    print('path to file: ' + str(data_path))

    train_dataset = IrisDataset(filepath=data_path, split='train', **dataset_args)
    valid_dataset = IrisDataset(filepath=data_path, split='val', **dataset_args)
    test_dataset = IrisDataset(filepath=data_path, split='test', **dataset_args)

    trainloader = DataLoader(train_dataset, batch_size=dataloading_args["batch_size"], collate_fn=custom_collate_fn, shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    validloader = DataLoader(valid_dataset, batch_size=dataloading_args["batch_size"], collate_fn=custom_collate_fn, shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=False)
    testloader = DataLoader(test_dataset, batch_size=dataloading_args["batch_size"], collate_fn=custom_collate_fn, shuffle=False, num_workers=dataloading_args["num_workers"], drop_last=False)
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

















# Since using 2dConvTransopse (upconvolutions) instead of simple upsampling, a few things have changed:
    # The upconvolution is a channel buffer,
    # so when the previous layer in the Up path is pruned, you have to prune the in-channels of the upconvolution, but you then mustn't
    # prune the actual next conv layer.

    # But the thing is, upconvs have weights like: (in_channels, out_channels, kernel_height, kernel_width)
    # So it's easier to fit them in as if they were batchnorms and have them prning in the kernel pruning function, even though we are pruning the input slice.
    
    # Bad idea:
    # But it makes more sense to do this: also prune the out-channels of the up-convolution and prune the in-channels of the next conv layer (as is done already).
    # But, the up-convolution is taking 2k channels to k channels.
    # So it doesn't make sense to just always also prune the up-convolution. Also, which channel would you even prune?!?

    # So let's just do the input slice pruning, and lets just remove the input slice pruning of the next conv layer.

UPCONV_LLM_IXS = [48, 55, 62, 69, 76, 83]







# Go see model graph to help you construct these connection functions.
# model_wrapper.model_graph()


def unet_tree_ix_2_skip_connection_start(tree_ix, conv_tree_ixs):
    #    tree_ix -> skip_conn_starting_index

    # It could be done programatically, however:
    # Assuming the layers that have skip connections have only one source of them,
    # we could calculate how many inputs come from the previous layer.
    # That is then the starting ix of skip connections.

    # To make this function, go look in the drawn matplotlib graph.
    # On the upstream, just look at the convolution's weight dimensions.
    # They are: [output_channels (num of kernels), input_channels (depth of kernels), kernel_height, kernel_width]
    # (output_dimensions - input_dimensions) is the ix of the first skip connection



    # Oh, I see. This is easily programmable.
    # Just use "initial_conv_resource_calc.pkl" and use 
    # (output_dimensions - input_dimensions) where output_dimensions > input_dimensions.
    # And that's it haha.

    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)

        if conv_ix == 24:
            
            return 64
        elif conv_ix == 22:
            
            return 128
        elif conv_ix == 20:
            
            return 256
        elif conv_ix == 18:
            
            return 512

        elif conv_ix == 16:
            
            return 1024
        
        elif conv_ix == 14:
            return 2048


    else:
        
        return None
    





"""
THIS HERE IS THE START OF BUILDING A CONNECTION fn
based on the _get_next_conv_id_list_recursive()
It is very early stage.
"""



def unet_input_slice_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
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

    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)
        conn_destinations.append((conv_tree_ixs[conv_ix+1], kernel_ix))
    
    # These are the convolutions right befor Upconvolutions. So here we actually shouldn't do this pruning.
    # Instead we have to only prune the in-channels of the Upconvolution. But this will be done in the kernel pruning function.
    # (the reason we do it that way is that the upconvolutions have weights dims: (in_channels, out_channels, kernel_height, kernel_width)
    # And normal Conv2d have weights dims: (out_channels, in_channels, kernel_height, kernel_width)
    # So pruning the input slice of Upconvolution is the same mechanic as pruning the outchannels of a regular Conv2d.

    # So here we just prevent the wrong input slice connection pruning:
    if conv_ix in [13, 15, 17, 19, 21, 23]:
        conn_destinations = []





    # We made it so that for conv layers who receive as input the previous layer and a skip connection
    # the first inpute slices are of the previous layer. This makes the line above as elegant as it is.
    # We will, however, have to deal with more trouble with skip connections. 

    
    # (however, we included in a different way, because it is more elegant and makes more sense that way) 
    # For the more general option (e.g. to include pruning of some other affected layers)
    # we can instead work with "lowest_level_modules" indexes.
    # These are modules that appear the lowest in the tree, and are the ones that actually 
    # do the work. Data passes through them. They arent just composites of less complex modules.
    # They are the actual building blocks.

    # LLM_ix = None
    # if tree_ix in lowest_level_modules:
    #     LLM_ix = lowest_level_modules.index(tree_ix)




    # We already handled the regular connections for convolutional networks.
    # Now, here come skip connections.
    # For explanation, look at the graphic in the original U-net paper.
    
    # We have to know where the skip connections start.
    # What real index is the zeroth index of the skip connections for the goal layer?
    # In this way we can then use the tree_ix to get the base ix.

    # For this, we will for now create a second function where we hardcode this.
    # It could be done programatically, however:
    # Assuming the layers that have skip connections have only one source of them,
    # we could calculate how many inputs come from the previous layer.
    # That is then the starting ix of skip connections.

    # To do this, we look at the code where the skip connections of the model are defined:
    # def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits
    
    # We then look at the graphic of our network. We see that the inc block and first three down blocks create skip connections.
    # Therefore the last (second) convolution in those blocks will be senging the skip connection forward.
    # This is how we identify the particular convolutional modules (LLMs) that are involved in skip connections.
    

    # if conv_ix in [1, 3, 5, 7]:
    
    goal_conv_ix = None
    if conv_ix == 1:
        goal_conv_ix = 24
    elif conv_ix == 3:
        goal_conv_ix = 22
    elif conv_ix == 5:
        goal_conv_ix = 20
    elif conv_ix == 7:
        goal_conv_ix = 18
    elif conv_ix == 9:
        goal_conv_ix = 16
    elif conv_ix == 11:
        goal_conv_ix = 14
    
    # adding the kernel ix
    if goal_conv_ix is not None:
        goal_input_slice_ix = kernel_ix + unet_tree_ix_2_skip_connection_start(conv_tree_ixs[goal_conv_ix], conv_tree_ixs)
        conn_destinations.append((conv_tree_ixs[goal_conv_ix], goal_input_slice_ix))


    # OUTC MUSTN'T BE PRUNED ANYWAY!!!!!!!!, BECAUSE IT IS THE OUTPUT OF THE NETWORK
    # outc has no next convolution
    # if conv_ix == 26:
        # conn_destinations = []
    
    
    return conn_destinations









def unet_kernel_connection_fn(tree_ix, kernel_ix, conv_tree_ixs, lowest_level_modules):
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
    
    conn_destinations = []

    LLM_ix = None
    if tree_ix in lowest_level_modules:
        LLM_ix = lowest_level_modules.index(tree_ix)





    # All convolutions have batchnorms right after them and those need to be pruned.
    conv_ix = None
    if tree_ix in conv_tree_ixs:
        conv_ix = conv_tree_ixs.index(tree_ix)

        # OUTC MUSTN'T BE PRUNED ANYWAY!!!!!!!!, BECAUSE IT IS THE OUTPUT OF THE NETWORK
        # out.conv doesn't have a batchnorm after it.
        # if conv_ix < 26:
        conn_destinations.append((lowest_level_modules[LLM_ix+1], kernel_ix))







    # These are the convolutions right befor Upconvolutions. So here we actually shouldn't do this pruning.
    # Instead we have to only prune the in-channels of the Upconvolution. But this will be done in the kernel pruning function.
    # (the reason we do it that way is that the upconvolutions have weights dims: (in_channels, out_channels, kernel_height, kernel_width)
    # And normal Conv2d have weights dims: (out_channels, in_channels, kernel_height, kernel_width)
    # So pruning the input slice of Upconvolution is the same mechanic as pruning the outchannels of a regular Conv2d.


    # The upconvolutions have LLM idxs , 48, 55, 62, 69, 76, 83
    # and corresponding in the order as the convs are listed below.
    if conv_ix in [13, 15, 17, 19, 21, 23]:
        ordered_ix = [13, 15, 17, 19, 21, 23].index(conv_ix)
        conn_destinations.append((lowest_level_modules[UPCONV_LLM_IXS[ordered_ix]], kernel_ix))
    



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








if IMPORTANCE_FN_DEFINER == "random":
    IMPORTANCE_FN = random_pruning_importance_fn
elif IMPORTANCE_FN_DEFINER == "uniform":
    IMPORTANCE_FN = uniform_random_pruning_importance_fn
elif IMPORTANCE_FN_DEFINER == "IPAD_eq":
    IMPORTANCE_FN = IPAD_and_weights(0.5, 0.5, 0.5)
else:
    raise ValueError(f"IMPORTANCE_FN_DEFINER must be 'random', 'uniform' or 'IPAD_eq'. Was: {IMPORTANCE_FN_DEFINER}")






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

    # model_wrapper.epoch_pass(dataloader_name="train")
    # maybe doing this on val, because it is faster and it kind of makes more sense
    model_wrapper.epoch_pass(dataloader_name="validation")

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
if IMPORTANCE_FN_DEFINER == "uniform" or IMPORTANCE_FN_DEFINER == "random":
    GET_IMPORTANCE_DICT_FN = dummy_get_importance_dict






if __name__ == "__main__":

    
    model_wrapper = ModelWrapper(UNet, model_parameters, dataloader_dict, learning_parameters, INPUT_EXAMPLE, save_path, device)






    if IS_PRUNING_READY:

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
            model_wrapper.conv_tree_ixs[26] : 1.1
        }
        generally_disallowed.set_by_tree_ix_dict(disallowed_dict)




        # Choice disallowing:
        # (only disallowed to be chosen for pruning, but still allowed to be pruned as a consequence of another pruning (through the kernel_connection_fn)).
        choice_disallowed = MinResourcePercentage(tree_ix_2_name)
        
        # For segnet:
        # conv_tree_ixs = model_wrapper.conv_tree_ixs
        # CHOICE_DISALLOWED_CONV_IXS = [15, 18, 21, 23]
        # The reasoning for this choice comes from kernel_connection_fn:
        # Because this then means, that [15, 18, 21, 23] haveto be disallowed to be chosen for pruning.
        # Because the kernel nums must match.
        # """
        # # So when we prune the layer right before a pooling, we have to prune the layer right before the corresonding unpoolong.

        # # Pairs of conv ixs:
        # # 1 23
        # # 3 21
        # # 6 18
        # # 9 15
        # """
        
        # for tree_ix in CHOICE_DISALLOWED_CONV_IXS:
        #     disallowed_dict[conv_tree_ixs[tree_ix]] = 1.1
        # choice_disallowed.set_by_tree_ix_dict(disallowed_dict)

        
        






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







        model_wrapper.initialize_pruning(GET_IMPORTANCE_DICT_FN, unet_input_slice_connection_fn, unet_kernel_connection_fn, pruning_disallowments, UPCONV_LLM_IXS)










    @py_log.autolog(passed_logger=MY_LOGGER)
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





    
    train_automatically(model_wrapper, main_save_path, val_stop_fn=validation_stop, max_training_iters=max_train_iters, max_total_training_iters=max_total_train_iters, 
                        max_auto_prunings=max_auto_prunings, train_iter_possible_stop=iter_possible_stop, pruning_phase=is_pruning_ph, cleanup_k=cleanup_k,
                         num_of_epochs_per_training=num_ep_per_iter, pruning_kwargs_dict=pruning_kwargs)








