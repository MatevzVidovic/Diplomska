
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
from losses import MultiClassDiceLoss, WeightedLosses

import ast


import helper_yaml_handler as yh




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process arguments that can change between trainings.")
    


    # Working with more then 5 arguments is a headache. With each argument you add the headache increases quadratically.


    parser.add_argument("--ptd", type=str, help='PATH_TO_DATA', required=True)
    parser.add_argument("--sd", type=str, help='SAVE_DIR', required=True)

    # To easily get the num of trainings to some nice round number.
    parser.add_argument('--mti', type=int, default=1e9, help='Max train iterations. After how many train iterations do we stop the program.')
    
    # To conduct pruning.
    parser.add_argument('-p', '--pruning_phase', action='store_true',
                        help='If present, enables pruning phase (automatic pruning)')
    parser.add_argument('--ifn', type=str, help='Importance function definer. Options: IPAD_eq, uniform, random.')
    

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




    # Overriding the YAML parameters (it is useful to activate this passing in through argument when you are trying to find the right parameter,
    # so you run multiple programs at once, and it's nice if you can have a bash argument that you just change and through the argument this value changes)
    # e.g. parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)







    args = parser.parse_args()


    PATH_TO_DATA = args.ptd
    SAVE_DIR = args.sd
    max_train_iters = args.mti
    is_pruning_ph = args.pruning_phase
    IMPORTANCE_FN_DEFINER = args.ifn
    iter_possible_stop = args.ips

    TEST_RUN_AND_SIZE = args.tras
    IS_TEST_RUN = TEST_RUN_AND_SIZE != -1
    TEST_PRUNING = args.tp

    yaml_path = args.yaml



    yaml_dict = yh.read_yaml(yaml_path)

    BATCH_SIZE = yaml_dict["batch_size"]
    LEARNING_RATE = yaml_dict["learning_rate"]
    NUM_OF_DATALOADER_WORKERS = yaml_dict["num_of_dataloader_workers"]
    TRAIN_EPOCH_SIZE_LIMIT = yaml_dict["train_epoch_size_limit"]

    num_ep_per_iter = yaml_dict["num_epochs_per_training_iteration"]
    cleaning_err_ix = yaml_dict["cleaning_error_ix"]
    cleanup_k = yaml_dict["cleanup_k"]
    DATASET = yaml_dict["dataset_option"]
    optimizer = yaml_dict["optimizer_used"]
    loss_fn_name = yaml_dict["loss_fn_name"]
    alphas = yaml_dict["alphas"]


    MODEL = yaml_dict["model"]
    INPUT_WIDTH = yaml_dict["input_width"]
    INPUT_HEIGHT = yaml_dict["input_height"]
    INPUT_CHANNELS = yaml_dict["input_channels"]
    OUTPUT_CHANNELS = yaml_dict["output_channels"]

    NUM_TRAIN_ITERS_BETWEEN_PRUNINGS = yaml_dict["num_train_iters_between_prunings"]
    max_auto_prunings = yaml_dict["max_auto_prunings"]
    proportion_to_prune = yaml_dict["proportion_to_prune"]
    
    prune_by_original_percent = yaml_dict["prune_by_original_percent"]
    num_to_prune = yaml_dict["num_filters_to_prune"]
    prune_n_kernels_at_once = yaml_dict["prune_n_kernels_at_once"]
    resource_name = yaml_dict["resource_name_to_prune_by"]



    # Override yaml with args here if you want to.



    # Parameter changes to prevent wrongness.

    if IMPORTANCE_FN_DEFINER == "uniform" or IMPORTANCE_FN_DEFINER == "random":
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

    






    
    train_automatically(model_wrapper, main_save_path, val_stop_fn=None, max_training_iters=max_train_iters, max_auto_prunings=max_auto_prunings, 
                        train_iter_possible_stop=iter_possible_stop, pruning_phase=is_pruning_ph, cleaning_err_ix=cleaning_err_ix, cleanup_k=cleanup_k,
                         num_of_epochs_per_training=num_ep_per_iter, pruning_kwargs_dict=pruning_kwargs)








