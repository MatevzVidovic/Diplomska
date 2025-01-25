

import logging
import yaml
import os.path as osp
import python_logger.log_helper as py_log_always_on

with open(f"{osp.join('pylog_configs', 'active_logging_config.txt')}", 'r') as f:
    cfg_name = f.read()
    yaml_path = osp.join('pylog_configs', cfg_name)

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

import y_helpers.helper_yaml_handler as yh

c=5
py_log.log_manual(MY_LOGGER, "brbr", c, a="Starting the program.", enm=c)


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

    model_type = yaml_dict["model_type"]
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
        print(f"OUTPUT_CHANNELS: {OUTPUT_CHANNELS}, should be 2, optimizer: {optimizer}, should be Adam, loss_fn_name: {loss_fn_name}, alphas: {alphas}, should be [].")
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
    "zero_out_non_sclera_on_predictions" : ZERO_OUT_NON_SCLERA_ON_PREDICTIONS
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



if model_type == "res_att":
    from unet_res_att import UNet
elif model_type == "att":
    from unet_att import UNet
else:
    raise ValueError("Model type not recognized.")

if MODEL == "64_2_6":
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
elif MODEL == "64_1_6":
    model_parameters = {
        # layer sizes
        "output_y" : OUTPUT_DIMS["height"],
        "output_x" : OUTPUT_DIMS["width"],
        "n_channels" : INPUT_DIMS["channels"],
        "n_classes" : OUTPUT_DIMS["channels"],
        "starting_kernels" : 64,
        "expansion" : 1,
        "depth" : 6,
        }







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






INPUT_EXAMPLE = torch.randn(1, INPUT_DIMS["channels"], INPUT_DIMS["height"], INPUT_DIMS["width"])











if __name__ == "__main__":

    
    model_wrapper = ModelWrapper(UNet, model_parameters, dataloader_dict, learning_parameters, INPUT_EXAMPLE, save_path, device)






    # model_wrapper.training_wrapper.test_showcase()







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








