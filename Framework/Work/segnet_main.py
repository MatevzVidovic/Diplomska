
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

py_log_always_on.limitations_setup(max_file_size_bytes=100 * 1024 * 1024, var_blacklist=["tree_ix_2_module", "mask_path"])
handlers = py_log_always_on.file_handler_setup(MY_LOGGER)




import y_helpers.shared_debug as debug
debug.start_debug()






import torch
from torch.utils.data import DataLoader

import argparse



from y_framework.min_resource_percentage import MinResourcePercentage
from y_framework.model_wrapper import ModelWrapper

from y_framework.training_support import *
from y_helpers.losses import *


import y_helpers.yaml_handler as yh

from y_helpers.pruning_importance import *

from y_helpers.samplers import BalancedRandomSampler, LimitedSampler


from y_framework.params_dataclasses import *



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process arguments that can change between trainings.")
    


    # These are all behavioural arguments.
    # They don't define the model or what the fundemental things about the way it should work are - that is all done through yaml.
    # These args just define what behaviour it should exhibit at this specific time of calling.
    # This is also how we keep num of params we have to explicitly pass low.


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
    parser.add_argument("--map", type=int, help="Max auto prunings.", default=None)






    args = parser.parse_args()
    args_dict = vars(args)

    SAVE_DIR = args.sd
    MAX_TRAIN_ITERS = args.mti
    MAX_TOTAL_TRAIN_ITERS = args.mtti
    IS_PRUNING_PH = args.pruning_phase
    IMPORTANCE_FN_DEFINER = args.ifn
    ITER_POSSIBLE_STOP = args.ips

    TEST_RUN_AND_SIZE = args.tras
    IS_TEST_RUN = TEST_RUN_AND_SIZE != -1
    TEST_PRUNING = args.tp



    yaml_path = args.yaml
    yaml_overrides = args.yo

    print(f"{yaml_path=}")

    YD = yh.read_yaml(yaml_path)

    if yaml_overrides is not None:
        for path in yaml_overrides:
            yo_dict = yh.read_yaml(path)
            for key, val in yo_dict.items():
                YD[key] = val

    


    # OG_YD = YD.copy()
    # print(f"Initial overridden YAML: {yh.get_readable_dict_str(OG_YD)}")




    # Override yaml with args here if you want to.

    # For writing tests:
    if args.ntibp is not None:
        YD["num_train_iters_between_prunings"] = args.ntibp
    if args.ptp is not None:
        YD["proportion_to_prune"] = args.ptp
    if args.map is not None:
        YD["max_auto_prunings"] = args.map



    # Constructing model id
    MODEL = ""
    # Making legacy model specification work
    if "model" in YD:
        if YD["model"] == "small":
            YD["model"] = "4_1"
        MODEL = YD["model"]
        info = MODEL.split("_")
        STARTING_KERNELS = int(info[0])
        EXPANSION = float(info[1])
    else:
        MODEL = f"{YD['starting_kernels']}_{YD['expansion']}_{YD['depth']}"
        STARTING_KERNELS = YD["starting_kernels"]
        EXPANSION = YD["expansion"]
        DEPTH = YD["depth"]

    shared.GLOBAL_DICT["MODEL"] = MODEL


    # Parameter changes to prevent wrongness.

    if IMPORTANCE_FN_DEFINER == "uniform" or IMPORTANCE_FN_DEFINER == "random":
        YD["prune_n_kernels_at_once"] = 1

    if TEST_RUN_AND_SIZE != -1:
        YD["train_epoch_size"] = TEST_RUN_AND_SIZE
        YD["val_epoch_size"] = TEST_RUN_AND_SIZE
        YD["test_epoch_size"] = TEST_RUN_AND_SIZE
    







    PRUNING_KWARGS = {
        "prune_by_original_percent": YD["prune_by_original_percent"],
        "prune_n_kernels_at_once": YD["prune_n_kernels_at_once"],
        "num_of_prunes": YD["num_filters_to_prune"],
        "resource_name": YD["resource_name_to_prune_by"],
        "original_proportion_to_prune": YD["proportion_to_prune"]
    }



    print(f"\n\n\nIs pruning phase: {IS_PRUNING_PH}")
    print(f"\nArgs_dict:\n{yh.get_readable_dict_str(args_dict)}\n")
    print(f"\nFinal YD:\n{yh.get_readable_dict_str(YD)}\n")









    # We used to take everything from the yaml dict and put it into variables first.
    # This increased clarity, because you could see all the parameters at the top of the script.
    # But it also complicated adding new params and keeping track of them and such.
    
    # Instead, we can now just copy this script, put it into an LLM 
    # and ask it to give us a documentation of the YD parameters, even with possible values for them.
    # And then of course check that it didn't halucinate,
    # but still, quick and easy.

    # E.g.:    Give all the parameters a yaml file is supposed to have here (YD dict) and group them in meaningful groups based on appearing in the same parts of the code / for the same function.
    #          With the wither possible options for the parameter if those are defined, or just the type of the parameter if they are not.
    # Take this as an example output:

    """"
    Model Architecture Parameters

    model: "64_2_6"  # Options: "64_2_6", "64_1_6", "4_1_4", "4_2_4", "8_2_5", "8_2_4", "6_2_4", "8_1.5_5", "8_1.5_6"
    input_height: int  # Height of input images
    input_width: int   # Width of input images

    Training Parameters

    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    num_of_dataloader_workers: int
    train_epoch_size: int
    val_epoch_size: int
    test_epoch_size: int

    Loss Function Configuration

    loss_fn_name: str  # Options: "MCDL", "MCDLW", "Tversky", "JACCARD"
    loss_params:
    bg_adj: float  # For MCDLW
    fp_imp: float  # For Tversky
    fn_imp: float  # For Tversky
    equalize: bool # For Tversky

    Dataset Configuration

    dataset_type: str  # Options: "vasd", "simple"
    path_to_data: str
    zero_out_non_sclera: bool
    add_sclera_to_img: bool
    add_bcosfire_to_img: bool
    add_coye_to_img: bool
    aug_type: str
    target: str

    Pruning Configuration

    is_pruning_ready: bool
    is_resource_calc_ready: bool
    conv2d_prune_limit: float
    num_train_iters_between_prunings: int
    max_auto_prunings: int
    cleanup_k: int
    prune_by_original_percent: bool
    prune_n_kernels_at_once: int
    num_filters_to_prune: int
    proportion_to_prune: float

    Patchification Parameters

    have_patchification: bool
    patchification_params:
    patch_y: int
    patch_x: int
    num_of_patches_from_img: int
    prob_zero_patch_resample: float

    Output Processing

    zero_out_non_sclera_on_predictions: bool





    Here's an example of how this would look in a YAML file:

    # Model Architecture
    model: "64_2_6"
    input_height: 256
    input_width: 256

    # Training Parameters
    learning_rate: 0.001
    train_batch_size: 32
    eval_batch_size: 64
    num_of_dataloader_workers: 4
    train_epoch_size: 1000
    val_epoch_size: 200
    test_epoch_size: 200

    # Loss Function Configuration
    loss_fn_name: "MCDL"
    loss_params:
    bg_adj: 0.5
    fp_imp: 0.5
    fn_imp: 0.5
    equalize: true

    # Dataset Configuration
    dataset_type: "vasd"
    path_to_data: "/path/to/data"
    zero_out_non_sclera: false
    add_sclera_to_img: false
    add_bcosfire_to_img: false
    add_coye_to_img: false
    aug_type: "standard"
    target: "sclera"

    # Pruning Configuration
    is_pruning_ready: true
    is_resource_calc_ready: true
    conv2d_prune_limit: 0.25
    num_train_iters_between_prunings: 100
    max_auto_prunings: 50
    cleanup_k: 5
    prune_by_original_percent: true
    prune_n_kernels_at_once: 1
    num_filters_to_prune: 100
    proportion_to_prune: 0.75

    # Patchification Parameters
    have_patchification: false
    patchification_params:
    patch_y: 64
    patch_x: 64
    num_of_patches_from_img: 10
    prob_zero_patch_resample: 0.1

    # Output Processing
    zero_out_non_sclera_on_predictions: false
    """













from y_models.segnet import SegNet



# In our UNet implementation the dims can be whatever you want.
# You could even change them between training iterations - but it might be a bad idea because all the weights had been learnt at the scale of the previous dims.
INPUT_DIMS = {
    "width" : YD["input_width"],
    "height" : YD["input_height"],
    "channels" : YD["input_channels"]
}

# In our UNet the output width and height have to be the same as the input width and height. 
OUTPUT_DIMS = {
    "width" : INPUT_DIMS["width"],
    "height" : INPUT_DIMS["height"],
    "channels" : 2
}




# patchification changesthings only for the model
# The dataset is giving the same exact images as usual.
# It's just that we do special things in the TrainingWrapper,
# and consequently patches get fed into the model.

# in and out dims of the model are the same anyways, so i won't say in_x and out_x, but just dim_x.
dim_y = OUTPUT_DIMS["height"]
dim_x = OUTPUT_DIMS["width"]
# We have to set the input dims of the model to the patch dims.
if YD["have_patchification"]:
    dim_y = YD["patchification_params"]["patch_y"]
    dim_x = YD["patchification_params"]["patch_x"]



model_parameters = {
        # layer sizes
        "output_y" : OUTPUT_DIMS["height"],
        "output_x" : OUTPUT_DIMS["width"],
        "in_chn" : INPUT_DIMS["channels"],
        "out_chn" : OUTPUT_DIMS["channels"],
        "starting_kernels" : STARTING_KERNELS,
        "expansion" : EXPANSION,
}




print(f"{INPUT_DIMS['channels']=}")
print(f"{dim_x=}")
print(f"{dim_y=}")
INPUT_EXAMPLE = torch.randn(1, INPUT_DIMS["channels"], dim_y, dim_x)









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

    

if YD["loss_fn_name"] == "MCDL":
    loss_fn = MultiClassDiceLoss()
elif YD["loss_fn_name"] == "WFTL":
    weights_list = None
    if "weights_list" in YD["loss_params"]:
        weights_list = YD["loss_params"]["weights_list"]
    loss_fn = WeightedFocalTverskyLoss(fp_imp=YD["loss_params"]["fp_imp"], fn_imp=YD["loss_params"]["fn_imp"], gamma=YD["loss_params"]["gamma"], weights_list=weights_list)
elif YD["loss_fn_name"] == "MCDLW":
    loss_fn = MultiClassDiceLoss(background_adjustment=YD["loss_params"]["bg_adj"])
elif YD["loss_fn_name"] == "Tversky":
    loss_fn = TverskyLoss(fp_imp=YD["loss_params"]["fp_imp"], fn_imp=YD["loss_params"]["fn_imp"], equalize=YD["loss_params"]["equalize"])
elif YD["loss_fn_name"] == "JACCARD":
    loss_fn = JaccardLoss()
else:
    raise ValueError("Loss function not recognized.")

optimizer = torch.optim.Adam




model_wrapper_params = ModelWrapperParams(
    model_class = SegNet,
    input_example = INPUT_EXAMPLE,
    save_path = save_path,
    device = device,
    learning_rate = YD["learning_rate"],
    optimizer_class = optimizer,
    is_resource_calc_ready = YD["is_resource_calc_ready"]
)

training_wrapper_params = TrainingWrapperParams(
    device = device,
    target = YD["target"],
    loss_fn = loss_fn,
    zero_out_non_sclera_on_predictions = YD["zero_out_non_sclera_on_predictions"],
    have_patchification = YD["have_patchification"],
    patchification_params = YD["patchification_params"],
    metrics_aggregation_fn = YD["metrics_aggregation_fn"],
    num_classes = OUTPUT_DIMS["channels"],
)


# model_wrapper_params = {
#     "model_class" : UNet,
#     "input_example" : INPUT_EXAMPLE,
#     "save_path" : save_path,
#     "device" : device,
#     "learning_rate" : YD["learning_rate"],
#     "optimizer_class" : optimizer,
#     "is_resource_calc_ready": YD["is_resource_calc_ready"]
# }

# training_wrapper_params = {
#     "device" : device,
#     "target" : YD["target"],
#     "loss_fn" : loss_fn,
#     "zero_out_non_sclera_on_predictions" : YD["zero_out_non_sclera_on_predictions"],
#     "have_patchification" : YD["have_patchification"],
#     "patchification_params" : YD["patchification_params"]
# }

















dataloading_args = {

    "train_batch_size" : YD["train_batch_size"],
    "eval_batch_size" : YD["eval_batch_size"], # val and test use torch.no_grad() so they use less memory
    "shuffle" : False,
    "num_workers" : YD["num_of_dataloader_workers"],
}


dataset_args = {

    "testrun" : IS_TEST_RUN,
    "testrun_size" : TEST_RUN_AND_SIZE,
   

    "input_width" : INPUT_DIMS["width"],
    "input_height" : INPUT_DIMS["height"],
    "output_width" : OUTPUT_DIMS["width"],
    "output_height" : OUTPUT_DIMS["height"],
    
    # iris dataset params
    "path_to_sclera_data" : YD["path_to_data"],
    # "transform" : transform,
    "n_classes" : OUTPUT_DIMS["channels"],
    "aug_type" : YD["aug_type"],

    "zero_out_non_sclera" : YD["zero_out_non_sclera"],
    "add_sclera_to_img" : YD["add_sclera_to_img"],
    "add_bcosfire_to_img" : YD["add_bcosfire_to_img"],
    "add_coye_to_img" : YD["add_coye_to_img"]

}

train_dataset_args = dataset_args.copy()

if YD["have_patchification"]:

    train_dataset_args['patchify'] = True
    train_dataset_args['patch_shape'] = (YD["patchification_params"]['patch_y'], YD["patchification_params"]['patch_x'])
    train_dataset_args['num_of_patches_from_img'] = YD["patchification_params"]['num_of_patches_from_img']
    train_dataset_args['prob_zero_patch_resample'] = YD["patchification_params"]['prob_zero_patch_resample']




if YD["dataset_type"] == "vasd":
    from y_datasets.dataset_all import IrisDataset, custom_collate_fn
elif YD["dataset_type"] == "multi":
    from y_datasets.dataset_multihead import IrisDataset, custom_collate_fn
elif YD["dataset_type"] == "simple":
    from y_datasets.dataset_simple import IrisDataset, custom_collate_fn
else:
    raise ValueError("Dataset type not recognized.")


def get_data_loaders():
    
    data_path = YD["path_to_data"]
    # n_classes = 4 if 'sip' in args.dataset.lower() else 2

    print('path to file: ' + str(data_path))


    train_dataset = IrisDataset(filepath=data_path, split='train', **train_dataset_args)

    valid_dataset = IrisDataset(filepath=data_path, split='val', **dataset_args)
    test_dataset = IrisDataset(filepath=data_path, split='test', **dataset_args)

    train_sampler = BalancedRandomSampler(train_dataset, num_samples=YD["train_epoch_size"])
    val_sampler = LimitedSampler(valid_dataset, num_samples=YD["val_epoch_size"], shuffle=False)
    test_sampler = LimitedSampler(test_dataset, num_samples=YD["val_epoch_size"], shuffle=False)

    trainloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=dataloading_args["train_batch_size"], collate_fn=custom_collate_fn, num_workers=dataloading_args["num_workers"])
    validloader = DataLoader(valid_dataset, sampler=val_sampler, batch_size=dataloading_args["eval_batch_size"], collate_fn=custom_collate_fn, num_workers=dataloading_args["num_workers"])
    testloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=dataloading_args["eval_batch_size"], collate_fn=custom_collate_fn, num_workers=dataloading_args["num_workers"])
    # in test_dataloader shuffle is False, because we want to keep the order of the images in test_showcase so they are easier to compare 
    # (But now we pass img_names through the dataloader anyway, so it doesn't matter anymore)
    
    # Let's not drop last.
    # it makes no sense. i think this might have been done because the IPAD fn was done only on the last batch, and so that
    # batch needed to be big.


    print('train dataset len: ' + str(len(train_dataset)))
    print('val dataset len: ' + str(len(valid_dataset)))
    print('test dataset len: ' + str(len(test_dataset)))

    print('train dataloader num of batches: ' + str(len(trainloader)))
    print('val dataloader num of batches: ' + str(len(validloader)))
    print('test dataloader num of batches: ' + str(len(testloader)))

    
    return trainloader, validloader, testloader






train_dataloader, valid_dataloader, test_dataloader = get_data_loaders()

save_preds_DL = None
save_preds_path = osp.join(YD["path_to_data"], "save_preds")
if osp.exists(save_preds_path):
    from y_datasets.dataset_for_save_preds import SavePredsDataset, save_preds_collate_fn
    save_preds_dataset = SavePredsDataset(filepath=YD["path_to_data"], split='save_preds', **dataset_args)
    save_preds_sampler = LimitedSampler(save_preds_dataset, num_samples=int(1e9), shuffle=False)
    save_preds_DL = DataLoader(save_preds_dataset, sampler=save_preds_sampler, batch_size=dataloading_args["eval_batch_size"], collate_fn=save_preds_collate_fn, num_workers=dataloading_args["num_workers"])

dataloader_dict = {
    "train" : train_dataloader,
    "validation" : valid_dataloader,
    "test" : test_dataloader,
    "save_preds" : save_preds_DL
}









MODEL_GRAPH_BREAKUP_PARAM = 0.05 # When getting model graph through breaking, 
# if a module in the graph would take up less width as a proportion of total width than this value, 
# then the graph is broken up recursively into more graphs. 
# This makes svgs viewable.
# For plt min fontsize is 1, and fontsize is in points, which are absolute units of space.
# So we need to do this breakup if we want the text to surely fit into the boxes. 

# Another way to solve the fontsize problem is to simply make the graph extremely huge.
# This way with smallere annd smaller rects (in terms of proportion to the entire plot) the fontsize will keep being over 1.
# But there are RAM problems and img size problems and such if this number is too big.
ONE_BIG_SVG_WIDTH = 700

INPUT_SLICE_CONNECTION_FN = None
KERNEL_CONNECTION_FN = None



# SegNet doesn't need an elif tree for connection functions, because the depth is fixed, and there are no skip connections.
# So the functions are the same for all chosen model parameters.




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




INPUT_SLICE_CONNECTION_FN = segnet_input_slice_connection_fn
KERNEL_CONNECTION_FN = segnet_kernel_connection_fn








if IMPORTANCE_FN_DEFINER == "random":
    IMPORTANCE_FN = random_pruning_importance_fn
elif IMPORTANCE_FN_DEFINER == "uniform":
    IMPORTANCE_FN = uniform_random_pruning_importance_fn
elif IMPORTANCE_FN_DEFINER == "IPAD_eq":
    IMPORTANCE_FN = IPAD_and_weights_granular(0.25, 0.25, 0.25, 0.25)
elif IMPORTANCE_FN_DEFINER == "IPAD1_L1":
    IMPORTANCE_FN = IPAD_and_weights_granular(0.5, 0, 0.5, 0)
elif IMPORTANCE_FN_DEFINER == "IPAD2_L2":
    IMPORTANCE_FN = IPAD_and_weights_granular(0, 0.5, 0, 0.5)
elif IMPORTANCE_FN_DEFINER == "IPAD1":
    IMPORTANCE_FN = IPAD_and_weights_granular(1.0, 0, 0, 0)
elif IMPORTANCE_FN_DEFINER == "IPAD2":
    IMPORTANCE_FN = IPAD_and_weights_granular(0, 1.0, 0, 0)
elif IMPORTANCE_FN_DEFINER == "L1":
    IMPORTANCE_FN = IPAD_and_weights_granular(0, 0, 1.0, 0)
elif IMPORTANCE_FN_DEFINER == "L2":
    IMPORTANCE_FN = IPAD_and_weights_granular(0, 0, 0, 1.0)
elif IMPORTANCE_FN_DEFINER == "L2_0.1":
    IMPORTANCE_FN = IPAD_and_weights_granular(0, 0.9, 0, 0.1)
elif IMPORTANCE_FN_DEFINER == "L2_0.9":
    IMPORTANCE_FN = IPAD_and_weights_granular(0, 0.1, 0, 0.9)
elif IMPORTANCE_FN_DEFINER == "L1_0.1":
    IMPORTANCE_FN = IPAD_and_weights_granular(0.9, 0, 0.1, 0)
elif IMPORTANCE_FN_DEFINER == "L1_0.9":
    IMPORTANCE_FN = IPAD_and_weights_granular(0.1, 0, 0.9, 0)
else:
    raise ValueError(f"IMPORTANCE_FN_DEFINER must be diff. Was: {IMPORTANCE_FN_DEFINER}")





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

    
    model_wrapper = ModelWrapper(model_wrapper_params, model_parameters, dataloader_dict, training_wrapper_params)






    if YD["is_pruning_ready"]:

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
        FLOPS_min_res_percents.set_by_name("Conv2d", YD["FLOPS_conv2d_prune_limit"])

        # tree_ix_2_percentage_dict = {
        #     (0,) : 0.2    # This will obviously have no effect, since all convolutional layers are capped. It is simply to show an example.
        # }
        # FLOPS_min_res_percents.set_by_tree_ix_dict(tree_ix_2_percentage_dict)








        weights_min_res_percents = MinResourcePercentage(tree_ix_2_name)
        weights_min_res_percents.set_by_name("Conv2d", YD["weights_conv2d_prune_limit"])

        if TEST_PRUNING:
            weights_min_res_percents.set_by_name("Conv2d", 0.999999)
        



        # When the network has been pruned severely, we would like to take it to 0% so we see how the model performs close to it.
        # For this reason we introduce relative percentages.

        # The original FLOPS limit might be 0.2. But as the FLOPS of thw overall network become close to 0.2, we run out of layers to prune.
        # So we introduce relative limits. Here the limit is scalled by the percent of FLOPS that are left to prune.
        # So if the relative limit is 0.8, and the network is at 0.2 of the original FLOPS, the limit is 0.2*0.8 = 0.16.
        # 
        # So we need to always be taking the minimum of the original limit and the relative limit.
        # And the relative limit should be wuite high, so that it doesn't come into effect too soon. 


        relative_FLOPS_min_res_percents = MinResourcePercentage(tree_ix_2_name)
        relative_FLOPS_min_res_percents.set_by_name("Conv2d", YD["relative_FLOPS_conv2d_prune_limit"])

        relative_weights_min_res_percents = MinResourcePercentage(tree_ix_2_name)
        relative_weights_min_res_percents.set_by_name("Conv2d", YD["relative_weights_conv2d_prune_limit"])



        # When we prune the network extremely it might happen that we prune the last kernel in a layer.
        # If there were 64 kernels in the layer, and we pruned 63, that kernel is 0.0156 of the original layer.
        # If the relative limit has come to be 0.012, we would prune this kernel. And that would be a disaster because the network wouldn't work anymore.

        kernel_num_min = MinResourcePercentage(tree_ix_2_name)
        kernel_num_min.set_by_name("Conv2d", 1)


        

        pruning_disallowments = {
            "general" : generally_disallowed.min_resource_percentage_dict,
            "choice" : choice_disallowed.min_resource_percentage_dict,
            "FLOPS" : FLOPS_min_res_percents.min_resource_percentage_dict,
            "weights" : weights_min_res_percents.min_resource_percentage_dict,
            "relative_FLOPS" : relative_FLOPS_min_res_percents.min_resource_percentage_dict,
            "relative_weights" : relative_weights_min_res_percents.min_resource_percentage_dict,
            "kernel_num" : kernel_num_min.min_resource_percentage_dict
        }







        model_wrapper.initialize_pruning(GET_IMPORTANCE_DICT_FN, INPUT_SLICE_CONNECTION_FN, KERNEL_CONNECTION_FN, pruning_disallowments)










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

        if (curr_train_iter - last_pruning_train_iter) >= YD["num_train_iters_between_prunings"]:
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





    
    train_automatically(model_wrapper, main_save_path, val_stop_fn=validation_stop, max_training_iters=MAX_TRAIN_ITERS, max_total_training_iters=MAX_TOTAL_TRAIN_ITERS, 
                        max_auto_prunings=YD["max_auto_prunings"], train_iter_possible_stop=ITER_POSSIBLE_STOP, pruning_phase=IS_PRUNING_PH, cleanup_k=YD["cleanup_k"],
                         num_of_epochs_per_training=1, pruning_kwargs_dict=PRUNING_KWARGS, viscinity_save_params=YD["viscinity_save"], model_graph_breakup_param=MODEL_GRAPH_BREAKUP_PARAM, one_big_svg_width=ONE_BIG_SVG_WIDTH)









    

    









