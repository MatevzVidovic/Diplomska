



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



import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

from timeit import default_timer as timer
import gc

from helper_img_and_fig_tools import show_image, save_plt_fig_quick_figs, save_img






def print_cuda_memory(do_total_mem=True, do_allocated_mem=True, do_reserved_mem=True, do_free_mem=True, do_mem_stats=True, do_gc_tensor_objects=True):
    
    try:
        # Get total memory
        if do_total_mem:
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # in GB
            print(f"Total memory: {total_memory:.2f} GB")
        
        if do_allocated_mem:
            # Get allocated memory
            allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Allocated memory: {allocated_memory:.2f} GB")
        
        if do_reserved_mem:
            # Get reserved memory
            reserved_memory = torch.cuda.memory_reserved(0) / 1024**3
            print(f"Reserved memory: {reserved_memory:.2f} GB")
        
        if do_free_mem:
            # Get free memory
            free_memory = total_memory - allocated_memory
            print(f"Free memory: {free_memory:.2f} GB")
        

        if do_mem_stats:
            # Get memory stats
            print(torch.cuda.memory_stats())
        
        if do_gc_tensor_objects:
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        size_mb = obj.element_size() * obj.nelement() / 1024**2
                        print(f"Tensor {obj.size()} {obj.device}, {size_mb:.2f} MB")
                except:
                    pass
    

    except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e




"""
get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU are adapted from:
from train_with_knowledge_distillation import get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU
"""


def get_conf_matrix(predictions, targets, num_classes=2):
    """
    predictions and targets can be matrixes or tensors.
    
    In both cases we only get a single confusion matrix
    - in the tensor case it is simply agreggated over all examples in the batch.
    """

    try:
        

        if isinstance(predictions, torch.Tensor):
            predictions_np = predictions.data.cpu().long().numpy()
        elif isinstance(predictions, np.ndarray):
            predictions_np = predictions.astype(np.int64)
        else:
            raise NotImplementedError(f"Type of predictions not supported: {type(predictions)}")

        if isinstance(targets, torch.Tensor):    
            targets_np = targets.cpu().long().numpy()
        elif isinstance(targets, np.ndarray):
            targets_np = targets.astype(np.int64)
        else:
            raise NotImplementedError(f"Type of targets not supported: {type(targets)}")
        
        # for batch of predictions
        # if len(np.unique(targets)) != 2:
        #    print(len(np.unique(targets)))
        
        
        try:
            assert (predictions.shape == targets.shape)
        except:
            print("predictions.shape: ", predictions.shape)
            print("targets.shape: ", targets.shape)
            raise AssertionError





        """
        c = get_conf_matrix(np.array([0,1,2,3,3]), np.array([0,2,2,3,0]))
        print(c)

        PREDICTIONS
        0, 1, 2, 3
        [[1 0 0 1]   0 |
        [0 0 0 0]   1 |
        [0 1 1 0]   2  TARGETS
        [0 0 0 1]]  3 |
        """


        # The mask is here mostly to make this a 1D array.
        mask = (targets_np >= 0) & (targets_np < num_classes)




        """
        Example for 4 classes:
        Possible target values are [0, 1, 2, 3].
        
        Possible label values are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
        
        Label values [0, 1, 2, 3] are those that are 0 in the target.
        0 is for those who are also 0 in the prediction, 1 for those which are 1 in the prediction, etc.
        Label values [4, 5, 6, 7] are those that are 1 in the target, etc.
        Then this gets reshaped into a confusion matrix.
        np.reshape fills the matrix row by row.
        

        So the leftmost column will be the background.
        The top row will be the background.

        The diagonal will be the correct predictions.

        
        """

        if num_classes > 8:
            raise NotImplementedError("This function is not intended for more than 8 classes. Because np.uint8. Its easy to make it more general.")

        # print(mask) # 2d/3d tensor of true/false
        label = num_classes * targets_np[mask].astype(np.uint8) + predictions_np[mask].astype(np.uint8)
        # show_image([(predictions_np[mask], "Predictions"), (targets_np[mask], "Targets"))
        # gt_image[mask] vzame samo tiste vrednosti, kjer je mask==True
        # print(mask.shape)  # batch_size, 128, 128
        # print(label.shape) # batch_size * 128 * 128 (with batch_size==1:   = 16384)
        # print(label)  # vector composed of 0, 1, 2, 3 (in the multilabel case)
        count = np.bincount(label, minlength=num_classes ** 2)  # number of repetitions of each unique value
        # print(count) # [14359   475    98  1452]
        # so [predBGisBG, predBGisFG, predFGisBG, predFGisFG]
        confusion_matrix = count.reshape(num_classes, num_classes)


        return confusion_matrix


    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e



if False:
    _ = True

    # # slow but surely correct

    # def get_conf_matrix(predictions, targets, num_classes=2):
    #     """
    #     predictions and targets can be matrixes or tensors.
        
    #     In both cases we only get a single confusion matrix
    #     - in the tensor case it is simply agreggated over all examples in the batch.
    #     """

    #     try:

    #         predictions_np = predictions.data.cpu().long().numpy().astype(np.uint8)
    #         targets_np = targets.cpu().long().numpy().astype(np.uint8)
            
            
    #         try:
    #             assert (predictions.shape == targets.shape)
    #         except:
    #             print("predictions.shape: ", predictions.shape)
    #             print("targets.shape: ", targets.shape)
    #             raise AssertionError


    #         confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.uint64)

    #         for tar in range(num_classes):
    #             tar_mask = targets_np == tar
    #             for pred in range(num_classes):
    #                 curr_pred = predictions_np[tar_mask] == pred
    #                 confusion_matrix[tar, pred] = curr_pred.sum()

    #         return confusion_matrix


    #     except Exception as e:
    #         py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
    #         raise e






    # def two_class_IoU_from_predictions(predictions, targets):
    #     """
    #     predictions and targets can be matrixes or tensors.
        
    #     In both cases we only get a single confusion matrix
    #     - in the tensor case it is simply agreggated over all examples in the batch.
    #     """

    #     try:

    #         predictions_np = predictions.data.cpu().long().numpy().astype(np.uint8)
    #         targets_np = targets.cpu().long().numpy().astype(np.uint8)
    #         # for batch of predictions
    #         # if len(np.unique(targets)) != 2:
    #         #    print(len(np.unique(targets)))
            
            
    #         try:
    #             assert (predictions.shape == targets.shape)
    #         except:
    #             print("predictions.shape: ", predictions.shape)
    #             print("targets.shape: ", targets.shape)
    #             raise AssertionError


    #         # Calculate intersection and union
    #         intersection = np.logical_and(predictions_np, targets_np).astype(np.uint8)
    #         union = np.logical_or(predictions_np, targets_np).astype(np.uint8)

    #         union_sum = union.sum()
    #         # Calculate IoU
    #         iou = intersection.sum() / union_sum if union_sum != 0 else 0

    #         if len(predictions_np.shape) > 2:
    #             for ix in range(union.shape[0]):
    #                 print("IoU for image", ix, ":", intersection[ix].sum() / union[ix].sum())
    #                 show_image([(predictions_np[ix], "Predictions"), (targets_np[ix], "Targets"), (intersection[ix], "Intersection"), (union[ix], "Union")])
    #                 input("Press Enter to continue...")

    #         return iou


    #     except Exception as e:
    #         py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
    #         raise e













def get_IoU_from_predictions(predictions, targets, num_classes=2):
    """
    Returns vector of IoU for each class.
    IoU[0] is the IoU for the background, for example.
    """

    try:

        confusion_matrix = get_conf_matrix(predictions, targets, num_classes)
        IoU, where_is_union_zero = conf_matrix_to_IoU(confusion_matrix, num_classes)

        return IoU, where_is_union_zero
    

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e

def conf_matrix_to_IoU(confusion_matrix, num_classes):
    """
    c = get_conf_matrix(np.array([0,1,2,3,3]), np.array([0,2,2,3,3]))
    print(c)
    [[1 0 0 0]
     [0 0 0 0]
     [0 1 1 0]
     [0 0 0 2]]
    miou = conf_matrix_to_mIoU(c)  # for each class: [1.  0.  0.5 1. ]
    print(miou) # 0.625
    """

    try:
        if confusion_matrix.shape != (num_classes, num_classes):
            print(confusion_matrix.shape)
            raise NotImplementedError()

        unions = (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
        
        where_is_union_zero = unions == 0
        unions[where_is_union_zero] = 1  # to make the division not fail

        IoU = np.diag(confusion_matrix) / unions

        IoU[where_is_union_zero] = np.nan  # if union is 0, then IoU is undefined
        
        # print("Conf matrix:", confusion_matrix)
        # print("IoU diag:", IoU)

        return IoU, where_is_union_zero

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e





def conf_matrix_to_F1_prec_rec(confusion_matrix):


    try:

        TN = confusion_matrix[0][0]
        FN = confusion_matrix[0][1]
        FP = confusion_matrix[1][0]
        TP = confusion_matrix[1][1]


        precision = TP / (TP + FP)    if TP + FP > 0 else 0
        recall = TP / (TP + FN)    if TP + FN > 0 else 0

        F1 = 2 * (precision * recall) / (precision + recall)    if precision + recall > 0 else 0

        return {"F1": F1, "precision": precision, "recall": recall}

    

    except Exception as e:
        py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
        raise e


if False:
    _ = True
    # # On mIoU: It is particularly useful for multi-class segmentation tasks.
    # # mIoU is calculated by averaging the Intersection over Union (IoU) for each class.
    # def get_mIoU_from_predictions(predictions, targets):
    #     confusion_matrix = get_conf_matrix(predictions, targets)
    #     mIoU = conf_matrix_to_mIoU(confusion_matrix)

    #     return mIoU


    # def conf_matrix_to_mIoU(confusion_matrix):
    #     """
    #     c = get_conf_matrix(np.array([0,1,2,3,3]), np.array([0,2,2,3,3]))
    #     print(c)
    #     [[1 0 0 0]
    #      [0 0 0 0]
    #      [0 1 1 0]
    #      [0 0 0 2]]
    #     miou = conf_matrix_to_mIoU(c)  # for each class: [1.  0.  0.5 1. ]
    #     print(miou) # 0.625
    #     """

    #     try:
    #         #print(confusion_matrix)
    #         n_classes = 2
    #         if confusion_matrix.shape != (n_classes, n_classes):
    #             print(confusion_matrix.shape)
    #             raise NotImplementedError()

    #         MIoU = np.diag(confusion_matrix) / (
    #                 np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
    #                 np.diag(confusion_matrix))

    #         if n_classes == 2:
    #             print("mIoU computed with only two classes. Background omitted.")
    #             return MIoU.item(1) # only IoU for sclera (not background)
    #         else:
    #             return np.mean(MIoU)

    #     except Exception as e:
    #         py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
    #         raise e












class TrainingWrapper:

    @py_log.autolog(passed_logger=MY_LOGGER)
    def __init__(self, model, dataloaders_dict, learning_parameters, device):
        
        try:

            self.device = device

            self.model = model.to(self.device)
            self.dataloaders_dict = dataloaders_dict


            # self.epochs = learning_parameters["epochs"]
    
            self.loss_fn = learning_parameters["loss_fn"]
            self.train_epoch_size_limit = learning_parameters["train_epoch_size_limit"]

            self.zero_out_non_sclera_on_predictions = learning_parameters["zero_out_non_sclera_on_predictions"]

            # self.gradient_clipping_norm = learning_parameters["gradient_clipping_norm"]
            # self.gradient_clip_value = learning_parameters["gradient_clip_value"]


        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e




    def initialize_optimizer(self, optimizer_class, learning_rate):
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)


    
    def train(self):
        
        try:
            train_times = []

            dataloader = self.dataloaders_dict["train"]
        
            size = int(min(self.train_epoch_size_limit, len(dataloader.dataset)))
            
            num_batches = len(dataloader)
            size_so_far = 0

            self.model.train()

            start = timer()

            # print_cuda_memory()
            agg_test_loss = 0

            for batch_ix, data_dict in enumerate(dataloader):
                X = data_dict["images"]
                y = data_dict["masks"]
                scleras = data_dict["scleras"]
                img_names = data_dict["img_names"]


                X, y = X.to(self.device), y.to(self.device)


                # Compute prediction error
                pred = self.model(X)

                if self.zero_out_non_sclera_on_predictions:
                    scleras = scleras.to(self.device)
                    scleras = torch.squeeze(scleras, dim=1)
                    where_sclera_is_zero = scleras == 0
                    pred[:, 0, :, :][where_sclera_is_zero] = 1_000_000
                    pred[:, 1, :, :][where_sclera_is_zero] = -1_000_000
                    # we want to make logits such that we are sure this is not a vein
                    # shapes:  pred: (batch_size, 2, 128, 128), scleras: (batch_size, 1, 128, 128)

                loss = self.loss_fn(pred, y)

                curr_test_loss = loss.item() # MCDL implicitly makes an average over the batch, because it does the calc on the whole tensor
                agg_test_loss += curr_test_loss



                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                size_so_far += len(X)

                if batch_ix % 1 == 0:
                    end = timer()
                    train_times.append(end - start)
                    start = timer()
                    print(f"per-ex loss: {curr_test_loss:>7f}  [{size_so_far:>5d}/{size:>5d}]")
                
                if size_so_far >= size:
                    break
                

                
                # print_cuda_memory()

                # del X, y, pred, loss
                # torch.cuda.empty_cache()

                # print_cuda_memory()
            test_loss = agg_test_loss / num_batches
            print(f"Train Error: Avg loss: {test_loss:>.8f}")

            return train_times

        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e

    def epoch_pass(self, dataloader_name="train"):

        try:

            dataloader = self.dataloaders_dict[dataloader_name]

            self.model.eval()
            with torch.no_grad():
                for batch_ix, data_dict in enumerate(dataloader):
                    X = data_dict["images"]
                    y = data_dict["masks"]
                    img_names = data_dict["img_names"]
                    X, y = X.to(self.device), y.to(self.device)
                    self.model(X)
        
        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e





    def validation(self):
        return self.test(dataloader_name="validation")

    def test(self, dataloader_name="test"):

        try:

            dataloader = self.dataloaders_dict[dataloader_name]

            num_batches = len(dataloader)

            self.model.eval()
            agg_test_loss = 0
            n_cl = 2
            aggregate_conf_matrix = np.zeros((n_cl, n_cl), dtype=np.long)
            with torch.no_grad():
                for batch_ix, data_dict in enumerate(dataloader):
                    X = data_dict["images"]
                    y = data_dict["masks"]
                    scleras = data_dict["scleras"]
                    img_names = data_dict["img_names"]
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)

                    if self.zero_out_non_sclera_on_predictions:
                        scleras = scleras.to(self.device)
                        scleras = torch.squeeze(scleras, dim=1)
                        where_sclera_is_zero = scleras == 0
                        pred[:, 0, :, :][where_sclera_is_zero] = 1_000_000
                        pred[:, 1, :, :][where_sclera_is_zero] = -1_000_000
                        # we want to make logits such that we are sure this is not a vein
                        # shapes:  pred: (batch_size, 2, 128, 128), scleras: (batch_size, 1, 128, 128)



                    # loss_fn computes the mean loss for the entire batch.
                    # We cold also get the loss for each image, but we don't need to.
                    # https://discuss.pytorch.org/t/loss-for-each-sample-in-batch/36200

                    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    # The fact the shape of pred and y are diferent seems to be correct regarding loss_fn.
                    curr_test_loss = self.loss_fn(pred, y).item() # MCDL implicitly makes an average over the batch, because it does the calc on the whole tensor
                    agg_test_loss += curr_test_loss





                    pred_binary = pred[:, 1] > pred[:, 0]


                    confusion_matrix = get_conf_matrix(pred_binary, y, num_classes=n_cl)
                    aggregate_conf_matrix = aggregate_conf_matrix + confusion_matrix





            test_loss = agg_test_loss / num_batches
            F1_prec_rec = conf_matrix_to_F1_prec_rec(aggregate_conf_matrix)
            F1 = F1_prec_rec["F1"]
            precision = F1_prec_rec["precision"]
            recall = F1_prec_rec["recall"]
            IoU, _ = conf_matrix_to_IoU(aggregate_conf_matrix, num_classes=n_cl)
            IoU = IoU[1]  # only IoU for sclera (not background)
            
            print(f"{dataloader_name} Error: \n Avg loss: {test_loss:>.8f} \n F1: {F1:>.6f} \n Precision: {precision:>.6f} \n Recall: {recall:>.6f}\n IoU: {IoU:>.6f}\n")
            

            # accuracies.append("{correct_perc:>0.1f}%".format(correct_perc=(100*correct)))
            # avg_losses.append("{test_loss:>8f}".format(test_loss=test_loss))

            # return (test_loss, approx_IoU, F1, IoU)
            return {"loss": test_loss, "F1": F1, "IoU": IoU, "precision": precision, "recall": recall}


        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e
    
















    # this is my own function for my own specific use case
    # It is not necessary if you are implementing your own TrainingWrapper

    def test_showcase(self, path_to_save_to, dataloader_name="test"):

        try:

            dataloader = self.dataloaders_dict[dataloader_name]

            self.model.eval()
            test_loss = 0
            quick_figs_counter = 0
            n_cl = 2
            aggregated_conf_matrix = np.zeros((n_cl, n_cl), dtype=np.long)
            with torch.no_grad():
                for batch_ix, data_dict in enumerate(dataloader):
                    X = data_dict["images"]
                    y = data_dict["masks"]
                    scleras = data_dict["scleras"]
                    img_names = data_dict["img_names"]
                    
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)

                    print(f"scleras.shape: {scleras.shape}")


                    if self.zero_out_non_sclera_on_predictions:
                        scleras = scleras.to(self.device)
                        scleras = torch.squeeze(scleras, dim=1)
                        where_sclera_is_zero = scleras == 0
                        pred[:, 0, :, :][where_sclera_is_zero] = 1_000_000
                        pred[:, 1, :, :][where_sclera_is_zero] = -1_000_000
                        # we want to make logits such that we are sure this is not a vein
                        # shapes:  pred: (batch_size, 2, 128, 128), scleras: (batch_size, 1, 128, 128)


                    # loss_fn computes the mean loss for the entire batch.
                    # We cold also get the loss for each image, but we don't need to.
                    # https://discuss.pytorch.org/t/loss-for-each-sample-in-batch/36200

                    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    # The fact the shape of pred and y are diferent seems to be correct regarding loss_fn.
                    test_loss += self.loss_fn(pred, y).item()



                    pred_binary = pred[:, 1] > pred[:, 0]


                    n_cl = 2
                    confusion_matrix = get_conf_matrix(pred_binary, y, num_classes=n_cl)
                    aggregated_conf_matrix += confusion_matrix

                    # X and y are tensors of a batch, so we have to go over them all
                    for i in range(X.shape[0]):

                        img_name = img_names[i]

                        pred_binary = pred[i][1] > pred[i][0]

                        
                        pred_binary_cpu_np = (pred_binary.cpu()).numpy()

                        pred_grayscale_mask = pred[i][1].cpu().numpy() - pred[i][0].cpu().numpy()
                        pred_grayscale_mask_min_max_normed = (pred_grayscale_mask - pred_grayscale_mask.min()) / (pred_grayscale_mask.max() - pred_grayscale_mask.min())

                        # matplotlib expects (height, width, channels), but pytorch has (channels, height, width)
                        image_tensor = X[i].cpu()
                        image_tensor = image_tensor[:3, :, :] # kep only img channels, not Bcos, coye, or sclera
                        image_np = image_tensor.permute(1, 2, 0).numpy()

                        # print("min(Ground truth):", y[i].min())
                        # print("max(Ground truth):", y[i].max())
                        # print("num of ones in Ground truth:", torch.sum(y[i] == 1).item())
                        # print("num of zeros in Ground truth:", torch.sum(y[i] == 0).item())
                        # print("num of all elements in Ground truth:", y[i].numel())

                        gt = y[i].cpu().numpy().astype(np.uint8)
                        
                        # fig, ax = plt.subplots(2, 2)
                        
                        # ax[0, 0].set_title('Original image')
                        # ax[0, 0].imshow(image_np)

                        # ax[0, 1].set_title('Ground truth')
                        # ax[0, 1].imshow(gt)

                        # ax[1, 0].set_title('Binarized predictions (pred[1] > pred[0] i.e. target prob > background prob)')
                        # ax[1, 0].imshow(pred_binary_cpu_np, cmap='gray')

                        # ax[1, 1].set_title('pred[1] - pred[0], min-max normed')
                        # ax[1, 1].imshow(pred_grayscale_mask_min_max_normed, cmap='gray')
                        # plt.show(block=False)
                        # print(f"gt.shape: {gt.shape}")
                        # print(f"scleras.shape: {scleras.shape}")
                        # print(f"scleras[i].shape: {scleras[i].shape}")
                        sclera = scleras[i].squeeze().cpu().numpy().astype(np.float32)
                        save_img(sclera, path_to_save_to, f"{img_name}_ts_sclera.png")

                        # save_plt_fig_quick_figs(fig, f"ts_{img_name}")
                        save_img(image_np, path_to_save_to, f"{img_name}_ts_img.png")

                        # save_img(gt, path_to_save_to, f"{img_name}_ts_gt.png")

                        # mask is int64, because torch likes it like that. Lets make it float, because the vals are only 0s and 1s, and so smart conversion in save_img()
                        # will make it 0s and 255s.
                        gt_float = gt.astype(np.float32)
                        save_img(gt_float, path_to_save_to, f"{img_name}_ts_gt_float.png")

                        
                        # Here we actually have bool, surprisingly. Again, lets just multiply by 255
                        pred_binary_cpu_np_255 = pred_binary_cpu_np.astype(np.uint8)
                        pred_binary_cpu_np_255 = pred_binary_cpu_np_255 * 255
                        save_img(pred_binary_cpu_np_255, path_to_save_to, f"{img_name}_ts_pred.png")


                        # Get an image where TP is green, FP is red, FN is yellow
                        # Initialize an RGB image with zeros (black)
                        height, width = gt.shape
                        pred_colormap = np.zeros((height, width, 3), dtype=np.uint8)
                        # Green where both gt and pred are 1
                        pred_colormap[(gt == 1) & (pred_binary_cpu_np_255 == 255)] = [0, 255, 0]
                        # Red where pred is 1 and gt is 0
                        pred_colormap[(gt == 0) & (pred_binary_cpu_np_255 == 255)] = [255, 0, 0]
                        # Yellow where pred is 0 and gt is 1
                        pred_colormap[(gt == 1) & (pred_binary_cpu_np_255 == 0)] = [255, 255, 0]

                        save_img(pred_colormap, path_to_save_to, f"{img_name}_ts_colormap.png")

                        sclera = (sclera * 255).astype(np.uint8)
                        sclera = cv2.cvtColor(sclera, cv2.COLOR_GRAY2BGR)
                        sclera_with_colormap = cv2.addWeighted(sclera, 0.5, pred_colormap, 0.5, 0)
                        save_img(sclera_with_colormap, path_to_save_to, f"{img_name}_ts_sclera_with_colormap.png")





                        # fp_and_fn = (gt == 1) & (pred_binary_cpu_np_255 == 0) | (gt == 0) & (pred_binary_cpu_np_255 == 255)
                        # fp_and_fn_on_image_np = image_np.copy()
                        # fp_and_fn_on_image_np[fp_and_fn] = pred_colormap[fp_and_fn]
                        # save_img(fp_and_fn_on_image_np, path_to_save_to, f"{img_name}_ts_fp_and_fn.png")


                        # image_np_blacked_out_in_tp_and_tn = image_np.copy()
                        # tp_and_tn = (gt == 1) & (pred_binary_cpu_np_255 == 255) | (gt == 0) & (pred_binary_cpu_np_255 == 0)
                        # image_np_blacked_out_in_tp_and_tn[tp_and_tn] = 0
                        # save_img(image_np_blacked_out_in_tp_and_tn, path_to_save_to, f"{img_name}_ts_blacked_out.png")



                        
                        # save_img(pred_grayscale_mask_min_max_normed, path_to_save_to, f"{img_name}_ts_pred_grayscale.png")
                        
                        plt.pause(1.0)

                        
                        # This way we will keep going after inp was 'all' once.
                        if quick_figs_counter == 0:
                            inp = input("Enter 'all' to go through all imgs without reasking for input. Enter anything else to stop. Press Enter to continue...")
                        else:
                            if inp != 'all':
                                inp = input("Enter 'all' to go through all imgs without reasking for input. Enter anything else to stop. Press Enter to continue...")
                        

                        if inp != "" and inp != 'all':
                            return

                        quick_figs_counter += 1


                aggregated_conf_matrix = aggregated_conf_matrix / aggregated_conf_matrix.sum()
                aggregated_conf_matrix_df = pd.DataFrame(aggregated_conf_matrix)
                aggregated_conf_matrix_df.to_csv(osp.join(path_to_save_to, "aggregated_conf_matrix.csv"), index=False)



        except Exception as e:
            py_log_always_on.log_stack(MY_LOGGER, attr_sets=["size", "math", "hist"])
            raise e




    
