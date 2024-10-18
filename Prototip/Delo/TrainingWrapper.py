import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import gc

import os

import torch

from timeit import default_timer as timer







# Logging preparation:

import logging
import sys
import os

# Assuming the submodule is located at 'python_logger'
submodule_path = os.path.join(os.path.dirname(__file__), 'python_logger')
sys.path.insert(0, submodule_path)

import python_logger.log_helper as py_log


MY_LOGGER = logging.getLogger("prototip") # or any string instead of __name__. Mind this: same string, same logger.
MY_LOGGER.setLevel(logging.DEBUG)

python_logger_path = os.path.join(os.path.dirname(__file__), 'python_logger')
handlers = py_log.file_handler_setup(MY_LOGGER, python_logger_path, add_stdout_stream=False)
# def file_handler_setup(logger, path_to_python_logger_folder, add_stdout_stream: bool = False)





def print_cuda_memory(do_total_mem=True, do_allocated_mem=True, do_reserved_mem=True, do_free_mem=True, do_mem_stats=True, do_gc_tensor_objects=True):
    
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
                    print(f"Tensor {obj.size()} {obj.device}")
            except:
                pass




"""
get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU are adapted from:
from train_with_knowledge_distillation import get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU
"""


def get_conf_matrix(predictions, targets):
    """
    predictions and targets can be matrixes or tensors.
    
    In both cases we only get a single confusion matrix
    - in the tensor case it is simply agreggated over all examples in the batch.
    """


    predictions_np = predictions.data.cpu().long().numpy()
    targets_np = targets.cpu().long().numpy()
    # for batch of predictions
    # if len(np.unique(targets)) != 2:
    #    print(len(np.unique(targets)))
    
    
    try:
        assert (predictions.shape == targets.shape)
    except:
        print("predictions.shape: ", predictions.shape)
        print("targets.shape: ", targets.shape)
        raise AssertionError





    num_classes = 2

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
    mask = (targets_np >= 0) & (targets_np < num_classes)


    # print(np.any(mask == False))
    # False
    """
    I'm not sure why the mask is needed - wouldn't all the values be in this range?
    And if they weren't, shouldn't we throw an error?
    """


    """
    Example for 4 classes:
    Possible target values are [0, 1, 2, 3].
    
    Possible label values are [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].
    
    Label values [0, 1, 2, 3] are those that are 0 in the target.
    0 is for those who are also 0 in the prediction, 1 for those which are 1 in the prediction, etc.
    Label values [4, 5, 6, 7] are those that are 1 in the target, etc.
    Then this gets reshaped into a confusion matrix.
    np.reshape fills the matrix row by row.
    I don't like this, because in the 2 class case it is intuitive that background is the true negative,
    and so this doesn't conform to the usual confusion matrix representation:
    (predicted values as columns and target values as rows)
    [[TP, FN],
     [FP, TN]]

    First, we tried transposing along the anti-diagonal (np.rot90(confusion_matrix, 2).T), but this is wrong.
    Just perform an antidiagonal transpose on a 3x3 matrix on paper by hand and keep track of the column and row labels.
    They switch places. It's all wrong.

    Instead, we shold flip the columns along their centre, and then flip the rows along their centre.
    Essentially reshuffling them along the centre. This way the labels stay correctly corresponding after each of the operations.
    
    This means performing flipud and fliplr.
    But this is actually the same as just doing
    np.rot90(confusion_matrix, 2)
    and not doing the transpose.
    """

    # print(mask) # 2d/3d tensor of true/false
    label = num_classes * targets_np[mask].astype('int') + predictions_np[
        mask]  # gt_image[mask] vzame samo tiste vrednosti, kjer je mask==True
    # print(mask.shape)  # batch_size, 128, 128
    # print(label.shape) # batch_size * 128 * 128 (with batch_size==1:   = 16384)
    # print(label)  # vector composed of 0, 1, 2, 3 (in the multilabel case)
    count = np.bincount(label, minlength=num_classes ** 2)  # number of repetitions of each unique value
    # print(count) # [14359   475    98  1452]
    confusion_matrix = count.reshape(num_classes, num_classes)
    confusion_matrix = np.rot90(confusion_matrix, 2)
    # print(confusion_matrix)
    # [[ 1452   475]
    #  [   98 14359]]

    return confusion_matrix





def get_IoU_from_predictions(predictions, targets):
    confusion_matrix = get_conf_matrix(predictions, targets)
    IoU = conf_matrix_to_IoU(confusion_matrix)

    return IoU

def conf_matrix_to_IoU(confusion_matrix):
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

    #print(confusion_matrix)
    n_classes = 2
    if confusion_matrix.shape != (n_classes, n_classes):
        print(confusion_matrix.shape)
        raise NotImplementedError()

    IoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))

    return IoU.item(1) # only IoU for sclera (not background)







def get_F1_from_predictions(predictions, targets):
    confusion_matrix = get_conf_matrix(predictions, targets)
    IoU = conf_matrix_to_F1(confusion_matrix)

    return IoU

def conf_matrix_to_F1(confusion_matrix):

    TP = confusion_matrix[0][0]
    FN = confusion_matrix[0][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1 = 2 * (precision * recall) / (precision + recall)

    return F1


# On mIoU: It is particularly useful for multi-class segmentation tasks.
# mIoU is calculated by averaging the Intersection over Union (IoU) for each class.
def get_mIoU_from_predictions(predictions, targets):
    confusion_matrix = get_conf_matrix(predictions, targets)
    mIoU = conf_matrix_to_mIoU(confusion_matrix)

    return mIoU


def conf_matrix_to_mIoU(confusion_matrix):
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

    #print(confusion_matrix)
    n_classes = 2
    if confusion_matrix.shape != (n_classes, n_classes):
        print(confusion_matrix.shape)
        raise NotImplementedError()

    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))

    if n_classes == 2:
        print("mIoU computed with only two classes. Background omitted.")
        return MIoU.item(1) # only IoU for sclera (not background)
    else:
        return np.mean(MIoU)














class TrainingWrapper:

    @py_log.log(passed_logger=MY_LOGGER)
    def __init__(self, model, dataloaders_dict, learning_parameters):

        # Get cpu, gpu or mps device for training.
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Device: {self.device}")



        self.model = model.to(self.device)
        self.dataloaders_dict = dataloaders_dict



        # # Tole je v modelu ze:
        # self.dropout = learning_parameters["dropout"]
        # self.learning_rate = learning_parameters["learning_rate"]
        
        # # Tole pomoje ze v dataloaderjih
        # self.epochs = learning_parameters["epochs"]
        # self.batch_size = learning_parameters["batch_size"]

        self.loss_fn = learning_parameters["loss_fn"]

        # self.gradient_clipping_norm = learning_parameters["gradient_clipping_norm"]
        # self.gradient_clip_value = learning_parameters["gradient_clip_value"]











        # Conceptually not needed here.
        # Either give me the model class definition and make me make it here / load it from a file.
        # Or give me the actual model and you deal with that stuff.
        # Otherwise the mechanism of how to name the saved model will be scattered across 2 files.


        # self.model_data_path = model_name + "_model/"

        # # True if directory already exists
        # load_previous_model = os.path.isdir(self.model_data_path)
        # # Set to False if you want to rewrite data
        # # load_previous_model = False


        # if load_previous_model:
        #     prev_model_details = pd.read_csv(self.model_data_path + "previous_model_details.csv")
        #     self.prev_serial_num = prev_model_details["previous_serial_num"][0]
        #     self.model.load_state_dict(torch.load(self.model_data_path + "model_" + str(self.prev_serial_num) + ".pth"))
        # else:
        #     self.prev_serial_num = 0



    def initialize_optimizer(self, optimizer_class, learning_rate):
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)

    """
    


train_times = []

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    start = timer()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            end = timer()
            train_times.append(end - start)
            start = timer()
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # del X
        # del y
        # torch.cuda.empty_cache()
        # """
    
    def train(self):

        train_times = []

        dataloader = self.dataloaders_dict["train"]
    
        size = len(dataloader.dataset)
        self.model.train()

        start = timer()

        # print_cuda_memory()

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)

            print("pred")
            print(pred)
            print("y")
            print(y)

            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 1 == 0:
                end = timer()
                train_times.append(end - start)
                start = timer()
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
            # print_cuda_memory()

            # del X, y, pred, loss
            # torch.cuda.empty_cache()

            # print_cuda_memory()

        return train_times


    def epoch_pass(self, dataloader_name="train"):

        dataloader = self.dataloaders_dict[dataloader_name]

        self.model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    self.model(X)







    


    def validation(self):
        return self.test(dataloader_name="validation")

    def test(self, dataloader_name="test"):

        dataloader = self.dataloaders_dict[dataloader_name]

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        self.model.eval()
        test_loss, IoU, F1 = 0, 0, 0
        IoU_as_avg_on_matrixes = 0
        with torch.no_grad():
            for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)


                    # loss_fn computes the mean loss for the entire batch.
                    # We cold also get the loss for each image, but we don't need to.
                    # https://discuss.pytorch.org/t/loss-for-each-sample-in-batch/36200

                    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    # The fact the shape of pred and y are diferent seems to be correct regarding loss_fn.
                    test_loss += self.loss_fn(pred, y).item()





                    pred_binary = pred[:, 1] > pred[:, 0]

                    F1 += get_F1_from_predictions(pred_binary, y)
                    IoU += get_IoU_from_predictions(pred_binary, y)


                    # X and y are tensors of a batch, so we have to go over them all
                    for i in range(X.shape[0]):

                        pred_binary = pred[i][1] > pred[i][0]


                        curr_IoU = get_IoU_from_predictions(pred_binary, y[i])
                        # print(f"This image's IoU: {curr_IoU:>.6f}%")
                        IoU_as_avg_on_matrixes += curr_IoU




        test_loss /= num_batches # not (num_batches * batch_size), because we are already adding batch means
        IoU /= num_batches
        F1 /= num_batches
        IoU_as_avg_on_matrixes /= size # should be same or even more accurate as (num_batches * batch_size)

        print(f"{dataloader_name} Error: \n Avg loss: {test_loss:>.8f} \n IoU: {(IoU):>.6f} \n F1: {F1:>.6f} \n")
        # print(f"IoU_as_avg_on_matrixes: {IoU_as_avg_on_matrixes:>.6f}")

        # avg_losses.append(test_loss)
        # IoUs.append(IoU)
        # F1s.append(F1)

        # accuracies.append("{correct_perc:>0.1f}%".format(correct_perc=(100*correct)))
        # avg_losses.append("{test_loss:>8f}".format(test_loss=test_loss))

        return (test_loss, IoU, F1, IoU_as_avg_on_matrixes)
    





    # This conceptually doesn't make sense to have here.
    # This wrapper is only meant to handle the dirty work of training and testing.

    # def save(self):
        
    #     try:
    #         os.mkdir(self.model_data_path)
    #     except:
    #         pass

    #     torch.save(self.model.state_dict(), self.model_data_path + "model_" + str(self.prev_serial_num+1) + ".pth")

    #     new_df = pd.DataFrame({"previous_serial_num": [self.prev_serial_num+1]})
    #     new_df.to_csv(self.model_data_path + "previous_model_details.csv")

    #     print("Saved PyTorch Model State")


















    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # These are only helping functions and are not a necessary part of the wrapper.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    def loop_train_test(self, epochs):
        avg_losses = []
        IoUs = []
        F1s = []

        all_train_times = []

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_times = self.train()
            if len(train_times) > 0:
                del train_times[0]
            print(train_times)
            all_train_times.extend(train_times)
            
            test_loss, IoU, F1, _ = self.test()
            avg_losses.append(test_loss)
            IoUs.append(IoU)
            F1s.append(F1)
            print("Loss: ", test_loss)
            print("IoU: ", IoU)
            print("F1: ", F1)




        try:
            os.mkdir(self.model_data_path)
        except:
            pass

        """
        Razlog za spodnji nacin kode:
        File "/home/matevzvidovic/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 677, in _extract_index
            raise ValueError("All arrays must be of the same length")
        ValueError: All arrays must be of the same length
        """
        a = {"train_times": all_train_times, "avg_losses": avg_losses, "IoUs": IoUs, "F1s": F1s}
        new_df = pd.DataFrame.from_dict(a, orient='index')
        new_df = new_df.transpose()
        new_df.to_csv(self.model_data_path + "train_times_and_test_scores_" + str(self.prev_serial_num+1) + ".csv")



        print("Done!")
    

    def test_showcase(self):


        dataloader = self.dataloaders_dict["test"]

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, IoU, F1 = 0, 0, 0
        IoU_as_avg_on_matrixes = 0
        with torch.no_grad():
            for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)

                    # print("pred")
                    # print(pred)
                    # print("y")
                    # print(y)
                    # print("pred.shape")
                    # print(pred.shape)
                    # print("y.shape")
                    # print(y.shape)





                    # loss_fn computes the mean loss for the entire batch.
                    # We cold also get the loss for each image, but we don't need to.
                    # https://discuss.pytorch.org/t/loss-for-each-sample-in-batch/36200

                    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                    # The fact the shape of pred and y are diferent seems to be correct regarding loss_fn.
                    test_loss += self.loss_fn(pred, y).item()





                    pred_binary = pred[:, 1] > pred[:, 0]

                    F1 += get_F1_from_predictions(pred_binary, y)
                    IoU += get_IoU_from_predictions(pred_binary, y)


                    # X and y are tensors of a batch, so we have to go over them all
                    for i in range(X.shape[0]):

                        pred_binary = pred[i][1] > pred[i][0]

    
                        pred_binary_cpu_np = (pred_binary.cpu()).numpy()

                        pred_grayscale_mask = pred[i][1].cpu().numpy() - pred[i][0].cpu().numpy()
                        pred_grayscale_mask_min_max_normed = (pred_grayscale_mask - pred_grayscale_mask.min()) / (pred_grayscale_mask.max() - pred_grayscale_mask.min())

                        plt.subplot(2, 2, 1)
                        plt.gca().set_title('Original image')
                        plt.imshow(X[i][0].cpu().numpy())
                        plt.subplot(2, 2, 2)
                        plt.gca().set_title('Ground truth')
                        plt.imshow(y[i].cpu().numpy())
                        plt.subplot(2, 2, 3)
                        plt.gca().set_title('Binary predictions (sclera > bg)')
                        plt.imshow(pred_binary_cpu_np, cmap='gray')
                        plt.subplot(2, 2, 4)
                        plt.gca().set_title('Sclera - bg, min-max normed')
                        plt.imshow(pred_grayscale_mask_min_max_normed, cmap='gray')
                        plt.show()


                        curr_IoU = get_IoU_from_predictions(pred_binary, y[i])
                        # print(f"This image's IoU: {curr_IoU:>.6f}%")
                        IoU_as_avg_on_matrixes += curr_IoU




        test_loss /= num_batches # not (num_batches * batch_size), because we are already adding batch means
        IoU /= num_batches
        F1 /= num_batches
        IoU_as_avg_on_matrixes /= size # should be same or even more accurate as (num_batches * batch_size)

        print(f"Test Error: \n Avg loss: {test_loss:>.8f} \n IoU: {(IoU):>.6f} \n F1: {F1:>.6f} \n")
        # print(f"IoU_as_avg_on_matrixes: {IoU_as_avg_on_matrixes:>.6f}")

        # avg_losses.append(test_loss)
        # IoUs.append(IoU)
        # F1s.append(F1)

        # accuracies.append("{correct_perc:>0.1f}%".format(correct_perc=(100*correct)))
        # avg_losses.append("{test_loss:>8f}".format(test_loss=test_loss))

        return test_loss, IoU, F1, IoU_as_avg_on_matrixes
    



