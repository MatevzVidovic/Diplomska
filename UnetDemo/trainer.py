import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
from scipy import sparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset

from timeit import default_timer as timer

from unet import UNet

from dataset import IrisDataset, transform








flag_for_mIoU_computed_with_two_classes = False

# dataset.py   LineAugment doesn't work yet


# Only means it goes into a separate test subfolder,
# so it doesn't mess up the already built models.
test_purposes = False

model_num = 0

model_num_2_model_name = {
    0 : "unet",
}

model_name_2_model_class = {
    "unet" : UNet,
}

epochs = 1
batch_size = 16
learning_rate = 1e-3




loss_fn = nn.CrossEntropyLoss() # nn.MSELoss()

# optimizer pa moras urejati spodaj, ker se bojim, da bi bilo kaj narobe,
# ce ni inicializiran po tem, ko je model ze na deviceu

if model_num == 0:
  model_parameters = {
    # layer sizes
    "n_channels" : 1,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }
elif model_num == 1:
  model_parameters = {
    # layer sizes
    "chosen_num_of_features" : 5000,
    "second_layer_size" : 1000,
    "middle_layer_size" : 700,

    # general parameters
    "dropout" : 0.1,
    "leaky_relu_alpha" : 0.1,
    "learning_rate" : 1e-3,
  }



model = model_name_2_model_class[model_num_2_model_name[model_num]](**model_parameters)

main_folder = "built_models/"
if test_purposes:
    main_folder = "test_models/"

model_data_path = main_folder + model_num_2_model_name[model_num] + "_" + str(model_parameters) + "_" + "_model/"




# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# load_previous_model has been moved down, so that it works with while True



# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



model = model.to(device)
# print(model)


# https://pytorch.org/docs/stable/optim.html
# SGD - stochastic gradient descent
# imajo tudi Adam, pa sparse adam, pa take.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)









dataloading_args = {


    "testrun" : test_purposes,
   

    # Image resize setting - don't know what it should be.
    "width" : 128,
    "height" : 128,
    
    # iris dataset params
    "path_to_sclera_data" : "./sclera_data",
    "transform" : transform,
    "n_classes" : 2,

    # DataLoader params
    "batch_size" : batch_size,
    "shuffle" : False, # TODO shuffle??
    "num_workers" : 4,
}


def get_data_loaders(**dataloading_args):
    
    data_path = dataloading_args["path_to_sclera_data"]
    # n_classes = 4 if 'sip' in args.dataset.lower() else 2

    print('path to file: ' + str(data_path))

    train_dataset = IrisDataset(filepath=data_path, split='train', **dataloading_args)
    valid_dataset = IrisDataset(filepath=data_path, split='val', **dataloading_args)
    test_dataset = IrisDataset(filepath=data_path, split='test', **dataloading_args)

    trainloader = DataLoader(train_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=True)
    validloader = DataLoader(valid_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"], drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=dataloading_args["batch_size"], shuffle=True, num_workers=dataloading_args["num_workers"])
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    # I'm not sure why we're dropping last, but okay.

    print('train dataset len: ' + str(train_dataset.__len__()))
    print('val dataset len: ' + str(valid_dataset.__len__()))
    print('test dataset len: ' + str(test_dataset.__len__()))

    return trainloader, validloader, testloader



train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(**dataloading_args)






for X, y in test_dataloader:
    # print(X)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break








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
























while True:


    # True if directory already exists
    load_previous_model = os.path.isdir(model_data_path)
    # Set to False if you want to rewrite data
    load_previous_model = load_previous_model






    if load_previous_model:
        prev_model_details = pd.read_csv(model_data_path + "previous_model" + "_details.csv")
        prev_serial_num = prev_model_details["previous_serial_num"][0]
        prev_cumulative_epochs = prev_model_details["previous_cumulative_epochs"][0]
        model.load_state_dict(torch.load(model_data_path + model_num_2_model_name[model_num] + "_" + str(prev_serial_num) + ".pth"))
    else:
        prev_serial_num = 0
        prev_cumulative_epochs = 0
        
        









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











    avg_losses = []
    IoUs = []
    F1s = []
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, IoU, F1 = 0, 0, 0
        IoU_as_avg_on_matrixes = 0
        with torch.no_grad():
            for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)

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
                    test_loss += loss_fn(pred, y).item()








                    pred_binary = pred[:, 1] > pred[:, 0]

                    F1 += get_F1_from_predictions(pred_binary, y)
                    IoU += get_IoU_from_predictions(pred_binary, y)


                    # X and y are tensors of a batch, so we have to go over them all
                    for i in range(X.shape[0]):

                        pred_binary = pred[i][1] > pred[i][0]

                        # pred_binary_cpu_np = (pred_binary.cpu()).numpy()
                        # plt.subplot(2, 2, 1)
                        # plt.imshow(X[i][0].cpu().numpy())
                        # plt.subplot(2, 2, 2)
                        # plt.imshow(y[i].cpu().numpy())
                        # plt.subplot(2, 2, 3)
                        # plt.imshow(pred_binary_cpu_np, cmap='gray')
                        # plt.show()

                        curr_IoU = get_IoU_from_predictions(pred_binary, y[i])
                        # print(f"This image's IoU: {curr_IoU:>.6f}%")
                        IoU_as_avg_on_matrixes += curr_IoU




        test_loss /= num_batches # not (num_batches * batch_size), because we are already adding batch means
        IoU /= num_batches
        F1 /= num_batches
        IoU_as_avg_on_matrixes /= (num_batches * batch_size)

        print(f"Test Error: \n Avg loss: {test_loss:>.8f} \n IoU: {(IoU):>.6f} \n F1: {F1:>.6f} \n")
        # print(f"IoU_as_avg_on_matrixes: {IoU_as_avg_on_matrixes:>.6f}")

        avg_losses.append(test_loss)
        IoUs.append(IoU)
        F1s.append(F1)

        # accuracies.append("{correct_perc:>0.1f}%".format(correct_perc=(100*correct)))
        # avg_losses.append("{test_loss:>8f}".format(test_loss=test_loss))










    all_train_times = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        if len(train_times) > 0:
            del train_times[0]
        print(train_times)
        all_train_times.extend(train_times)
        train_times = []
        test(test_dataloader, model, loss_fn)
    print("Done!")





    try:
        os.mkdir(main_folder)
    except:
        pass

    try:
        os.mkdir(model_data_path)
    except:
        pass



    torch.save(model.state_dict(), model_data_path + model_num_2_model_name[model_num] + "_" + str(prev_serial_num+1) + ".pth")

    new_df = pd.DataFrame({"previous_serial_num": [prev_serial_num+1], "previous_cumulative_epochs": [prev_cumulative_epochs+epochs]})
    new_df.to_csv(model_data_path + "previous_model" + "_details.csv")

    """
    Razlog za spodnji nacin kode:
    File "/home/matevzvidovic/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 677, in _extract_index
        raise ValueError("All arrays must be of the same length")
    ValueError: All arrays must be of the same length
    """
    a = {"train_times": all_train_times, "avg_losses": avg_losses, "IoUs": IoUs, "F1s": F1s}
    new_df = pd.DataFrame.from_dict(a, orient='index')
    new_df = new_df.transpose()
    new_df.to_csv(model_data_path + "train_times_and_stats_" + str(prev_serial_num+1) + ".csv")

    print("Saved PyTorch Model State")


