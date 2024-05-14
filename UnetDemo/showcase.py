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
import matplotlib.pyplot as plt

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



train_dataloader, valida_dataloader, test_dataloader = get_data_loaders(**dataloading_args)






for X, y in test_dataloader:
    # print(X)
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break








"""
The next threee functions are copies of:
from train_with_knowledge_distillation import get_mIoU_from_predictions, get_conf_matrix, conf_matrix_to_mIoU

with slight changes to make it work nicely for our scenario.
"""










def get_conf_matrix(predictions, targets):
    predictions_np = predictions.data.cpu().long().numpy()
    targets_np = targets.cpu().long().numpy()
    # for batch of predictions
    # if len(np.unique(targets)) != 2:
    #    print(len(np.unique(targets)))

    assert (predictions.shape == targets.shape)
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

    # print(mask) # 3d tensor true/false
    label = num_classes * targets_np[mask].astype('int') + predictions_np[
        mask]  # gt_image[mask] vzame samo tiste vrednosti, kjer je mask==True
    # print(mask.shape)  # batch_size, 513, 513
    # print(label.shape) # batch_size * 513 * 513 (= 1052676)
    # print(label)  # vektor sestavljen iz 0, 1, 2, 3
    count = np.bincount(label, minlength=num_classes ** 2)  # kolikokrat se ponovi vsaka unique vrednost
    # print(count) # [816353  16014 204772  15537]
    confusion_matrix = count.reshape(num_classes, num_classes)
    # [[738697 132480]
    #  [106588  74911]]

    return confusion_matrix






# On mIoU: It is particularly useful for multi-class segmentation tasks.
# mIoU is calculated by averaging the Intersection over Union (IoU) for each class.
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
        
        










    import matplotlib.pyplot as plt


    IoUs = []
    avg_losses = []
    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, IoU = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = model(X)

                    # X and y are tensors of a batch, so we have to go over them all
                    # NOPE. These 16 are all of the same picture.
                    for i in range(batch_size):

                        pred_binary = pred[0][1] > pred[0][0]
                        pred_binary_cpu_np = (pred_binary.cpu()).numpy()

                        plt.subplot(2, 2, 1)
                        plt.imshow(X[0][0].cpu().numpy())
                        plt.subplot(2, 2, 2)
                        plt.imshow(y[0].cpu().numpy())
                        plt.subplot(2, 2, 3)
                        plt.imshow(pred_binary_cpu_np, cmap='gray')
                        plt.show()

                        # print("X")
                        # print(X)
                        # print("y")
                        # print(y)


                        # print("pred")
                        # print(pred)
                        # print("y")
                        # print(y)
                        # print("pred.shape")
                        # print(pred.shape)
                        # print("y.shape")
                        # print(y.shape)

                        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                        # The fact the shape of pred and y are diferent seems to be correct regarding loss_fn.
                        test_loss += loss_fn(pred, y).item()
                        # The problem is with mIoU
                        IoU += get_IoU_from_predictions(pred_binary, y[0])

        test_loss /= num_batches # (num_batches * batch_size)
        IoU /= num_batches # (num_batches * batch_size)

        print(f"Test Error: \n IoU: {(IoU):>.6f}%, Avg loss: {test_loss:>.8f} \n")

        IoUs.append(IoU)
        avg_losses.append(test_loss)
        # accuracies.append("{correct_perc:>0.1f}%".format(correct_perc=(100*correct)))
        # avg_losses.append("{test_loss:>8f}".format(test_loss=test_loss))










    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        test(test_dataloader, model, loss_fn)
    print("Done!")






