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









# optimizer pa moras urejati spodaj, ker se bojim, da bi bilo kaj narobe,
# ce ni inicializiran po tem, ko je model ze na deviceu

if model_num == 0:
  
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


