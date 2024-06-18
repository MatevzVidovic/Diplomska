
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from unet import UNet

from dataset import IrisDataset, transform

from ModelWrapper import ModelWrapper

from ConvResourceCalc import ConvResourceCalc








test_purposes = True

batch_size = 16
learning_rate = 1e-3








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





model_parameters = {
    # layer sizes
    "n_channels" : 1,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }

if __name__ == "__main__":


    model = UNet(**model_parameters)



    train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(**dataloading_args)

    dataloader_dict = {
        "train" : train_dataloader,
        "valid" : valid_dataloader,
        "test" : test_dataloader,
    }


    for X, y in test_dataloader:
        # print(X)
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


    
    # https://pytorch.org/docs/stable/optim.html
    # SGD - stochastic gradient descent
    # imajo tudi Adam, pa sparse adam, pa take.
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss() # nn.MSELoss()

    learning_dict = {
        "optimizer" : optimizer,
        "loss_fn" : loss_fn,
    }


    wrap_model = ModelWrapper(model, dataloader_dict, learning_dict, model_name="UNet")

    wrap_model.loop_train_test(epochs=1)

    wrap_model.save()

    # wrap_model.test_showcase()


    resource_calc = ConvResourceCalc(wrap_model)
    resource_calc.calculate_resources(torch.randn(1, 1, 128, 128))
    FLOPs = resource_calc.cur_flops
    resource_dict = resource_calc.module_resources_dict

    print(f"FLOPs: {FLOPs}")
    print(f"Resource dict: {resource_dict}")


