import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from timeit import default_timer as timer








model_params = {
    "model_type" : types_dict[0],
    "chosen_num_of_features" : 5000,
    "second_layer_size" : 1000,
    "middle_layer_size" : 700,

    "dropout" : 0.1,
    "leaky_relu_alpha" : 0.1,

    "learning_rate" : 1e-3
}

model_parameters = {
    # layer sizes
    "n_channels" : 1,
    "n_classes" : 2,
    "bilinear" : True,
    "pretrained" : False,
  }











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



class RunModel:
    
        def __init__(self, data_matrix, given_y) -> None:

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




            
            print("data_matrix.shape")
            print(data_matrix.shape)


            model_parameters = model_params
            model_parameters["model_type"] = types_dict[3]
            model_parameters["chosen_num_of_features"] = data_matrix.shape[1]
            model_parameters["second_layer_size"] = 5
            # model_parameters["middle_layer_size"] = 3

            model_parameters["dropout"] = 0.0 # nočem ga ker weird reasoning, da je sploh zraven
                                            # 0.1 # pomaga proti overfittingu
            model_parameters["learning_rate"] = 1e-4 # 1e-3
            
            # na novo dodane
            model_parameters["weight_decay"] = 1e-5
            
            # Za vse tfidfje:
             #1.7 najboljse 0.418 nekje je R2 pa nic ne overfitta, ker ko gre druga runda epochov sploh niso high zadeve
            #1.8 ne deluje vec #1.5 boljse # 0.9 boljse # pri 0.7 ne overfitta, R2 nekje 0.41, oziroma zelo pocasi pada test R2
            # pri 0.5 R2 pride do 0.43, ampak zacne overfittat
            #0.05 #0.01 # 0.1 je considered high. # glavno za overiftting
            model_parameters["gradient_clipping_norm"] = None #1.0 # 1.0 # to ima problem, da normalizira celoten gradient
            # Ce en gradient dominira, ker je eksplodiral, to unici ucinek vseh ostalih.
            model_parameters["gradient_clip_value"] = 1.0 # 20.0 # to deluje za vsak value in je boljse

            self.model_data_path = main_folder + str(model_parameters["model_type"]) + "_" + str(model_parameters["chosen_num_of_features"]) + "_" + str(model_parameters["second_layer_size"]) + "_" + str(model_parameters["middle_layer_size"]) + "_model/"


            print(model_parameters)


            # nn.MSELoss()
            # self.loss_fn = nn.CrossEntropyLoss() # good for classification., doesn't work with regression
            self.loss_fn = nn.MSELoss() # good for regression
            # self.loss_fn = nn.L1Loss() # good for regression


            main_folder = "built_models/"
            if test_purposes:
                main_folder = "test_models/"




            train_dataloader, valid_dataloader, test_dataloader = get_data_loaders(**dataloading_args)
















            chosen_num_of_features = model_parameters["chosen_num_of_features"]
            learning_rate = model_parameters["learning_rate"]
            weight_decay = model_parameters["weight_decay"]
            gradient_clipping_norm = model_parameters["gradient_clipping_norm"]
            gradient_clip_value = model_parameters["gradient_clip_value"]

            self.chosen_num_of_features = chosen_num_of_features
            self.gradient_clipping_norm = gradient_clipping_norm
            self.gradient_clip_value = gradient_clip_value




            self.batch_size = 64

            self.data = (data_matrix.astype(np.float32), given_y.astype(np.float32))





            # Get cpu, gpu or mps device for training.
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )



            self.model = NeuralNetwork(model_parameters).to(self.device)
            self.model.float() # makes sure the weights are float32
            print(self.model)


            # True if directory already exists
            load_previous_model = os.path.isdir(self.model_data_path)
            # Set to False if you want to rewrite data
            load_previous_model = load_previous_model





            if load_previous_model:
                prev_model_details = pd.read_csv(self.model_data_path + "previous_model_" + str(chosen_num_of_features) + "_details.csv")
                self.prev_serial_num = prev_model_details["previous_serial_num"][0]
                self.prev_cumulative_epochs = prev_model_details["previous_cumulative_epochs"][0]
                self.model.load_state_dict(torch.load(self.model_data_path + "model_" + str(chosen_num_of_features) + "_" + str(self.prev_serial_num) + ".pth"))
            else:
                self.prev_serial_num = 0
                self.prev_cumulative_epochs = 0
                    
                




            # https://pytorch.org/docs/stable/optim.html
            # SGD - stochastic gradient descent
            # imajo tudi Adam, pa sparse adam, pa take.
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)






        def predict(self):

            # X = list(self.data[0])
            # y = list(self.data[1])

            # X, _, y, _ = train_test_split(self.data[0], self.data[1], test_size=0.0, shuffle=False) #, random_state=42)

            # X = self.data[0].toarray()
            # y = self.data[1].toarray()

            # print(type(self.data[0]))
            X = self.data[0].toarray()
            # y = np.array(self.data[1])
            y = self.data[1]


            predict_data = CustomDataset(X, y)
            dataloader = DataLoader(predict_data, batch_size=self.batch_size)
            model = self.model
            
            model.eval()

            predictions = []
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    predictions.extend(pred.tolist())
                    # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            predictions = np.array(predictions)
            predictions = np.reshape(predictions, (-1, 1))

            if os.path.exists('run_model_predictions.txt'):
                os.remove('run_model_predictions.txt')

            np.savetxt('run_model_predictions.txt', predictions)

            return predictions










          
        def train(self, dataloader):
            train_times = []
        
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

            return train_times





            

        def test(self, dataloader):

            model = self.model
            loss_fn = self.loss_fn


            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            model.eval()
            test_loss = 0
            
            all_y = []
            all_y_pred = []


            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = model(X)
                    test_loss += loss_fn(pred, y).item()
                    # print("y")
                    # print(y)
                    # print("pred")
                    # print(pred)
                    # print("loss_fn(pred, y)")
                    # print(loss_fn(pred, y))
                    # input("Press Enter to continue...")
                    all_y.extend(y.tolist())
                    all_y_pred.extend(pred.tolist())

            all_y = np.array(all_y)
            all_y_pred = np.array(all_y_pred)
            
            test_loss /= num_batches
            R2 = 1 - (all_y_pred - all_y).var() / all_y.var()
            MAE = np.abs(all_y_pred - all_y).mean()

            temp_y = all_y.copy()**2
            R2_if_using_root = 1 - (all_y_pred**2 - temp_y).var() / temp_y.var()
            MAE_if_using_root = np.abs(all_y_pred**2 - temp_y).mean()

            temp_y = np.exp(all_y.copy()) - 1
            R2_if_using_log = 1 - ((np.exp(all_y_pred) - 1) - temp_y).var() / temp_y.var()
            MAE_if_using_log = np.abs((np.exp(all_y_pred) - 1) - temp_y).mean()

            print(f"Test Error: \n R2: {(R2):>0.3f}%, Avg loss: {test_loss:>8f} \n")
            print(f"R2_if_using_root: {(R2_if_using_root):>0.3f}%, R2_if_using_log: {R2_if_using_log:>0.3f} \n")
            print(f"MAE: {MAE:>0.3f}, MAE_if_using_root: {MAE_if_using_root:>0.3f}, MAE_if_using_log: {MAE_if_using_log:>0.3f} \n")



            temp_y = all_y.copy()**3
            R2_if_using_3_root = 1 - (all_y_pred**3 - temp_y).var() / temp_y.var()
            MAE_if_using_3_root = np.abs(all_y_pred**3 - temp_y).mean()
            print(f"R2_if_using_3_root: {(R2_if_using_3_root):>0.3f}%, MAE_if_using_3_root: {MAE_if_using_3_root:>0.3f} \n")

            temp_y = all_y.copy()**4
            R2_if_using_4_root = 1 - (all_y_pred**4 - temp_y).var() / temp_y.var()
            MAE_if_using_4_root = np.abs(all_y_pred**4 - temp_y).mean()
            print(f"R2_if_using_4_root: {(R2_if_using_4_root):>0.3f}%, MAE_if_using_4_root: {MAE_if_using_4_root:>0.3f} \n")


            temp_y = all_y.copy()**5
            R2_if_using_5_root = 1 - (all_y_pred**5 - temp_y).var() / temp_y.var()
            MAE_if_using_5_root = np.abs(all_y_pred**5 - temp_y).mean()
            print(f"R2_if_using_5_root: {(R2_if_using_5_root):>0.3f}%, MAE_if_using_5_root: {MAE_if_using_5_root:>0.3f} \n")


            temp_y = all_y.copy()**8
            R2_if_using_8_root = 1 - (all_y_pred**8 - temp_y).var() / temp_y.var()
            MAE_if_using_8_root = np.abs(all_y_pred**8 - temp_y).mean()
            print(f"R2_if_using_8_root: {(R2_if_using_8_root):>0.3f}%, MAE_if_using_8_root: {MAE_if_using_8_root:>0.3f} \n")

            return R2_if_using_root, test_loss
        

   



        def loop_trainings(self):
            while True:
                self.load_train_test_and_save()
                
                self.prev_serial_num += 1
                self.prev_cumulative_epochs += 1
                


        def load_train_test_and_save(self):


            


            X_train, X_test, y_train, y_test = train_test_split(self.data[0], self.data[1], test_size=0.2) #, random_state=42)

            X_train = X_train.toarray()
            X_test = X_test.toarray()

            train_data = CustomDataset(X_train, y_train)
            test_data = CustomDataset(X_test, y_test)


            train_dataloader = DataLoader(train_data, batch_size=self.batch_size)
            test_dataloader = DataLoader(test_data, batch_size=self.batch_size)

            for X, y in test_dataloader:
                print(f"Shape of X [N, C, H, W]: {X.shape}")
                print(f"Shape of y: {y.shape} {y.dtype}")
                break




            R2s = []
            avg_losses = []

            all_train_times = []

            for t in range(epochs):
                print(f"Epoch {t+1}\n-------------------------------")
                train_times = self.train(train_dataloader)
                del train_times[0]
                print(train_times)
                all_train_times.extend(train_times)
                
                R2, loss = self.test(test_dataloader)
                R2s.append(R2)
                avg_losses.append(loss)

                if len(R2s) > 10 and R2s[-1] < R2s[-2] and R2s[-2] < R2s[-3]:
                    print("Early Stopping!")
                    break
            print("Done!")





            try:
                os.mkdir(main_folder)
            except:
                pass

            try:
                os.mkdir(self.model_data_path)
            except:
                pass

            torch.save(self.model.state_dict(), self.model_data_path + "model_" + str(self.chosen_num_of_features) + "_" + str(self.prev_serial_num+1) + ".pth")

            new_df = pd.DataFrame({"previous_serial_num": [self.prev_serial_num+1], "previous_cumulative_epochs": [self.prev_cumulative_epochs+epochs]})
            new_df.to_csv(self.model_data_path + "previous_model_" + str(self.chosen_num_of_features) + "_details.csv")

            """
            Razlog za spodnji nacin kode:
            File "/home/matevzvidovic/.local/lib/python3.10/site-packages/pandas/core/internals/construction.py", line 677, in _extract_index
                raise ValueError("All arrays must be of the same length")
            ValueError: All arrays must be of the same length
            """
            a = {"train_times": all_train_times, "R2s": R2s, "avg_losses": avg_losses}
            new_df = pd.DataFrame.from_dict(a, orient='index')
            new_df = new_df.transpose()
            new_df.to_csv(self.model_data_path + "train_times_" + str(self.prev_serial_num+1) + ".csv")

            print("Saved PyTorch Model State")


            





