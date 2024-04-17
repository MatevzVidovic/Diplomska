#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:22:32 2019

@author: aayush
"""

from models import model_dict
from torch.utils.data import DataLoader
from dataset import IrisDataset
import torch
from utils import mIoU, CrossEntropyLoss2d,total_metric,get_nparams,Logger,GeneralizedDiceLoss,SurfaceLoss
import numpy as np
from dataset import transform
from opt import parse_args
import os
from pathlib import Path
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from visdom import Visdom

import torch.nn as nn
from torchvision import transforms
#from sklearn.metrics import log_loss
#%%

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    print('learning rate: ' + str(lr))

    if lr < 0.0001:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


vis_epoch = 0
def lossandaccuracy(loader,model,factor):
    epoch_loss = []
    ious = []
    model.eval()
    global vis_epoch
    global args
    bottom_ten = []
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
            #print (len(batchdata))
            img,labels,index,spatialWeights,maxDist=batchdata
            data = img.to(device)
            #print(data.shape)
            target = labels.to(device).long()
            output = model(data)

            # loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output,target)
            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))

            loss=torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target)
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))

            #total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (1-factor)*loss_sl+factor*(loss_dice)+loss

            epoch_loss.append(loss.item())
            predict = get_predictions(output)
            bottom_ten.append((loss.item(), img, predict, labels))
            bottom_ten.sort()
            bottom_ten = bottom_ten[:10]
            iou = mIoU(predict,labels)  # if binary, only return IoU of positive class
            ious.append(iou)

    os.makedirs('test/val/{}/epoch_{}'.format(args.expname, vis_epoch), exist_ok=True)
    for i, (loss, img, predict, label) in enumerate(bottom_ten):
        for b in range(len(img)):
            pred_img = predict[b].cpu().numpy()
            pred_img[pred_img > 1] = 0
            inp = img[b].squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp,0,1)
            img_orig = np.array(img_orig)
            label_img = label[b].view(640, 400)
            label_img = np.array(label_img)

            combine = np.hstack([img_orig,pred_img,label_img])
            plt.imsave('test/val/{}/epoch_{}/{}_{}.jpg'.format(args.expname, vis_epoch, i, b), combine)
    vis_epoch += 1

    return np.average(epoch_loss),np.average(ious), np.sum(epoch_loss) #TODO

#%%
if __name__ == '__main__':

    args = parse_args()
    kwargs = vars(args)

    n_classes = 4 if 'sip' in args.dataset.lower() else 2

#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.useGPU)
    if args.useGPU == 'True' or args.useGPU == 'true' or args.useGPU:
        print('USE GPU')
        device=torch.device("cuda")
        torch.cuda.manual_seed(12)
    else:
        print('USE CPU')
        device=torch.device("cpu")
        torch.manual_seed(12)

    torch.backends.cudnn.deterministic=False

    model_name = args.model + (str(n_classes) if n_classes > 2 else '')
    print(model_name)
    if model_name not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)

    LOGDIR = Path('logs')/args.expname
    (LOGDIR/'models').mkdir(parents=True, exist_ok=True)
    logger = Logger(LOGDIR/'logs.log')
    short_logger = Logger(LOGDIR/'short.log')

    model = model_dict[model_name]
    # if torch.cuda.device_count() > 1 and 'densenet' not in type(model).__name__.lower():  # can't use parallelism because our models use module changing logic in forward methods
    #   logger.write(f"Using all {torch.cuda.device_count()} GPUs")
    #   model = nn.DataParallel(model)
    model = model.to(device)

    # print('model eval()')
    # model.eval()
    #
    # summary(model, input_size=(1, 640, 400)) #, batch_size=args.bs)  #  input_size=(channels, H, W)
    # #print(model)
    # exit(0)

    #torch.save(model.state_dict(), '{}/models/{}_{}.pkl'.format(LOGDIR,args.model,'_0'))
    model.train()
    nparams = get_nparams(model)

    # try:
    #     from torchsummary import summary
    #     summary(model,input_size=(1,640,400))
    #     print("Max params:", 1024*1024/4.0)
    #     logger.write_summary(str(model.parameters))
    # except:
    #     print ("Torch summary not found !!!")

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True, n_classes=n_classes)
    criterion_SL = SurfaceLoss()

    # visdom
    # RUN python -m visdom.server
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)
    win_loss = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
    )
    win_iou = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
    )

    Path2file = args.dataset
    print('path to file: ' + str(Path2file))
    train = IrisDataset(filepath = Path2file,split='train',
                             transform = transform, n_classes=n_classes, **kwargs)
    print('len: ' + str(train.__len__()))
    valid = IrisDataset(filepath = Path2file , split='validation',
                            transform = transform, n_classes=n_classes, **kwargs)

    trainloader = DataLoader(train, batch_size = args.bs,
                             shuffle=True, num_workers = args.workers, drop_last=True)

    validloader = DataLoader(valid, batch_size = args.bs,
                             shuffle= False, num_workers = args.workers, drop_last=True)

    test = IrisDataset(filepath = Path2file , split='test',
                            transform = transform, n_classes=n_classes, **kwargs)

    testloader = DataLoader(test, batch_size = args.bs,
                             shuffle=False, num_workers = args.workers)

    #print('**********************************************')
#    alpha = 1 - np.arange(1,args.epochs)/args.epoch
    ##The weighing function for the dice loss and surface loss
    alpha=np.zeros(((args.epochs)))
    end=np.min([125,args.epochs])
    alpha[0:end]=1 - np.arange(1,end+1)/end
    if args.epochs>125:
        alpha[125:]=1

    best_val_loss_so_far = float('inf')
    epochs_since_best_loss = 0
    for epoch in range(args.epochs):
        print("Emptying cache..")
        torch.cuda.empty_cache()

        #optimizer = adjust_learning_rate(optimizer, epoch, args)
        ious = [] # TODO TOLE SEM PRESTAVIL
        training_loss_sum = 0.0
        for i, batchdata in enumerate(trainloader):
            img,labels,index,spatialWeights,maxDist= batchdata

            data = img.to(device)
            #print('data shape: ' + str(data.shape))
            target = labels.to(device).long() #todo kaj ce vzames temu channel=1 z view?
            #print("unique target: " + str(torch.unique(target)))
            #print('target: ' + str(target.shape)) # 2, 1, 640, 400

            optimizer.zero_grad()
            output = model(data)
            #print("unique OUTPUT: " + str(torch.unique(output)))
            #print('output: ' + str(output.shape))  # 2, 1, 640, 400

            # loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
            CE_loss = criterion(output,target) #  CEL aims to maximize the output probability at a pixel location, it remains agnos-tic to the structure inherent to eye images.
            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))

            loss=torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target)
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device)))

            #total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
            loss = (1-alpha[epoch])*loss_sl+alpha[epoch]*(loss_dice)+loss

            training_loss_sum += loss.item()
            predictions = get_predictions(output)
            iou = mIoU(predictions,labels)
            ious.append(iou)

            if i%10 == 0:
                logger.write('Epoch: {} [{}/{}], Loss: {:.3f}'.format(epoch,i,len(trainloader),loss.item()))

            loss.backward()
            optimizer.step()

        if epoch > 0:
            # visualize total training loss
            try:
                viz.line(
                    X=np.array([epoch]),
                    Y=np.array([training_loss_sum]),
                    win=win_loss,
                    name='training loss',
                    update='append'
                )
            except Exception:
                pass

        s = 'Epoch: {}, Train mIoU: {}'.format(epoch,np.average(ious))
        logger.write(s)
        short_logger.write_silent(s)

        lossvalid , miou, validation_loss_sum = lossandaccuracy(validloader,model,alpha[epoch])
        totalperf = total_metric(nparams,miou)
        s = 'Epoch: {}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'.format(epoch,lossvalid, miou,nparams,totalperf)
        logger.write(s)
        short_logger.write_silent(s)

        # visualize validation loss
        try:
            viz.line(
                X=np.array([epoch]),
                Y=np.array([validation_loss_sum]),
                win=win_loss,
                name='validation loss',
                update='append'
            )
        except Exception:
            print("ne dela..")
            pass

        # visualize validation iou
        try:
            viz.line(
                X=np.array([epoch]),
                Y=np.array([miou]),
                win=win_iou,
                name='validation IoU',
                update='append'
            )
        except Exception:
            pass

        scheduler.step(lossvalid)

        # save only the last checkpoint
        torch.save(model.state_dict(), LOGDIR/'models'/f'{args.model}_final.pkl')

        # every epoch calculate test IoU
        ious = []
        with torch.no_grad():
            for i, batchdata in tqdm(enumerate(testloader),total=len(testloader)):
                img,label_tensor,index,x,maxDist= batchdata
                data = img.to(device)
                target = label_tensor.to(device).long()  # todo kaj ce vzames temu channel=1 z view?
                output = model(data)
                predict = get_predictions(output)

                iou = mIoU(predict, label_tensor)
                ious.append(iou)

                #VISUALIZE ONLY EVERY 10 EPOCHS
                if epoch % 10 == 0:
                    os.makedirs('test/mask/{}/'.format(args.expname), exist_ok=True)
                    os.makedirs('test/mask/{}/epoch_{}'.format(args.expname, epoch), exist_ok=True)
                    for j in range(len(index)):
                        pred_img = predict[j].cpu().numpy()#/3.0
                        pred_img[pred_img > 1] = 0
                        inp = img[j].squeeze() * 0.5 + 0.5
                        img_orig = np.clip(inp,0,1)
                        img_orig = np.array(img_orig)
                        label = label_tensor[j].view(640, 400)
                        label = np.array(label)

                        combine = np.hstack([img_orig,pred_img,label])
                        plt.imsave('test/mask/{}/epoch_{}/{}.jpg'.format(args.expname, epoch,index[j]),combine)

            miou = np.average(ious)

            # visualize test iou
            try:
                viz.line(
                    X=np.array([epoch]),
                    Y=np.array([miou]),
                    win=win_iou,
                    name='test IoU',
                    update='append'
                )
            except Exception:
                pass

            s = 'Epoch: {}, Test mIoU: {}'.format(epoch, miou)
            logger.write(s)
            short_logger.write_silent(s)

        # save best model
        if validation_loss_sum < best_val_loss_so_far:
            logger.write(f"Saving new best model to {args.model}_best.pkl")
            torch.save(model.state_dict(), LOGDIR/'models'/f'{args.model}_best.pkl')
            best_val_loss_so_far = validation_loss_sum
            epochs_since_best_loss = 0

        # at epoch 125 restart loss-based stuff, since loss changes dramatically
        if epoch == 125:
            for group in optimizer.param_groups:
                group['lr'] = args.lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
            best_val_loss_so_far = float('inf')
            epochs_since_best_loss = 0

        # stopping criteria (only after epoch 125)
        elif epoch > 125 and epochs_since_best_loss >= 10:
            logger.write(f"Validation loss hasn't improved for 10 epochs, stopping training")
            break

        epochs_since_best_loss += 1
