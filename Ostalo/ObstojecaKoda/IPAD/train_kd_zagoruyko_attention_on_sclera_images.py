#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

from PIL import Image
from torch import nn
import torch.nn.functional as F

from models import model_dict
from torch.utils.data import DataLoader
from dataset import IrisDataset
import torch
from utils import mIoU, CrossEntropyLoss2d, total_metric, get_nparams, Logger, GeneralizedDiceLoss, SurfaceLoss
import numpy as np
from dataset import transform
from opt import parse_args
import os
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary
from visdom import Visdom
from pprint import pprint
import argparse
from models import model_dict

import torchvision.transforms as T

# USE train-ATTENTION-TEST images
# python train_kd_zagoruyko_attention_on_sclera_images.py --bs 1 --workers 1


if __name__ == '__main__':
    #STUDENT_DICT_START_PATH = 'logs/student_without_distillation_lr0_001/models/dense_net_0.pkl'
    #STUDENT_DICT_END_PATH = 'logs/student_without_distillation_lr0_001/models/dense_net_200.pkl'

    STUDENT_DICT_START_PATH = 'logs/distillation_attention_beta0_01_lr0_001_all_layers__sum_definition/models/dense_net_0.pkl'
    STUDENT_DICT_END_PATH = 'logs/distillation_attention_beta0_01_lr0_001_all_layers__sum_definition/models/dense_net_200.pkl'


    TEACHER_END_PATH = "logs/ritnet_original_teacher_lr0_001/models/dense_net_200.pkl"

    # parse
    args = parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)

    kwargs = vars(args)

    #    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.useGPU)
    if args.useGPU == 'True' or args.useGPU == 'true':
        print('USE GPU')
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        torch.cuda.manual_seed(7)
    else:
        print('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(7)

    torch.backends.cudnn.deterministic = False

    # if args.model not in model_dict:
    #     print("Model not found !!!")
    #     print("valid models are:", list(model_dict.keys()))
    #     exit(1)


    # load teacher model
    teacher_state_dict = torch.load(TEACHER_END_PATH)
    # vedet moraš tudi katera arhitektura je bla shranjena
    teacher_model = model_dict['teacher']
    teacher_model = teacher_model.to(device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model_copy = copy.deepcopy(teacher_model)
    #print(teacher_model.parameters())
    #teacher_model.train()
    # cancel gradients for teacher model, because we don't want to update this network
    #for params in teacher_model.parameters():
    #    params.requires_grad = False


    # LOAD STUDENT MODEL - STARTING POINT
    student_state_dict = torch.load(STUDENT_DICT_START_PATH)
    # vedet moraš tudi katera arhitektura je bla shranjena
    student_model = model_dict['student']
    student_model = student_model.to(device)
    student_model.load_state_dict(student_state_dict)
    student_model_copy = copy.deepcopy(student_model)
    #student_model.train()
    #for params in student_model.parameters():
    #    params.requires_grad = False

    print('models loaded correctly...')

    Path2file = "eyes_attention"
    print('path to file: ' + str(Path2file))
    train = IrisDataset(filepath=Path2file, split='train',
                        transform=transform, **kwargs)
    print('len: ' + str(train.__len__()))

    trainloader = DataLoader(train, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True)

    print('datasets made... ' + str(len(trainloader)))

    teacher_model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            teacher_output = teacher_model(data)
            print('NUM OF ELEMENTS IN ATTENTION MAP:')
            print(teacher_model.att_map_x1_sum.shape)
            for i, g in enumerate([teacher_model.att_map_x1_sum, teacher_model.att_map_x2_sum, teacher_model.att_map_x3_sum,
                                   teacher_model.att_map_x4_sum, teacher_model.att_map_x5_sum, teacher_model.att_map_x6_sum,
                                   teacher_model.att_map_x7_sum, teacher_model.att_map_x8_sum, teacher_model.att_map_x9_sum]):
                #print(g.size()) # 1 x 640 x 400
                g_cpu = g[0].cpu().numpy()
                # set negative to zero
                #g_cpu[g_cpu < 0] = 0
                print(g_cpu.shape)
                print('min: ' + str(np.min(g_cpu)))
                print('max: ' + str(np.max(g_cpu)))
                plt.imshow(g_cpu, interpolation='bicubic')
                plt.title(f'g{i} - teacher')
                plt.show()
                #input()
    """
    student_model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            student_output = student_model(data)

            for i, g in enumerate([student_model.att_map_x1_sum, student_model.att_map_x3_sum, student_model.att_map_x5_sum, student_model.att_map_x6_sum, student_model.att_map_x7_sum]):
                g_cpu = g[0].cpu().numpy()
                plt.imshow(g_cpu, interpolation='bicubic')
                plt.title(f'g{i} - student (start)')
                plt.show()
                #input()



    # LOAD STUDENT MODEL - END POINT
    student_state_dict = torch.load(STUDENT_DICT_END_PATH)
    # vedet moraš tudi katera arhitektura je bla shranjena
    student_model = model_dict['student']
    student_model = student_model.to(device)
    student_model.load_state_dict(student_state_dict)
    student_model_copy = copy.deepcopy(student_model)
    #student_model.train()
    #for params in student_model.parameters():
    #    params.requires_grad = False

    student_model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            student_output = student_model(data)

            for i, g in enumerate([student_model.att_map_x1_sum, student_model.att_map_x3_sum, student_model.att_map_x5_sum, student_model.att_map_x6_sum, student_model.att_map_x7_sum]):
                g_cpu = g[0].cpu().numpy()
                plt.imshow(g_cpu, interpolation='bicubic')
                plt.title(f'g{i} - student (end)')
                plt.show()
                #input()
    """



