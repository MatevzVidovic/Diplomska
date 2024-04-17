#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:37:59 2019

@author: aaa
"""
import torch
from dataset import IrisDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataset import transform
import os
from opt import parse_args
from models import model_dict
from tqdm import tqdm
from utils import get_predictions, ResourceManager
from train_with_knowledge_distillation import conf_matrix_to_mIoU, initialize_globals, \
    get_conf_matrix
import re
from train_with_pruning import count_number_of_learnable_parameters
from train_with_pruning import initialize_globals as initialize_globals_train_with_pruning

def classify_to_group(filename):
    if 'i_' in filename:
        return 'indoor'
    elif 'n_' in filename:
        return 'normal'
    elif 'p_' in filename:
        return 'poor'
    else:
        return 'sbvpi'




def visualize_on_test_images(args, filename, student_predictions):
    os.makedirs('eyes_visualize/visualized_masks/', exist_ok=True)
    os.makedirs('eyes_visualize/visualized_masks/{}'.format(args.expname), exist_ok=True)
    for j in range(len(filename)):
        pred_img = student_predictions[j].cpu().numpy() #/3.0

        if len(np.unique(pred_img)) == 4:
            pred_img = pred_img / 3.0
        elif len(np.unique(pred_img)) == 3 and np.max(pred_img) == 2:
            pred_img = pred_img / 2.0

        if len(np.unique(pred_img)) == 4:
            clist = [(0., [0,0,0]), (1./3., [0,1,0]), (2./3., [1,0,0]), (3./3., [0,0,1])]
        elif len(np.unique(pred_img)) == 3: # exists an image that does not have iris recognized..
            clist = [(0., [0, 0, 0]), (1. / 2., [0, 1, 0]), (2. / 2., [1, 0, 0])]

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)


        plt.imsave('eyes_visualize/visualized_masks/{}/{}.jpg'.format(args.expname, filename[j]), pred_img, cmap=cmap)





def evaluate_on_test_set(args, epoch, testloader, student_model, teacher_model, device):
    student_model.eval()
    conf_matrix_whole = np.zeros((args.numberOfClasses, args.numberOfClasses))
    list_of_test_mIoUs = []  # for calculating distribution and standard deviation
    conf_matrix_sbvpi = np.zeros((args.numberOfClasses, args.numberOfClasses))
    sbvpi_count = 0
    list_of_test_sbvpi_mIoUs = []
    conf_matrix_mobius_poor = np.zeros((args.numberOfClasses, args.numberOfClasses))
    mobius_poor_count = 0
    list_of_test_poor_mIoUs = []
    conf_matrix_mobius_normal = np.zeros((args.numberOfClasses, args.numberOfClasses))
    mobius_normal_count = 0
    list_of_test_normal_mIoUs = []
    conf_matrix_mobius_indoor = np.zeros((args.numberOfClasses, args.numberOfClasses))
    mobius_indoor_count = 0
    list_of_test_indoor_mIoUs = []
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
            img, label_tensor, filename, x, y = batchdata
            data = img.to(device)
            output = student_model(data)
            target = label_tensor.to(device).long()
            predictions = get_predictions(output)

            visualize_on_test_images(args, filename, predictions)

            # iou_b = mIoU(predictions, labels)
            # ious_all.append(iou_b)
            conf_matrix_batch = get_conf_matrix(args, predictions, target)
            conf_matrix_whole += conf_matrix_batch
            for idx, (prediction, label) in enumerate(zip(predictions, target)):
                conf_matrix = get_conf_matrix(args, prediction, label)
                # get mIoU for each image in batch
                image_mIoU = conf_matrix_to_mIoU(args, conf_matrix, False) # vrne mIoU, mean od vseh stirih IoU-jev
                #print(image_mIoU)
                list_of_test_mIoUs.append(image_mIoU)

                group = classify_to_group(filename[idx])
                if group == 'indoor':
                    conf_matrix_mobius_indoor = conf_matrix_mobius_indoor + conf_matrix
                    mobius_indoor_count += 1
                    list_of_test_indoor_mIoUs.append(image_mIoU)
                elif group == 'poor':
                    conf_matrix_mobius_poor = conf_matrix_mobius_poor + conf_matrix
                    mobius_poor_count += 1
                    list_of_test_poor_mIoUs.append(image_mIoU)
                elif group == 'normal':
                    conf_matrix_mobius_normal = conf_matrix_mobius_normal + conf_matrix
                    mobius_normal_count += 1
                    list_of_test_normal_mIoUs.append(image_mIoU)
                elif group == 'sbvpi':
                    conf_matrix_sbvpi = conf_matrix_sbvpi + conf_matrix
                    sbvpi_count += 1
                    list_of_test_sbvpi_mIoUs.append(image_mIoU)
                else:
                    raise Exception("UNKNOWN GROUP")

        # check group conf matrices and whole matrix == prve štiri bi se mogle seštet v zadnjo.. [OK]
        # print(conf_matrix_mobius_poor)
        # print(conf_matrix_mobius_normal)
        # print(conf_matrix_mobius_indoor)
        # print(conf_matrix_sbvpi)
        # print('===========================')
        # print(conf_matrix_whole)


        # conf matrix to iou

        print('Statistics:')
        #ious_mobius_poor = conf_matrix_to_mIoU(args, conf_matrix_mobius_poor)
        #print('# images in poor group: ' + str(mobius_poor_count) + ', mIoU (for all 4 classes): ' + str(ious_mobius_poor))
        print('poor category (len {0}): mean: mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_poor_mIoUs), np.mean(list_of_test_poor_mIoUs), np.std(list_of_test_poor_mIoUs)))
        #ious_mobius_normal = conf_matrix_to_mIoU(args, conf_matrix_mobius_normal)
        #print('# images in normal group: ' + str(mobius_normal_count) + ', mIoU (for all 4 classes): ' + str(ious_mobius_normal))
        print('normal category (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_normal_mIoUs), np.mean(list_of_test_normal_mIoUs), np.std(list_of_test_normal_mIoUs)))
        #ious_mobius_indoor = conf_matrix_to_mIoU(args, conf_matrix_mobius_indoor)
        #print('# images in indoor group: ' + str(mobius_indoor_count) + ', mIoU (for all 4 classes): ' + str(ious_mobius_indoor))
        print('indoor category (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_indoor_mIoUs), np.mean(list_of_test_indoor_mIoUs), np.std(list_of_test_indoor_mIoUs)))
        if sbvpi_count != 0:
            #ious_sbvpi = conf_matrix_to_mIoU(args, conf_matrix_sbvpi)
            #print('# images in sbvpi group: ' + str(sbvpi_count) + ', mIoU (for all 4 classes): ' + str(ious_sbvpi))
            print('sbvpi category (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_sbvpi_mIoUs),
                                                                        np.mean(list_of_test_sbvpi_mIoUs),
                                                                        np.std(list_of_test_sbvpi_mIoUs)))
        print('Test mIoU: {:0.5f}'.format(conf_matrix_to_mIoU(args, conf_matrix_whole, False)))
        print('Test mIoU (len {0}) mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_mIoUs), np.mean(list_of_test_mIoUs), np.std(list_of_test_mIoUs)))
        #composed_list_test_mIoUs = list_of_test_poor_mIoUs + list_of_test_indoor_mIoUs + list_of_test_normal_mIoUs + list_of_test_sbvpi_mIoUs
        #print('compose whole from categories (len {0}), mean: {1}, std: {2}'.format(len(composed_list_test_mIoUs), np.mean(composed_list_test_mIoUs), np.std(composed_list_test_mIoUs)))

        #print('list of test mious: ')
        #print(list_of_test_mIoUs)
        # plot histogram of test mIoUs
        #axes = plt.gca()
        #axes.set_ylim([0, 18])
        #axes.set_xlim([0, 1])
        #plt.hist(list_of_test_mIoUs, bins=100)
        #plt.show()


def main():
    args = parse_args()
    opt = vars(args)
    kwargs = vars(args)

    #debug
    args.load = 'logs/train_prunned_model_prunned_without_distillation/models/dense_net_200.pt'
    #args.gpu = [1]

    initialize_globals(args)
    initialize_globals_train_with_pruning(args)

    if args.model not in model_dict:
        print("Model not found !!!")
        print("valid models are:", list(model_dict.keys()))
        exit(1)

    if args.useGPU:
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
    else:
        device = torch.device("cpu")

    if args.teacher != '':
        # load teacher model
        teacher_state_dict = torch.load(args.teacher)
        # vedet moraš tudi katera arhitektura je bla shranjena
        teacher_model = model_dict['teacher']
        teacher_model.load_state_dict(teacher_state_dict)
        teacher_model = teacher_model.to(device)
    else:
        teacher_model = None

    if args.load.endswith('.pkl'):
        student_model = model_dict[args.model]
        student_model = student_model.to(device)
        filename = args.load
        if not os.path.exists(filename):
            print("model path not found !!!")
            exit(1)

        student_model.load_state_dict(torch.load(filename), strict=True)
        student_model = student_model.to(device)

        # extract epoch information from filename
        x = re.search("logs/(.*)/models/dense_net_(\d*).pkl$", filename)

        if x is None:
            x = re.search("logs/(.*)/models/(.*).pkl$", filename)
    elif args.load.endswith('.pt'):
        filename = args.load
        if not os.path.exists(filename):
            print("model path not found !!!")
            exit(1)

        student_model = torch.load(filename)
        student_model = student_model.to(device)

        # extract epoch information from filename
        x = re.search("logs/(.*)/models/ckpt_(\d*).pt$", filename)

        if x is None:
            x = re.search("logs/(.*)/models/dense_net_(\d*).pt$", filename)
    else:
        raise Exception('suffix not supported!')

    from torchsummary import summary
    summary(student_model, input_size=(1, args.height, args.width))  # , batch_size=args.bs)  #  input_size=(channels, H, W)

    # show resources
    rm = ResourceManager(student_model)
    rm.calculate_resources(torch.zeros((1, 1, args.height, args.width), device=device))
    #learnable_parameters, all_parameters = count_number_of_learnable_parameters(student_model, device)
    print('flops: {0}'.format(rm.cur_flops))

    epoch = 'TEST_' + str(x.group(2))
    # expname is used for save folder, use load instead (always present in test)
    args.expname = 'TEST_' + str(x.group(1))
    print(args.expname)

    Path2file = 'eyes_visualize'
    test_set = IrisDataset(filepath=Path2file, \
                           split='test', transform=transform, **kwargs)

    testloader = DataLoader(test_set, batch_size=args.bs,
                            shuffle=False)

    evaluate_on_test_set(args, epoch, testloader, student_model, teacher_model, device)





if __name__ == '__main__':
    main()

    # TEACHER
    # python visualize_masks.py --load logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl

    # student brez destilacije (tudi referencni model)
    # python visualize_masks.py --load logs/student_without_distillation_lr0_001_RUN3/models/dense_net_200.pkl


    # nas postopek z destilacijo
    # python visualize_masks.py --load logs/train_prunned_model_with_attention_beta0_001_prunned_with_same_distillation_model/models/dense_net_200.pt

    # nas postopek brez destilacije
    # python visualize_masks.py --load logs/train_prunned_model_prunned_without_distillation/models/dense_net_200.pt

    # LeGR
    # python visualize_masks.py --load logs/legr_best_retrained_200epochs/matic_pruneAway74_generations10_retrain50epochs_lr0_001_rank_l2_weight.pt



    # RM5 (139k)
    # python visualize_masks.py --load logs/reference_139k/models/dense_net_200.pkl

    # RM6 (139k)
    # python visualize_masks.py --load logs/reference_139k_alternative/models/dense_net_200.pkl


