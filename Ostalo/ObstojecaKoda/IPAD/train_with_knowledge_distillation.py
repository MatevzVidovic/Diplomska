#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import IrisDataset, transform
import torch

from utils import CrossEntropyLoss2d, Logger, GeneralizedDiceLoss, SurfaceLoss
import numpy as np
# from dataset import transform
from opt import parse_args
import os
from pathlib import Path
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from visdom import Visdom
from models import model_dict


def get_mIoU_from_predictions(args, predictions, targets):
    confusion_matrix = get_conf_matrix(args, predictions, targets)
    mIoU = conf_matrix_to_mIoU(args, confusion_matrix)

    return mIoU


def get_conf_matrix(args, predictions, targets):
    predictions_np = predictions.data.cpu().long().numpy()
    targets_np = targets.cpu().long().numpy()
    # for batch of predictions
    # if len(np.unique(targets)) != 2:
    #    print(len(np.unique(targets)))
    assert (predictions.shape == targets.shape)
    num_classes = 4 if 'sip' in args.dataset.lower() else 2

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


def conf_matrix_to_mIoU(args, confusion_matrix, log_per_class_miou=True):
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
    n_classes = 4 if 'sip' in args.dataset.lower() else 2
    if confusion_matrix.shape != (n_classes, n_classes):
        print(confusion_matrix.shape)
        raise NotImplementedError()

    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))

    if log_per_class_miou:
        logger.write("per-class mIOU: " + str(MIoU))

    if n_classes == 2:
        return MIoU.item(1) # only IoU for sclera (not background)
    else:
        return np.mean(MIoU)




# https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/4
def cross_entropy_with_soft_targets(student_prediction, teacher_prediction, T, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         student_prediction: predictions for student
         teacher_prediction: predictions for teacher
         size_average: if false, sum is returned instead of mean

    Examples (changed function, this wont work enymore)::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    if len(student_prediction.shape) != 4 and len(teacher_prediction.shape) != 4:
        logger.write("student_prediction and teacher_prediction shape len should be 4!!!")

    softmax = nn.Softmax(dim=1)
    target = softmax(teacher_prediction/T)
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        print("Target: " + str(target.shape))
        print("Student: " + str(student_prediction.shape))
        print(str((logsoftmax(student_prediction/T)).shape))
        test = torch.sum(-target * logsoftmax(student_prediction/T), dim=1)
        return torch.mean(test)
    else:
        return torch.sum(torch.sum(-target * logsoftmax(student_prediction/T), dim=1))


def get_indexes_with_lower_student_iou(args, student_batch_outputs, teacher_batch_outputs, target, epoch, index):
    indexes_with_lower_student_iou = []
    if teacher_batch_outputs is None or args.alwaysPenalize.lower() == 'true':
        #if only_poor:
        #    print('only poor used')
        #    # TODO izberi samo poor kategorijo..
        #    for i, image_name in enumerate(index):
        #        group = classify_to_group(image_name)
        #        if group == 'poor':
        #            indexes_with_lower_student_iou.append(i)
        #else:
        indexes_with_lower_student_iou = [x for x in range(args.bs)]
    else:
        student_predictions = get_predictions(student_batch_outputs)
        teacher_predictions = get_predictions(teacher_batch_outputs)
        for i in range(args.bs):
            student_prediction = student_predictions[i].unsqueeze(0)  # keep 4 dimensions
            target_one_image = target[i].unsqueeze(0)
            student_iou_i = get_mIoU_from_predictions(args, student_prediction, target_one_image)
            teacher_prediction = teacher_predictions[i].unsqueeze(0)
            teacher_iou_i = get_mIoU_from_predictions(args, teacher_prediction, target_one_image)
            # logger.write('student iou: ' + str(student_iou_i))
            # logger.write('teacher iou: ' + str(teacher_iou_i))
            if (student_iou_i < teacher_iou_i):
                indexes_with_lower_student_iou.append(i)
            else:
                logger.write('student iou (' + str(student_iou_i) + ') better than teacher\'s (' + str(
                    teacher_iou_i) + '). Epoch ' + str(epoch) + ', index: ' + str(i) + ', image: ' + str(
                    index))

    return indexes_with_lower_student_iou


def get_original_loss(device, student_batch_outputs, target, spatialWeights, maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL):
    CE_loss = criterion(student_batch_outputs, target)
    loss = CE_loss * (
                torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device) + (spatialWeights).to(
            torch.float32).to(device))
    loss = torch.mean(loss).to(torch.float32).to(device)
    loss_dice = criterion_DICE(student_batch_outputs, target)
    loss_sl = torch.mean(criterion_SL(student_batch_outputs.to(device), (maxDist).to(device)))
    if alpha is None:
        loss_original = (1 - 0) * loss_sl + 0 * (loss_dice) + loss # if training from pruning, take alpha=0, because you are training pretrained model
    else:
        loss_original = (1 - alpha[epoch]) * loss_sl + alpha[epoch] * (loss_dice) + loss
    """
    logger.write('CE loss (using spatialWeights): ' + str(loss.item()))
    logger.write('DICE loss: ' + str(loss_dice.item()))
    logger.write('loss SL: ' + str(loss_sl.item()))
    logger.write('alpha: ' + str(alpha[epoch]))
    logger.write('original loss: ' + str(loss_original.item()))
    logger.write('--------------------------')
    """
    return loss_original


def get_hinton_loss(device, student_model, teacher_model, indexes_with_lower_student_iou, student_batch_outputs, teacher_batch_outputs, parameters):
    if parameters['alpha_distillation'] == 0.0:
        return None

    if not indexes_with_lower_student_iou: # if empty, then return 0.0
        return None

    # ADD ALL INTERMEDIATE LAYERS
    torch_indices_lower_student_iou = torch.tensor(indexes_with_lower_student_iou).to(device)
    dim = torch.tensor(0).to(device)

    if len(indexes_with_lower_student_iou) > student_model.x1.shape[0]:
        logger.write('x1: len(indices) > x1 batch size')
    s_x1 = torch.index_select(student_model.x1, dim, torch_indices_lower_student_iou)
    t_x1 = torch.index_select(teacher_model.x1, dim, torch_indices_lower_student_iou)
    loss_ce_x1 = cross_entropy_with_soft_targets(s_x1, t_x1, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x3.shape[0]:
        logger.write('x3: len(indices) > x3 batch size')
    s_x3 = torch.index_select(student_model.x3, dim, torch_indices_lower_student_iou)
    t_x3 = torch.index_select(teacher_model.x3, dim, torch_indices_lower_student_iou)
    loss_ce_x3 = cross_entropy_with_soft_targets(s_x3, t_x3, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x5.shape[0]:
        logger.write('x5: len(indices) > x5 batch size')
    s_x6 = torch.index_select(student_model.x6, dim, torch_indices_lower_student_iou)
    t_x7 = torch.index_select(teacher_model.x7, dim, torch_indices_lower_student_iou)
    loss_ce_x6 = cross_entropy_with_soft_targets(s_x6, t_x7, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x6.shape[0]:
        logger.write('x6: len(indices) > x6 batch size')
    s_x7 = torch.index_select(student_model.x7, dim, torch_indices_lower_student_iou)
    t_x9 = torch.index_select(teacher_model.x9, dim, torch_indices_lower_student_iou)
    loss_ce_x7 = cross_entropy_with_soft_targets(s_x7, t_x9, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x7.shape[0]:
        logger.write('x7: len(indices) > x7 batch size')
    # add smallest layer:
    s_x5 = torch.index_select(student_model.x5, dim, torch_indices_lower_student_iou)
    t_x5 = torch.index_select(teacher_model.x5, dim, torch_indices_lower_student_iou)
    loss_ce_smallest_layer = cross_entropy_with_soft_targets(s_x5, t_x5, parameters['T'])

    s_output = torch.index_select(student_batch_outputs, dim, torch_indices_lower_student_iou)
    t_output = torch.index_select(teacher_batch_outputs, dim, torch_indices_lower_student_iou)
    loss_ce_output = cross_entropy_with_soft_targets(s_output, t_output, parameters['T'])
    loss_kd = loss_ce_smallest_layer + loss_ce_output + loss_ce_x1 + loss_ce_x3 + loss_ce_x6 + loss_ce_x7

    return loss_kd


def get_attention_loss(device, student_model, teacher_model, indexes_with_lower_student_iou, parameters):
    if parameters['beta'] == 0.0:
        return None

    if not indexes_with_lower_student_iou: # if empty, then return 0.0
        return None

    torch_indices_lower_student_iou = torch.tensor(indexes_with_lower_student_iou).to(device)#.long()
    dim = torch.tensor(0).to(device)

    # print(student_model.att_map_x1_sum.shape)  # torch.Size([4, 640, 400])

    # ATTENTION
    s_att_map_x1 = torch.index_select(student_model.att_map_x1_sum, dim, torch_indices_lower_student_iou)
    t_att_map_x1 = torch.index_select(teacher_model.att_map_x1_sum, dim, torch_indices_lower_student_iou)
    # TODO spremeni beta on the fly, beta=1000/(height*width*bs) -> w in h sta od attention mapa
    # zagoruyko paper, page 5, equation (2)
    loss_att_x1 = torch.norm(
        s_att_map_x1 / torch.norm(s_att_map_x1, 2) - t_att_map_x1 / torch.norm(t_att_map_x1, 2), 2)

    s_att_map_x3 = torch.index_select(student_model.att_map_x3_sum, dim, torch_indices_lower_student_iou)
    t_att_map_x3 = torch.index_select(teacher_model.att_map_x3_sum, dim, torch_indices_lower_student_iou)
    loss_att_x3 = torch.norm(
        s_att_map_x3 / torch.norm(s_att_map_x3, 2) - t_att_map_x3 / torch.norm(t_att_map_x3, 2), 2)

    s_att_map_x5 = torch.index_select(student_model.att_map_x5_sum, dim, torch_indices_lower_student_iou)
    t_att_map_x5 = torch.index_select(teacher_model.att_map_x5_sum, dim, torch_indices_lower_student_iou)
    loss_att_x5 = torch.norm(
        s_att_map_x5 / torch.norm(s_att_map_x5, 2) - t_att_map_x5 / torch.norm(t_att_map_x5, 2), 2)

    s_att_map_x6 = torch.index_select(student_model.att_map_x6_sum, dim, torch_indices_lower_student_iou)
    t_att_map_x7 = torch.index_select(teacher_model.att_map_x7_sum, dim, torch_indices_lower_student_iou)
    loss_att_x6 = torch.norm(
        s_att_map_x6 / torch.norm(s_att_map_x6, 2) - t_att_map_x7 / torch.norm(t_att_map_x7, 2), 2)

    s_att_map_x7 = torch.index_select(student_model.att_map_x7_sum, dim, torch_indices_lower_student_iou)
    t_att_map_x9 = torch.index_select(teacher_model.att_map_x9_sum, dim, torch_indices_lower_student_iou)
    loss_att_x7 = torch.norm(s_att_map_x7 / torch.norm(s_att_map_x7, 2) - t_att_map_x9 / torch.norm(t_att_map_x9, 2), 2)

    beta_x1 = 1000 / (np.prod(list(s_att_map_x1.shape))) # 1000 / n_elements in list * batch_size
    beta_x3 = 1000 / (np.prod(list(s_att_map_x3.shape)))
    beta_x5 = 1000 / (np.prod(list(s_att_map_x5.shape)))
    beta_x6 = 1000 / (np.prod(list(s_att_map_x6.shape)))
    beta_x7 = 1000 / (np.prod(list(s_att_map_x7.shape)))
    loss_attention = beta_x1 * loss_att_x1 + beta_x3 * loss_att_x3 + beta_x5 * loss_att_x5 + beta_x6 * loss_att_x6 + beta_x7 * loss_att_x7
    #loss_attention = beta_x7 * loss_att_x7

    return loss_attention


def get_fsp_loss(device, student_model, teacher_model, indexes_with_lower_student_iou, parameters):
    if parameters['lambda'] == 0.0:
        return None

    if not indexes_with_lower_student_iou: # if empty, then return 0.0
        return None

    torch_indices_lower_student_iou = torch.tensor(indexes_with_lower_student_iou).to(device)
    dim = torch.tensor(0).to(device)

    # TEACHER FSP MATRICES
    teacher_db1_first_layer = torch.index_select(teacher_model.db1_first_fsp_layer, dim, torch_indices_lower_student_iou)  # torch.Size([4, 32, 640, 400])
    teacher_db5_last_layer = torch.index_select(teacher_model.db5_last_fsp_layer, dim, torch_indices_lower_student_iou)  # torch.Size([4, 32, 40, 25])
    teacher_ub4_last_layer = torch.index_select(teacher_model.ub4_last_fsp_layer, dim, torch_indices_lower_student_iou)  # torch.Size([4, 32, 640, 400])

    # maxpool for db1 and ub4 from 640x400 to 40x25
    maxpool_teacher = nn.MaxPool2d(kernel_size=(16, 16))
    teacher_db1_first_layer_downsampled = maxpool_teacher(teacher_db1_first_layer)  # torch.Size([4, 32, 40, 25])
    teacher_ub4_last_layer_downsampled = maxpool_teacher(teacher_ub4_last_layer)  # torch.Size([4, 32, 40, 25])

    # encoder fsp matrix
    h = teacher_db1_first_layer_downsampled.shape[2]
    w = teacher_db1_first_layer_downsampled.shape[3]
    teacher_encoder_fsp_matrix = teacher_db1_first_layer_downsampled * teacher_db5_last_layer / (h * w)
    # print(teacher_encoder_fsp_matrix.shape) # torch.Size([4, 32, 40, 25])
    teacher_decoder_fsp_matrix = teacher_ub4_last_layer_downsampled * teacher_db5_last_layer / (h * w)

    # STUDENT FSP MATRICES
    student_db1_first_layer = torch.index_select(student_model.db1_first_fsp_layer, dim, torch_indices_lower_student_iou)  # torch.Size([4, 32, 640, 400])
    student_db5_last_layer = torch.index_select(student_model.db5_last_fsp_layer, dim, torch_indices_lower_student_iou)  # torch.Size([4, 32, 40, 25])
    student_ub2_last_layer = torch.index_select(student_model.ub2_first_fsp_layer, dim, torch_indices_lower_student_iou)  # FIRST == LAST  # torch.Size([4, 32, 640, 400])

    # maxpool for db1 and ub4 from 640x400 to 40x25
    maxpool_student = nn.MaxPool2d(kernel_size=(16, 16))
    student_db1_first_layer_downsampled = maxpool_student(student_db1_first_layer)  # torch.Size([4, 32, 40, 25])
    student_ub2_last_layer_downsampled = maxpool_student(student_ub2_last_layer)  # torch.Size([4, 32, 40, 25])

    # encoder fsp matrix
    h = student_db1_first_layer_downsampled.shape[2]
    w = student_db1_first_layer_downsampled.shape[3]
    student_encoder_fsp_matrix = student_db1_first_layer_downsampled * student_db5_last_layer / (h * w)
    student_decoder_fsp_matrix = student_ub2_last_layer_downsampled * student_db5_last_layer / (h * w)

    # 2. CREATE FSP LOSS FROM FSP TEACHER MATRIX AND FSP STUDENT MATRIX
    # equation (2) TODO

    encoder_fsp_loss_for_batch = 0.0
    decoder_fsp_loss_for_batch = 0.0
    # vzemi za vsako sliko posebej...
    for i in range(teacher_encoder_fsp_matrix.shape[0]):  # bs
        encoder_diff_norm2 = torch.norm(teacher_encoder_fsp_matrix[i] - student_encoder_fsp_matrix[i], 2)
        encoder_squared = torch.pow(encoder_diff_norm2, 2)
        encoder_fsp_loss_for_batch += encoder_squared

        decoder_diff_norm2 = torch.norm(teacher_decoder_fsp_matrix[i] - student_decoder_fsp_matrix[i], 2)
        decoder_squared = torch.pow(decoder_diff_norm2, 2)
        decoder_fsp_loss_for_batch += decoder_squared

    fsp_loss = encoder_fsp_loss_for_batch + decoder_fsp_loss_for_batch
    return fsp_loss


def calculate_loss_for_batch_CE(args, device, student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights, maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL):
    # ORIGINAL LOSS
    loss_original = get_original_loss(device, student_batch_outputs, target, spatialWeights, maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL)

    indexes_with_lower_student_iou = get_indexes_with_lower_student_iou(args, student_batch_outputs, teacher_batch_outputs, target, epoch, index)

    # hinton loss
    loss_hinton = get_hinton_loss(device, student_model, teacher_model, indexes_with_lower_student_iou, student_batch_outputs, teacher_batch_outputs, parameters)

    # attention loss
    loss_attention = get_attention_loss(device, student_model, teacher_model, indexes_with_lower_student_iou, parameters)

    # fsp loss
    loss_fsp = get_fsp_loss(device, student_model, teacher_model, indexes_with_lower_student_iou, parameters)

    # SUM LOSSES ACCORDING TO PARAMETERS
    loss = parameters['alpha_original'] * loss_original
    if loss_hinton is not None:
        # Since the magnitudes of the gradients produced by the soft targets scale as 1/T2 it is important to multiply them by T2 when using both hard and soft targets
        loss_hinton = (parameters['alpha_distillation'] * parameters['T'] * parameters['T']) * loss_hinton
        loss = loss + loss_hinton
    else:
        loss_hinton = torch.tensor(0.0).to(device)
    if loss_attention is not None:
        loss_attention = parameters['beta'] * loss_attention
        loss = loss + loss_attention
    else:
        loss_attention = torch.tensor(0.0).to(device)
    if loss_fsp is not None:
        loss_fsp = parameters['lambda'] * loss_fsp
        loss = loss + loss_fsp
    else:
        loss_fsp = torch.tensor(0.0).to(device)

    return loss, loss_hinton.item(), loss_attention.item(), loss_fsp.item()

"""
def calculate_loss_for_batch_original_with_kldiv(student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights, maxDist, alpha, epoch, alpha_distillation, alpha_original, T):
    # ORIGINAL LOSS
    CE_loss = criterion(student_batch_outputs, target)
    loss = CE_loss * (torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device) + (spatialWeights).to(torch.float32).to(device))
    loss = torch.mean(loss).to(torch.float32).to(device)
    loss_dice = criterion_DICE(student_batch_outputs, target)
    loss_sl = torch.mean(criterion_SL(student_batch_outputs.to(device), (maxDist).to(device)))
    loss_original = (1 - alpha[epoch]) * loss_sl + alpha[epoch] * (loss_dice) + loss

    # report linkan iz github repozrotija pravi, da so uporabili KLDivLoss zaradi efficiency (namesto CE) TODO primerjaj oboje?
    # PRIMERJAJ TEACHERJA IN STUDENTA, CE JE STUDENT BOLJSI, DEJ LOSS NJEGOV NA 0
    indexes_with_lower_student_iou = get_indexes_with_lower_student_iou(student_batch_outputs, teacher_batch_outputs, target, epoch, index)

    if args.alwaysPenalize.lower() == 'true' or (
            args.alwaysPenalize.lower() != 'true' and indexes_with_lower_student_iou):

        if alpha_distillation != 0.0:
            # ADD ALL INTERMEDIATE LAYERS
            torch_indices_lower_student_iou = torch.tensor(indexes_with_lower_student_iou).to(device)
            dim = torch.tensor(0).to(device)
            #print(student_model.x1.shape) # (4, 32, 640, 400)
            #print(len(torch_indices_lower_student_iou)) # 3 ali 4

            if len(indexes_with_lower_student_iou) > student_model.x1.shape[0]:
                print('x1: len(indices) > x1 batch size')
            s_x1 = torch.index_select(student_model.x1, dim, torch_indices_lower_student_iou)
            t_x1 = torch.index_select(teacher_model.x1, dim, torch_indices_lower_student_iou)
            loss_kldiv_x1 = nn.KLDivLoss()(F.log_softmax(s_x1 / T, dim=1),
                                           F.softmax(t_x1 / T, dim=1))
            if len(indexes_with_lower_student_iou) > student_model.x3.shape[0]:
                print('x3: len(indices) > x3 batch size')
            s_x3 = torch.index_select(student_model.x3, dim, torch_indices_lower_student_iou)
            t_x3 = torch.index_select(teacher_model.x3, dim, torch_indices_lower_student_iou)
            loss_kldiv_x3 = nn.KLDivLoss()(F.log_softmax(s_x3 / T, dim=1),
                                           F.softmax(t_x3 / T, dim=1))
            if len(indexes_with_lower_student_iou) > student_model.x5.shape[0]:
                print('x5: len(indices) > x5 batch size')
            s_x6 = torch.index_select(student_model.x6, dim, torch_indices_lower_student_iou)
            t_x7 = torch.index_select(teacher_model.x7, dim, torch_indices_lower_student_iou)
            loss_kldiv_x6 = nn.KLDivLoss()(F.log_softmax(s_x6 / T, dim=1),
                                           F.softmax(t_x7 / T, dim=1))
            if len(indexes_with_lower_student_iou) > student_model.x6.shape[0]:
                print('x6: len(indices) > x6 batch size')
            s_x7 = torch.index_select(student_model.x7, dim, torch_indices_lower_student_iou)
            t_x9 = torch.index_select(teacher_model.x9, dim, torch_indices_lower_student_iou)
            loss_kldiv_x7 = nn.KLDivLoss()(F.log_softmax(s_x7 / T, dim=1),
                                           F.softmax(t_x9 / T, dim=1))
            if len(indexes_with_lower_student_iou) > student_model.x7.shape[0]:
                print('x7: len(indices) > x7 batch size')
            #add smallest layer:
            s_x5 = torch.index_select(student_model.x5, dim, torch_indices_lower_student_iou)
            t_x5 = torch.index_select(teacher_model.x5, dim, torch_indices_lower_student_iou)
            loss_kldiv_smallest_layer = nn.KLDivLoss()(F.log_softmax(s_x5 / T, dim=1),
                                                       F.softmax(t_x5 / T, dim=1))
            s_output = torch.index_select(student_batch_outputs, dim, torch_indices_lower_student_iou)
            t_output = torch.index_select(teacher_batch_outputs, dim, torch_indices_lower_student_iou)
            loss_kldiv_output = nn.KLDivLoss()(F.log_softmax(s_output / T, dim=1),
                                              F.softmax(t_output / T, dim=1))
            loss_kd = loss_kldiv_smallest_layer + loss_kldiv_output + loss_kldiv_x1 + loss_kldiv_x3 + loss_kldiv_x6 + loss_kldiv_x7
        else:
            loss_kd = 0.0
    else:
        loss_kd = 0.0

    # SUM THOSE TWO LOSSES
    # Since the magnitudes of the gradients produced by the soft targets scale as 1/T2 it is important to multiply them by T2 when using both hard and soft targets
    if alpha_distillation != 0.0:
        loss = (alpha_distillation * T * T) * loss_kd + alpha_original * loss_original
    else:
        loss = loss_original

    if isinstance(loss_kd, float):
        loss_hinton_float = (alpha_distillation * T * T) * loss_kd
    else:
        loss_hinton_float = (alpha_distillation * T * T) * loss_kd.item()

    return loss, loss_hinton_float
"""


def validation_loss(args, device, loader, student_model, teacher_model, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL):
    epoch_loss = []
    n_classes = 4 if 'sip' in args.dataset.lower() else 2
    conf_matrix_whole = np.zeros((n_classes, n_classes))
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()

            student_batch_outputs = student_model(data)
            teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data)

            #loss, loss_hinton_float = calculate_loss_for_batch_original_with_kldiv(student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights, maxDist, alpha, epoch, alpha_distillation, alpha_original, T)
            loss, _, _, _ = calculate_loss_for_batch_CE(args, device, student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights,
                                                                  maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL)

            epoch_loss.append(loss.item())
            predict = get_predictions(student_batch_outputs)
            conf_matrix_batch = get_conf_matrix(args, predict, target)
            conf_matrix_whole += conf_matrix_batch

    average_val_iou = conf_matrix_to_mIoU(args, conf_matrix_whole)
    return np.average(epoch_loss), average_val_iou, np.sum(epoch_loss)


def fetch_teacher_outputs(args, teacher_model, data):
    if args.useGPU != 'True' and args.useGPU != 'true':
        raise NotImplementedError("Use GPU, CPU not implemented.")

    if teacher_model is None:
        return None

    # set teacher_model to evaluation mode
    set_model_to_eval(teacher_model)
    with torch.no_grad():
        output_teacher_batch = teacher_model(data)

    return output_teacher_batch


def compare_models(model_1, model_2):
    set_model_to_eval(model_1)
    set_model_to_eval(model_2)
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                logger.write('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        return True
        #print('Models match perfectly! :)')
    else:
        return False


def load_student(args, device):
    key = f'student-{args.model}4' if 'sip' in args.dataset.lower() else f'student-{args.model}'
    student_model = model_dict[key]
    logger.write('using student model: ' + str(type(student_model)))
    student_model = student_model.to(device)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=135, gamma=0.1)
    if args.resume != '':
        logger.write("EXISTING STUDENT DICT from: {}".format(args.resume))
        student_state_dict = torch.load(args.resume)
        student_model.load_state_dict(student_state_dict)
        # student_model.eval() # not needed if training continues

        optimizer_state_dict_path = args.resume.replace('models', 'optimizers')
        logger.write("EXISTING OPTIMIZER DICT from: {}".format(optimizer_state_dict_path))
        optimizer_state_dict = torch.load(optimizer_state_dict_path)
        optimizer.load_state_dict(optimizer_state_dict)
        if args.useGPU:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        scheduler_path = args.resume.replace('models', 'schedulers')
        scheduler = torch.load(scheduler_path)

    return student_model, optimizer, scheduler


def load_teacher(args, device):
    if args.teacher == '':
        return None

    teacher_state_dict = torch.load(args.teacher)
    # vedet moraš tudi katera arhitektura je bla shranjena
    key = f'teacher-{args.model}4' if 'sip' in args.dataset.lower() else f'teacher-{args.model}'
    teacher_model = model_dict[key]
    logger.write('using teacher model: ' + str(type(teacher_model)))
    teacher_model = teacher_model.to(device)
    teacher_model.load_state_dict(teacher_state_dict)
    logger.write('teacher.training: ' + str(teacher_model.training))
    # print(teacher_model.parameters())
    set_model_to_train(teacher_model) # it is already in train
    # cancel gradients for teacher model, because we don't want to update this network
    for params in teacher_model.parameters():
        params.requires_grad = False

    return teacher_model


def validate(args, device, student_model, validloader, teacher_model,  viz, win_loss, win_iou, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL):
    lossvalid, miou, validation_loss_sum = validation_loss(args, device, validloader, student_model, teacher_model, alpha, epoch,
                                                           parameters, criterion, criterion_DICE, criterion_SL)
    # totalperf = total_metric(nparams, miou)
    f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {}'
    logger.write(f.format(epoch, lossvalid, miou))

    # visualize validation loss
    plot_with_visdom(viz, win_loss, epoch, validation_loss_sum, 'validation loss')
    # visualize validation iou
    plot_with_visdom(viz, win_iou, epoch, miou, 'validation IoU')

    return lossvalid, validation_loss_sum # nedded for scheduler, needed for stopping criteria


def evaluate_on_test_during_train(args, device, student_model, testloader, teacher_model, epoch, viz, win_iou, visualize_on_test_set=True):
    n_classes = 4 if 'sip' in args.dataset.lower() else 2
    conf_matrix_whole = np.zeros((n_classes, n_classes))
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
            img, label_tensor, filename, x, maxDist = batchdata
            data = img.to(device)
            target = label_tensor.to(device).long()
            output = student_model(data)
            student_predictions = get_predictions(output)

            conf_matrix_batch = get_conf_matrix(args, student_predictions, target)
            conf_matrix_whole += conf_matrix_batch

            # VISUALIZE
            if visualize_on_test_set:
                if epoch % 10 == 0:
                    visualize_on_test_images(args, epoch, filename, teacher_model, img, data, label_tensor, student_predictions)

        miou = conf_matrix_to_mIoU(args, conf_matrix_whole)

        # visualize test iou
        plot_with_visdom(viz, win_iou, epoch, miou, 'test iou')


def visualize_on_test_images(args, epoch, filename, teacher_model, img, data, label_tensor, student_predictions):
    if teacher_model is not None:
        teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data)
        teacher_predictions = get_predictions(teacher_batch_outputs)

        os.makedirs('test/mask/{}/'.format(args.expname), exist_ok=True)
        os.makedirs('test/mask/{}/epoch_{}'.format(args.expname, epoch), exist_ok=True)
        for j in range(len(filename)):
            pred_img = student_predictions[j].cpu().numpy()  # /3.0
            pred_img_teacher = teacher_predictions[j].cpu().numpy()  # /3.0
            #pred_img[pred_img > 1] = 0 # TODO odstrani, da se bojo vidle predikcije cseh 4 razredov
            #pred_img_teacher[pred_img_teacher > 1] = 0
            #print(np.unique(pred_img))
            #print(np.unique(pred_img_teacher))

            inp = img[j].squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp, 0, 1)
            img_orig = np.array(img_orig)
            label = label_tensor[j].view(args.height, args.width)
            label = np.array(label)

            combine = np.hstack([img_orig, pred_img, pred_img_teacher, label])
            plt.imsave('test/mask/{}/epoch_{}/{}.jpg'.format(args.expname, epoch, filename[j]),
                       combine)

    else:  # plot only student and original (without teacher output)
        os.makedirs('test/mask/{}/'.format(args.expname), exist_ok=True)
        os.makedirs('test/mask/{}/epoch_{}'.format(args.expname, epoch), exist_ok=True)
        for j in range(len(filename)):
            pred_img = student_predictions[j].cpu().numpy()  # /3.0
            #pred_img[pred_img > 1] = 0
            #print(np.unique(pred_img))
            inp = img[j].squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp, 0, 1)
            img_orig = np.array(img_orig)
            label = label_tensor[j].view(args.height, args.width)
            label = np.array(label)

            combine = np.hstack([img_orig, pred_img, label])
            plt.imsave('test/mask/{}/epoch_{}/{}.jpg'.format(args.expname, epoch, filename[j]),
                       combine)


def plot_with_visdom(viz, win_loss, epoch, value, description):
    try:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([value]),
            win=win_loss,
            name=description,
            update='append',
            #opts=dict(
            #    width=500,
            #    height=500
            #)
        )
    except Exception:
        pass


def train(args, device, teacher_model, trainloader, optimizer, student_model, validloader, testloader, viz, win_loss, win_iou, parameters, scheduler, criterion, criterion_DICE, criterion_SL):
    alpha = parameters['alpha']

    set_model_to_train(student_model)
    set_model_to_eval(teacher_model)
    #nparams = get_nparams(student_model)
    teacher_model_copy = copy.deepcopy(teacher_model)

    prev_prev_val_loss = 10000
    prev_val_loss = 10000
    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):
        set_model_to_train(student_model)
        assert student_model.training, 'Student must be in training mode'
        if teacher_model is not None:
            assert not teacher_model.training, 'Teacher must be in eval mode'

        if teacher_model is not None:
            models_equal = compare_models(teacher_model_copy, teacher_model)
            if not models_equal:
                logger.write('Second comparison is false!')

        training_hinton_loss_sum = 0.0
        training_attention_loss_sum = 0.0
        training_fsp_loss_sum = 0.0
        n_classes = 4 if 'sip' in args.dataset.lower() else 2
        conf_matrix_whole = np.zeros((n_classes, n_classes))
        training_loss_sum = 0.0
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()
            optimizer.zero_grad()

            student_batch_outputs = student_model(data)
            teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data)
            loss, loss_hinton_float, loss_attention_float, loss_fsp_float = calculate_loss_for_batch_CE(args, device, student_model, student_batch_outputs, teacher_model,
                                                                  teacher_batch_outputs, target, index, spatialWeights,
                                                                  maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL)

            # 4. backprop only on student model
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            training_loss_sum += loss.item()

            training_hinton_loss_sum += loss_hinton_float
            training_attention_loss_sum += loss_attention_float
            training_fsp_loss_sum += loss_fsp_float

            predictions = get_predictions(student_batch_outputs)
            conf_matrix_batch = get_conf_matrix(args, predictions, target)
            conf_matrix_whole += conf_matrix_batch

            if i % 10 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), loss.item()))
            #gradients = torch.zeros(0).to(device)
            """
            for name, param in student_model.named_parameters():
                #print(name)
                #print(torch.min(param.grad))
                #print(torch.max(param.grad))
                flatten_tensor = torch.flatten(param.grad)
                gradients = torch.cat((gradients, flatten_tensor))
            """
            #print(gradients.shape)
            """
            logger.write('gradients max: ' + str(torch.max(gradients)))
            logger.write('gradients min: ' + str(torch.min(gradients)))
            logger.write('gradients mean: ' + str(torch.mean(gradients)))
            gradients = torch.abs(gradients)
            logger.write('gradients abs max: ' + str(torch.max(gradients)))
            logger.write('gradients abs min: ' + str(torch.min(gradients)))
            logger.write('-------------------')
            """
            #plot_grad_flow(student_model.named_parameters())
            #print('plotting done..')


        # visualize total training loss
        plot_with_visdom(viz, win_loss, epoch, training_loss_sum, 'training loss')
        plot_with_visdom(viz, win_loss, epoch, training_hinton_loss_sum, 'training hinton loss')
        plot_with_visdom(viz, win_loss, epoch, training_attention_loss_sum, 'training attention loss')
        plot_with_visdom(viz, win_loss, epoch, training_fsp_loss_sum, 'training fsp loss')

        miou_train = conf_matrix_to_mIoU(args, conf_matrix_whole)
        logger.write('Epoch:{}, Train mIoU: {}'.format(epoch, miou_train))

        # *********VALIDATION*******************************************************************************************

        set_model_to_eval(student_model)  # for validation loss and also test afterwards
        set_model_to_eval(teacher_model) # should already be in eval mode
        assert not student_model.training, 'Student must be in eval mode'
        if teacher_model is not None:
            assert not teacher_model.training, 'Teacher must be in eval mode'

        lossvalid, validation_loss_sum = validate(args, device, student_model, validloader, teacher_model,  viz, win_loss, win_iou, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL)
        scheduler.step(lossvalid)

        # every epoch calculate test IoU
        evaluate_on_test_during_train(args, device, student_model, testloader, teacher_model, epoch, viz, win_iou)

        # SAVE MODEL
        if epoch % 5 == 0:
            torch.save(student_model.state_dict(), LOGDIR/'models'/f'dense_net_{epoch}.pkl')
            torch.save(optimizer.state_dict(), LOGDIR/'optimizers'/f'dense_net_{epoch}.pkl')
            torch.save(scheduler, LOGDIR/'schedulers'/f'dense_net_{epoch}.pkl')

        # stopping criteria
        if prev_prev_val_loss < prev_val_loss and prev_val_loss < validation_loss_sum:
            logger.write('validation loss increased two times in a row')
            break

        # save validation losses
        prev_prev_val_loss = prev_prev_val_loss
        prev_val_loss = validation_loss_sum


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.show()


def set_model_to_train(model):
    if model is not None and not model.training:
        logger.write('setting model ' + str(type(model)) + ' to train()')
        model.train()


def set_model_to_eval(model):
    if model is not None and model.training:
        logger.write('setting model ' + str(type(model)) + ' to eval()')
        model.eval()


def get_data_loaders(args, ):
    kwargs = vars(args)
    Path2file = args.dataset
    n_classes = 4 if 'sip' in args.dataset.lower() else 2
    logger.write('path to file: ' + str(Path2file))
    train_dataset = IrisDataset(filepath=Path2file, split='train',
                        transform=transform,n_classes=n_classes, **kwargs)
    valid = IrisDataset(filepath=Path2file, split='validation',
                        transform=transform,n_classes=n_classes, **kwargs)

    trainloader = DataLoader(train_dataset, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True) # TODO shuffle??

    validloader = DataLoader(valid, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True)

    test = IrisDataset(filepath=Path2file, split='test',
                       transform=transform,n_classes=n_classes, **kwargs)

    testloader = DataLoader(test, batch_size=args.bs,
                            shuffle=False, num_workers=args.workers)

    logger.write('train dataset len: ' + str(train_dataset.__len__()))
    logger.write('val dataset len: ' + str(valid.__len__()))
    logger.write('test dataset len: ' + str(test.__len__()))

    return trainloader, validloader, testloader


def initialize_globals(args):
    global LOGDIR
    LOGDIR = Path('logs')/args.expname
    global logger
    logger = Logger(LOGDIR/'logs.log')


def main():
    args = parse_args()

    initialize_globals(args)
    (LOGDIR/'models').mkdir(parents=True, exist_ok=True)
    (LOGDIR/'optimizers').mkdir(parents=True, exist_ok=True)
    (LOGDIR/'schedulers').mkdir(parents=True, exist_ok=True)

    if args.useGPU == 'True' or args.useGPU == 'true':
        logger.write('USE GPU')
        device = torch.device("cuda")
        # os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        torch.cuda.manual_seed(7)
    else:
        logger.write('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(7)

    torch.backends.cudnn.deterministic = True

    teacher_model = load_teacher(args, device)
    student_model, optimizer, scheduler = load_student(args, device)

    #from torchsummary import summary
    #summary(student_model, input_size=(1, args.height, args.width))  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    #print(student_model)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    criterion_SL = SurfaceLoss()

    # visdom
    # RUN python -m visdom.server
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)
    win_loss = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            showlegend=True,
            width=550,
            height=400
        )
    )
    win_iou = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            showlegend=True,
            width=550,
            height=400
        )
    )

    n_epochs = args.startEpoch + args.epochs
    alpha = np.zeros((n_epochs))
    alpha[0:np.min([125, n_epochs])] = 1 - np.arange(1, np.min([125, n_epochs]) + 1) / np.min([125, n_epochs])
    if args.epochs > 125:
        alpha[125:] = 0

    # PARAMETERS:
    parameters = {
        'alpha': alpha,
        'alpha_original': 1,
        # -------------------HINTON-------------------
        'alpha_distillation': 100,
        'T': 8, # TODO 2, 4, 8, 16
        # ---------------ATTENTION------------------
        'beta': 0.00,
        # -------------FSP------------------
        'lambda': 0.00
    }

    if teacher_model is None and (parameters['alpha_distillation'] != 0.0 or parameters['beta'] != 0.0 or parameters['lambda'] != 0.0):
        logger.write('When using distillation methods, teacher model must be present!')
        return 0

    opt = vars(args)
    logger.write(str(opt))
    logger.write(str(parameters))

    trainloader, validloader, testloader = get_data_loaders(args)

    train(args, device, teacher_model, trainloader, optimizer, student_model, validloader, testloader, viz, win_loss, win_iou, parameters, scheduler, criterion, criterion_DICE, criterion_SL)


if __name__ == '__main__':
    main()

# python train_with_knowledge_distillation.py --expname student_without_distillation --gpu 0
# python train_with_knowledge_distillation.py --expname student_distillation_hinton_a1_T8_lr0_001_all_intermediate_layers --gpu 1 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl

# python train_with_knowledge_distillation.py --expname student_distillation_hinton_a0_1_T8_lr0_001_all_intermediate_layers --gpu 0 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl
# python train_with_knowledge_distillation.py --expname student_distillation_hinton_a0_01_T8_lr0_001_all_intermediate_layers --gpu 1 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl




# python train_with_knowledge_distillation.py --expname student_distillation_attention_b1_lr0_001_all_intermediate_layers_sum_definition --gpu 0 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl
# python train_with_knowledge_distillation.py --expname student_distillation_attention_b0_1_lr0_001_all_intermediate_layers_sum_definition --gpu 1 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl

# python train_with_knowledge_distillation.py --expname student_distillation_attention_b0_01_lr0_001_all_intermediate_layers_sum_definition --gpu 0 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl
# python train_with_knowledge_distillation.py --expname student_distillation_attention_b0_001_lr0_001_all_intermediate_layers_sum_definition --gpu 1 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl


# FSP START: (imas teacherja ker delas destilacijo, ampak nimas resume ker treniras from scratch
# python train_with_knowledge_distillation.py --expname student_distillation_fsp_lambda0_001_lr0_001_v1  --gpu 1 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl

# FSP RESUME: (brez teacherja, ker retrinas normalno brez destilacije)
# python train_with_knowledge_distillation.py --expname student_distillation_fsp_lambda0_001_lr0_001_v1_RUN2_RETRAINED --resume logs/student_distillation_fsp_lambda0_001_lr0_001_v1_RUN2/models/dense_net_200.pkl --gpu 0

# python train_with_knowledge_distillation.py --expname student_distillation_fsp_lambda0_000001_lr0_001_v1_RUN3_RETRAINED --resume logs/student_distillation_fsp_lambda0_000001_lr0_001_v1_RUN3/models/dense_net_200.pkl --gpu 1



#  kombinacija hinton + attention
# python train_with_knowledge_distillation.py --expname student_distillation_hinton_attention_a0_0001_T8_b0_001_lr0_001_all_intermediate_layers_sum_definition_ --gpu 0 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl
# python train_with_knowledge_distillation.py --expname student_distillation_hinton_attention_alpha_original2_a0_0001_T8_b0_001_lr0_001_all_intermediate_layers_sum_definition_ --gpu 0 --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl
