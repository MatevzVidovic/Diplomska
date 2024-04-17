#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy

from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import IrisDataset
import torch

from test import conf_matrix_to_iou, classify_to_group
from utils import mIoU, CrossEntropyLoss2d, total_metric, get_nparams, Logger, GeneralizedDiceLoss, SurfaceLoss, \
    mIoU_conf_matrix
import numpy as np
from dataset import transform
from opt import parse_args
import os
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from visdom import Visdom
from pprint import pprint
from models import model_dict


def calculate_loss_for_batch(student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights, maxDist, alpha, epoch, alpha_distillation, T, beta):
    # ORIGINAL LOSS
    CE_loss = criterion(student_batch_outputs, target)
    loss = CE_loss * (torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device) + (
        spatialWeights).to(torch.float32).to(device))

    loss = torch.mean(loss).to(torch.float32).to(device)
    loss_dice = criterion_DICE(student_batch_outputs, target)
    loss_sl = torch.mean(criterion_SL(student_batch_outputs.to(device), (maxDist).to(device)))
    loss_original = (1 - alpha[epoch]) * loss_sl + alpha[epoch] * (loss_dice) + loss

    # ATTENTION MAPS LOSS (DISTILLATION LOSS)
    # PRIMERJAJ TEACHERJA IN STUDENTA, CE JE STUDENT BOLJSI, DEJ LOSS NJEGOV NA 0
    indexes_with_lower_student_iou = []
    if args.alwaysPenalize.lower() == 'true':
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
            student_iou_i = mIoU(student_prediction, target_one_image)
            teacher_prediction = teacher_predictions[i].unsqueeze(0)
            teacher_iou_i = mIoU(teacher_prediction, target_one_image)
            # logger.write('student iou: ' + str(student_iou_i))
            # logger.write('teacher iou: ' + str(teacher_iou_i))
            if (student_iou_i < teacher_iou_i):
                indexes_with_lower_student_iou.append(i)
            else:
                logger.write('student iou (' + str(student_iou_i) + ') better than teacher\'s (' + str(
                    teacher_iou_i) + '). Epoch ' + str(epoch) + ', index: ' + str(i) + ', image: ' + str(
                    index))

    # logger.write('indices in indexes_with_lower_student_iou: ' + str(indexes_with_lower_student_iou))

    if indexes_with_lower_student_iou: # and only_poor: # todo remove
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

        # s_att_map_x5 = torch.index_select(student_model.att_map_x5_sum, dim, torch_indices_lower_student_iou)
        # t_att_map_x5 = torch.index_select(teacher_model.att_map_x5_sum, dim, torch_indices_lower_student_iou)
        # loss_att_x5 = torch.norm(
        #     s_att_map_x5 / torch.norm(s_att_map_x5, 2) - t_att_map_x5 / torch.norm(t_att_map_x5, 2), 2)

        # s_att_map_x6 = torch.index_select(student_model.att_map_x6_sum, dim, torch_indices_lower_student_iou)
        # t_att_map_x7 = torch.index_select(teacher_model.att_map_x7_sum, dim, torch_indices_lower_student_iou)
        # loss_att_x6 = torch.norm(
        #     s_att_map_x6 / torch.norm(s_att_map_x6, 2) - t_att_map_x7 / torch.norm(t_att_map_x7, 2), 2)

        # s_att_map_x7 = torch.index_select(student_model.att_map_x7_sum, dim, torch_indices_lower_student_iou)
        # t_att_map_x9 = torch.index_select(teacher_model.att_map_x9_sum, dim, torch_indices_lower_student_iou)
        # loss_att_x7 = torch.norm(s_att_map_x7 / torch.norm(s_att_map_x7, 2) - t_att_map_x9 / torch.norm(t_att_map_x9, 2), 2)

        beta_x1 = 1000 / (np.prod(list(s_att_map_x1.shape))) # 1000 / n_elements in list * batch_size
        beta_x3 = 1000 / (np.prod(list(s_att_map_x3.shape)))
        # beta_x5 = 1000 / (np.prod(list(s_att_map_x5.shape)))
        # beta_x6 = 1000 / (np.prod(list(s_att_map_x6.shape)))
        # beta_x7 = 1000 / (np.prod(list(s_att_map_x7.shape)))
        loss_attention = beta_x1 * loss_att_x1 + beta_x3 * loss_att_x3 #+ beta_x5 * loss_att_x5 + beta_x6 * loss_att_x6 #+ loss_att_x7

        if alpha_distillation != 0.0:
            # HINTON
            # ADD ALL INTERMEDIATE LAYERS
            s_x1 = torch.index_select(student_model.x1, dim, torch_indices_lower_student_iou)
            t_x1 = torch.index_select(teacher_model.x1, dim, torch_indices_lower_student_iou)
            loss_kldiv_x1 = nn.KLDivLoss()(F.log_softmax(s_x1 / T, dim=1),
                                           F.softmax(t_x1 / T, dim=1))
            s_x3 = torch.index_select(student_model.x3, dim, torch_indices_lower_student_iou)
            t_x3 = torch.index_select(teacher_model.x3, dim, torch_indices_lower_student_iou)
            loss_kldiv_x3 = nn.KLDivLoss()(F.log_softmax(s_x3 / T, dim=1),
                                           F.softmax(t_x3 / T, dim=1))
            s_x6 = torch.index_select(student_model.x6, dim, torch_indices_lower_student_iou)
            t_x7 = torch.index_select(teacher_model.x7, dim, torch_indices_lower_student_iou)
            loss_kldiv_x6 = nn.KLDivLoss()(F.log_softmax(s_x6 / T, dim=1),
                                           F.softmax(t_x7 / T, dim=1))
            s_x7 = torch.index_select(student_model.x7, dim, torch_indices_lower_student_iou)
            t_x9 = torch.index_select(teacher_model.x9, dim, torch_indices_lower_student_iou)
            loss_kldiv_x7 = nn.KLDivLoss()(F.log_softmax(s_x7 / T, dim=1),
                                           F.softmax(t_x9 / T, dim=1))
            # add smallest layer:
            s_x5 = torch.index_select(student_model.x5, dim, torch_indices_lower_student_iou)
            t_x5 = torch.index_select(teacher_model.x5, dim, torch_indices_lower_student_iou)
            loss_kldiv_smallest_layer = nn.KLDivLoss()(F.log_softmax(s_x5 / T, dim=1),
                                                       F.softmax(t_x5 / T, dim=1))
            s_output = torch.index_select(student_batch_outputs, dim, torch_indices_lower_student_iou)
            t_output = torch.index_select(teacher_batch_outputs, dim, torch_indices_lower_student_iou)
            loss_kldiv_output = nn.KLDivLoss()(F.log_softmax(s_output / T, dim=1),
                                               F.softmax(t_output / T, dim=1))
            loss_hinton = loss_kldiv_smallest_layer + loss_kldiv_output + loss_kldiv_x1 + loss_kldiv_x3 + loss_kldiv_x6 + loss_kldiv_x7
        else:
            loss_hinton = 0.0
    else:
        loss_attention = 0.0
        loss_hinton = 0.0

    loss = loss_original + beta * loss_attention + (alpha_distillation * T * T) * loss_hinton

    if isinstance(loss_hinton, float):
        loss_hinton_float = (alpha_distillation * T * T) * loss_hinton
    else:
        loss_hinton_float = (alpha_distillation * T * T) * loss_hinton.item()

    if isinstance(loss_attention, float):
        loss_attention_float = beta * loss_attention
    else:
        loss_attention_float = beta * loss_attention.item()

    #print('original loss: ' + str(loss_original.item()))
    #print('zagoruyko loss: ' + str(beta * loss_attention.item()))
    return loss, loss_attention_float, loss_hinton_float, loss_original.item()


def validation_loss(loader, student_model, teacher_model, alpha, epoch, T, alpha_distillation, beta):
    epoch_loss = []
    #ious = []
    conf_matrix_whole = np.zeros((2,2))
    student_model.eval()

    validation_original_loss_sum = 0.0
    validation_hinton_loss_sum = 0.0
    validation_attention_loss_sum = 0.0

    with torch.no_grad():
        for i, batchdata in enumerate(loader):
            # print (len(batchdata))
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()
            student_batch_outputs = student_model(data)
            teacher_batch_outputs = fetch_teacher_outputs(teacher_model, data)

            loss, loss_attention_float, loss_hinton_float, loss_original_float = calculate_loss_for_batch(student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target,
                                            index, spatialWeights, maxDist, alpha, epoch, alpha_distillation, T, beta)

            validation_attention_loss_sum += loss_attention_float
            validation_hinton_loss_sum += loss_hinton_float
            validation_original_loss_sum += loss_original_float

            epoch_loss.append(loss.item())
            predict = get_predictions(student_batch_outputs)
            conf_matrix_batch = mIoU_conf_matrix(predict, target)
            conf_matrix_whole += conf_matrix_batch

    average_val_iou = conf_matrix_to_iou(conf_matrix_whole)
    return np.average(epoch_loss), average_val_iou, np.sum(epoch_loss), validation_attention_loss_sum, validation_hinton_loss_sum, validation_original_loss_sum



def fetch_teacher_outputs(teacher_model, data):
    if args.useGPU != 'True' and args.useGPU != 'true':
        raise NotImplementedError("Use GPU, CPU not implemented.")

    # set teacher_model to evaluation mode
    teacher_model.eval()
    with torch.no_grad():
        output_teacher_batch = teacher_model(data)

    return output_teacher_batch


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        return True
        #print('Models match perfectly! :)')
    else:
        return False


# python train_kd_zagoruyko.py --teacher logs/ORIGINAL_MODEL_MERGED_DATASET_TRAIN_501EPOCHS_LR_0_0001_AGAIN_SEED12/models/dense_net_160.pkl --expname distillation_zagoruyko_test

# python train_kd_zagoruyko.py --teacher logs/ORIGINAL_MODEL_MERGED_DATASET_TRAIN_501EPOCHS_LR_0_0001_AGAIN_SEED12/models/dense_net_160.pkl --expname distillation_zagoruyko_hinton_beta0_1_alpha_distill0_1_T8_without_out_layer



def main():
    global args
    args = parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)

    kwargs = vars(args)

    print(args.useGPU)
    global device
    if args.useGPU == 'True' or args.useGPU == 'true':
        print('USE GPU')
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        torch.cuda.manual_seed(12)
    else:
        print('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(12)

    torch.backends.cudnn.deterministic = True

    LOGDIR = 'logs/{}'.format(args.expname)
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(LOGDIR + '/models', exist_ok=True)
    os.makedirs(LOGDIR + '/optimizers', exist_ok=True)
    global logger
    logger = Logger(os.path.join(LOGDIR, 'logs.log'))

    # load teacher model
    teacher_state_dict = torch.load(args.teacher)
    teacher_model = model_dict['teacher']
    logger.write('using teacher model: ' + str(type(teacher_model)))
    teacher_model = teacher_model.to(device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model_copy = copy.deepcopy(teacher_model)
    teacher_model.train()
    # cancel gradients for teacher model, because we don't want to update this network
    for params in teacher_model.parameters():
        params.requires_grad = False

    # TODO probaj tako, da vzames naucenega studenta in ga naucis bolj preko teacher modela - nov nacin ucenja!
    student_model = model_dict['student']
    logger.write('using student model: ' + str(type(student_model)))
    student_model = student_model.to(device)

    # print('student model eval()')
    # student_model.eval()
    #from torchsummary import summary
    #summary(student_model, input_size=(1, 640, 400))  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    # print(student_model)

    student_model.train()
    nparams = get_nparams(student_model)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    if args.resume != '':
        optimizer_state_dict_path = args.resume.replace('models', 'optimizers')
        print("EXISTING OPTIMIZER DICT from: {}".format(optimizer_state_dict_path))
        optimizer_state_dict = torch.load(optimizer_state_dict_path)
        optimizer.load_state_dict(optimizer_state_dict)
        if args.useGPU:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

    global criterion
    criterion = CrossEntropyLoss2d()
    global criterion_DICE
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    global criterion_SL
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
    train = IrisDataset(filepath=Path2file, split='train',
                        transform=transform, **kwargs)
    print('len: ' + str(train.__len__()))
    valid = IrisDataset(filepath=Path2file, split='validation',
                        transform=transform, **kwargs)

    trainloader = DataLoader(train, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True)

    validloader = DataLoader(valid, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True)

    test = IrisDataset(filepath=Path2file, split='test',
                       transform=transform, **kwargs)

    testloader = DataLoader(test, batch_size=args.bs,
                            shuffle=False, num_workers=args.workers)

    alpha = np.zeros(((args.epochs)))
    alpha[0:np.min([125, args.epochs])] = 1 - np.arange(1, np.min([125, args.epochs]) + 1) / np.min([125, args.epochs])
    if args.epochs > 125:
        alpha[125:] = 1

    # PARAMETERS:
    beta = 0.2
    alpha_distillation = 0.0
    T = 8
    #global only_poor
    #only_poor = False

    logger.write(str(opt))
    logger.write('beta: {}, alpha_distillation: {}, T: {}'.format(beta, alpha_distillation, T))
    #logger.write('beta==0.0 means that it is calculated on the fly as stated in zagoruyko paper'.format())

    prev_prev_val_loss = 10000
    prev_val_loss = 10000
    for epoch in range(args.epochs):
        #print("Emptying cache..")
        #torch.cuda.empty_cache()

        models_equal = compare_models(teacher_model_copy, teacher_model)
        if not models_equal:
            print('Second comparison is false!')

        # 1. get outputs from teacher model (grad=false). Ali bo šlo vse v RAM preden delam epohe, ali moram za vsak batch na novo zalaufat tole?

        # 2. get outputs from student model (train)
        conf_matrix_whole = np.zeros((2,2))
        training_loss_sum = 0.0
        training_original_loss_sum = 0.0
        training_hinton_loss_sum = 0.0
        training_attention_loss_sum = 0.0
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()
            optimizer.zero_grad()
            student_batch_outputs = student_model(data)
            # get teacher batch output
            teacher_batch_outputs = fetch_teacher_outputs(teacher_model, data)

            loss, loss_attention_float, loss_hinton_float, loss_original_float = calculate_loss_for_batch(student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target,
                                            index, spatialWeights, maxDist, alpha, epoch, alpha_distillation, T, beta)

            # 4. backprop only on student model
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            training_loss_sum += loss.item()
            training_attention_loss_sum += loss_attention_float
            training_hinton_loss_sum += loss_hinton_float
            training_original_loss_sum += loss_original_float

            predictions = get_predictions(student_batch_outputs)

            conf_matrix_batch = mIoU_conf_matrix(predictions, target)
            conf_matrix_whole += conf_matrix_batch

            if i % 10 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), loss.item()))

        # visualize total training loss
        try:
            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_loss_sum]),
                win=win_loss,
                name='training loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_original_loss_sum]),
                win=win_loss,
                name='training original loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_hinton_loss_sum]),
                win=win_loss,
                name='training hinton loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_attention_loss_sum]),
                win=win_loss,
                name='training attention loss',
                update='append'
            )

        except Exception:
            pass

        miou_train = conf_matrix_to_iou(conf_matrix_whole)
        logger.write('Epoch:{}, Train mIoU: {}'.format(epoch, miou_train))

        # *************************************************************************************************************
        lossvalid, miou, validation_loss_sum, validation_attention_loss_sum, validation_hinton_loss_sum, validation_original_loss_sum = validation_loss(
            validloader, student_model, teacher_model, alpha, epoch, T, alpha_distillation, beta)
        totalperf = total_metric(nparams, miou)
        f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'
        logger.write(f.format(epoch, lossvalid, miou, nparams, totalperf))

        # visualize validation loss
        try:
            viz.line(
                X=np.array([epoch]),
                Y=np.array([validation_loss_sum]),
                win=win_loss,
                name='validation loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([validation_original_loss_sum]),
                win=win_loss,
                name='validation original loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([validation_hinton_loss_sum]),
                win=win_loss,
                name='validation hinton loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([validation_attention_loss_sum]),
                win=win_loss,
                name='validation attention loss',
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

        ##save the model every epoch
        if epoch % 5 == 0:
            torch.save(student_model.state_dict(), '{}/models/dense_net_{}.pkl'.format(LOGDIR, epoch))
            torch.save(optimizer.state_dict(), '{}/optimizers/dense_net_{}.pkl'.format(LOGDIR, epoch))

        # every epoch calculate test IoU
        conf_matrix_whole = np.zeros((2,2))
        with torch.no_grad():
            for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
                img, label_tensor, index, x, maxDist = batchdata
                data = img.to(device)
                target = label_tensor.to(device).long()
                output = student_model(data)
                predict = get_predictions(output)

                conf_matrix_batch = mIoU_conf_matrix(predict, target)
                conf_matrix_whole += conf_matrix_batch

                teacher_batch_outputs = fetch_teacher_outputs(teacher_model, data)
                teacher_predictions = get_predictions(teacher_batch_outputs)

                # VISUALIZE ONLY EVERY 10 EPOCHS
                if epoch % 10 == 0:
                    os.makedirs('test/epoch/mask_{}/'.format(args.expname), exist_ok=True)
                    os.makedirs('test/epoch/mask_{}/epoch_{}'.format(args.expname, epoch), exist_ok=True)
                    for j in range(len(index)):
                        pred_img = predict[j].cpu().numpy()  # /3.0
                        pred_img_teacher = teacher_predictions[j].cpu().numpy()  # /3.0
                        pred_img[pred_img > 1] = 0
                        pred_img_teacher[pred_img_teacher > 1] = 0
                        inp = img[j].squeeze() * 0.5 + 0.5
                        img_orig = np.clip(inp, 0, 1)
                        img_orig = np.array(img_orig)
                        label = label_tensor[j].view(args.height, args.width)
                        label = np.array(label)

                        combine = np.hstack([img_orig, pred_img, pred_img_teacher, label])
                        plt.imsave('test/epoch/mask_{}/epoch_{}/{}.jpg'.format(args.expname, epoch, index[j]), combine)

            miou = conf_matrix_to_iou(conf_matrix_whole)

            #if miou > 0.82:
            #    only_poor = True

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

        # stopping criteria
        if prev_prev_val_loss < prev_val_loss and prev_val_loss < validation_loss_sum:
            print('validation loss increased two times in a row')
            break

        # save validation losses
        prev_prev_val_loss = prev_prev_val_loss
        prev_val_loss = validation_loss_sum


if __name__ == '__main__':
    main()
