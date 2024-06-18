#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import operator

from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import IrisDataset, transform
import torch
import torchvision.utils as tvutils

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
from train_with_pruning_combined import initialize_globals as initialize_globals_train_with_pruning
from train_with_pruning_combined import calculate_loss_for_batch_CE


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


def conf_matrix_to_mIoU(args, confusion_matrix):
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

    # print(confusion_matrix)
    n_classes = 4 if 'sip' in args.dataset.lower() else 2
    if confusion_matrix.shape != (n_classes, n_classes):
        print(confusion_matrix.shape)
        raise NotImplementedError()

    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))

    logger.write("per-class mIOU: " + str(MIoU))

    if n_classes == 2:
        return MIoU.item(1) # only IoU for sclera (not background)
    else:
        return np.mean(MIoU)


def validation_loss(args, device, device_teacher, loader, student_model, teacher_model, alpha, epoch, parameters, criterion,
                    criterion_DICE, criterion_SL):
    epoch_loss = []
    n_classes = 4 if 'sip' in args.dataset.lower() else 2
    conf_matrix_whole = np.zeros((n_classes, n_classes))
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()

            student_batch_outputs = student_model(data)
            if device_teacher is not None:
                data_device_teacher = data.to(device_teacher)
            else:
                data_device_teacher = data
            teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data_device_teacher)

            # loss, loss_hinton_float = calculate_loss_for_batch_original_with_kldiv(student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights, maxDist, alpha, epoch, alpha_distillation, alpha_original, T)
            loss, _, _, _ = calculate_loss_for_batch_CE(args, device, device_teacher, student_model, student_batch_outputs,
                                                        teacher_model, teacher_batch_outputs, target, index,
                                                        spatialWeights,
                                                        maxDist, alpha, epoch, parameters, criterion, criterion_DICE,
                                                        criterion_SL)

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
        # print('Models match perfectly! :)')
    else:
        return False




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
    set_model_to_train(teacher_model)  # it is already in train
    # cancel gradients for teacher model, because we don't want to update this network
    for params in teacher_model.parameters():
        params.requires_grad = False

    return teacher_model


def validate(args, device, device_teacher, student_model, validloader, teacher_model, viz, win_loss, win_iou, alpha, epoch, parameters,
             criterion, criterion_DICE, criterion_SL):
    lossvalid, miou, validation_loss_sum = validation_loss(args, device, device_teacher, validloader, student_model, teacher_model,
                                                           alpha, epoch,
                                                           parameters, criterion, criterion_DICE, criterion_SL)
    # totalperf = total_metric(nparams, miou)
    s = 'Epoch: {}, Valid Loss: {:.3f} mIoU: {}'.format(epoch, lossvalid, miou)
    logger.write(s)
    short_logger.write_silent(s)

    # visualize validation loss
    plot_with_visdom(viz, win_loss, epoch, validation_loss_sum, 'validation loss')
    # visualize validation iou
    plot_with_visdom(viz, win_iou, epoch, miou, 'validation IoU')

    return lossvalid, validation_loss_sum  # nedded for scheduler, needed for stopping criteria


def evaluate_on_test_during_train(args, device, device_teacher, student_model, testloader, teacher_model, epoch, viz, win_iou):
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
            if args.visualize and epoch % 10 == 0:
                visualize_on_test_images(args, epoch, device_teacher, filename, teacher_model, img, data, label_tensor, student_predictions)

        if args.visualize and epoch % 10 == 0:
            visualize_filters_and_activation_maps(args, epoch, student_model)
        miou = conf_matrix_to_mIoU(args, conf_matrix_whole)

        # visualize test iou
        plot_with_visdom(viz, win_iou, epoch, miou, 'test iou')

        s = 'Epoch: {}, Test mIoU: {}'.format(epoch, miou)
        logger.write(s)
        short_logger.write_silent(s)


def visualize_on_test_images(args, epoch, device_teacher, filename, teacher_model, img, data, label_tensor, student_predictions):
    if teacher_model is not None:
        if device_teacher is not None:
            data_device_teacher = data.to(device_teacher)
        else:
            data_device_teacher = data
        teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data_device_teacher)
        teacher_predictions = get_predictions(teacher_batch_outputs)

        os.makedirs('test/mask/{}/'.format(args.expname), exist_ok=True)
        os.makedirs('test/mask/{}/epoch_{}'.format(args.expname, epoch), exist_ok=True)
        for j in range(len(filename)):
            pred_img = student_predictions[j].cpu().numpy()  # /3.0
            pred_img_teacher = teacher_predictions[j].cpu().numpy()  # /3.0
            # pred_img[pred_img > 1] = 0 # TODO odstrani, da se bojo vidle predikcije cseh 4 razredov
            # pred_img_teacher[pred_img_teacher > 1] = 0
            # print(np.unique(pred_img))
            # print(np.unique(pred_img_teacher))

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
            # pred_img[pred_img > 1] = 0
            # print(np.unique(pred_img))
            inp = img[j].squeeze() * 0.5 + 0.5
            img_orig = np.clip(inp, 0, 1)
            img_orig = np.array(img_orig)
            label = label_tensor[j].view(args.height, args.width)
            label = np.array(label)

            combine = np.hstack([img_orig, pred_img, label])
            plt.imsave('test/mask/{}/epoch_{}/{}.jpg'.format(args.expname, epoch, filename[j]),
                       combine)


def visualize_filters_and_activation_maps(args, epoch, model):
    os.makedirs(f'test/filters/{args.expname}/epoch_{epoch}', exist_ok=True)
    os.makedirs(f'test/activations/{args.expname}/epoch_{epoch}', exist_ok=True)

    conv_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    for name, module in tqdm(conv_layers, "Visualizing filters and activation maps"):
        if not isinstance(module, nn.Conv2d):
            continue

        # Filter visualizations
        filters = module.weight.detach().cpu()
        filters -= filters.min()
        filters /= filters.max()
        idx = []  # Ignore pruned filters
        zero_tensor = torch.zeros(filters.shape[1:])
        for i, f in enumerate(filters):
            if (f == zero_tensor).all():
                continue
            idx.append(i)
            if len(idx) < 5:  # Visualize first 5 (not pruned) filters
                if len(f) == 3:  # If filter with 3 channels, save it as RGB
                    tvutils.save_image(f, f'test/filters/{args.expname}/epoch_{epoch}/{name}-{i}.png')
                else:  # Otherwise save each channel separately as greyscale
                    for j, c in enumerate(f[:5]):  # But at most 5 channels
                        tvutils.save_image(c, f'test/filters/{args.expname}/epoch_{epoch}/{name}-{i}-{j}.png')
        if filters.shape[1] == 3:
            nrows = int(np.sqrt(len(idx)))
            tvutils.save_image(filters[idx].reshape(len(idx), 3, *filters.shape[2:]), f'test/filters/{args.expname}/epoch_{epoch}/{name}-combined.png', nrow=nrows)
        else:
            n = len(idx) * filters.shape[1]
            nrows = int(np.sqrt(n))
            tvutils.save_image(filters[idx].reshape(n, 1, *filters.shape[2:]), f'test/filters/{args.expname}/epoch_{epoch}/{name}-combined.png', nrow=nrows)

        # Activation map visualizations
        try:
            block_name, _, layer_name = name.rpartition('.')
            block = operator.attrgetter(block_name)(model)
            features = getattr(block, f'{layer_name}_activations').detach().cpu()
        except AttributeError:
            continue
        features = features.mean(dim=0)[idx]
        features -= features.min()
        features /= features.max()
        f_avg = features.mean(dim=0)
        tvutils.save_image(f_avg, f'test/activations/{args.expname}/epoch_{epoch}/{name}-avg.jpg')
        for i, f in enumerate(features[:10]):  # Save first 10 activations
            tvutils.save_image(f, f'test/activations/{args.expname}/epoch_{epoch}/{name}-{i}.jpg')
            tvutils.save_image(torch.abs(f - f_avg), f'test/activations/{args.expname}/epoch_{epoch}/{name}-{i}-diff.jpg')
        width, height = features.shape[1:]
        nrows = int(np.sqrt(len(idx) * height / width))
        tvutils.save_image(features.unsqueeze(1), f'test/activations/{args.expname}/epoch_{epoch}/{name}-combined.jpg', nrow=nrows)
        tvutils.save_image(torch.abs(features - f_avg).unsqueeze(1), f'test/activations/{args.expname}/epoch_{epoch}/{name}-combined-diff.jpg', nrow=nrows)


def plot_with_visdom(viz, win_loss, epoch, value, description):
    try:
        viz.line(
            X=np.array([epoch]),
            Y=np.array([value]),
            win=win_loss,
            name=description,
            update='append'
        )
    except Exception:
        pass


def train(args, device, teacher_model, device_teacher, trainloader, optimizer, student_model, validloader, testloader, viz, win_loss,
          win_iou, parameters, scheduler, criterion, criterion_DICE, criterion_SL): #, all_zeroed_filters_copy):
    alpha = parameters['alpha']

    set_model_to_train(student_model)
    set_model_to_eval(teacher_model)
    # nparams = get_nparams(student_model)
    teacher_model_copy = copy.deepcopy(teacher_model)

    best_val_loss_so_far = float('inf')
    epochs_since_best_loss = 0
    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):
        set_model_to_train(student_model)
        assert student_model.training, 'Student must be in training mode'
        if teacher_model is not None:
            assert not teacher_model.training, 'Teacher must be in eval mode'

        if teacher_model is not None:
            models_equal = compare_models(teacher_model_copy, teacher_model)
            if not models_equal:
                logger.write('Second comparison is false!')

        #all_zeroed_filters, all_used_filters = count_zeroed_filters_for_model(student_model, device)
        #learnable_parameters, all_parameters = count_number_of_learnable_parameters(student_model, device)
        #logger.write('zeroed filters: {0}'.format(all_zeroed_filters))
        #logger.write('n zeroed filters: {0}'.format(len(all_zeroed_filters)))
        #logger.write('n used filters: {0}'.format(len(all_used_filters)))
        #logger.write('number of parameters: {0}/{1}'.format(learnable_parameters, all_parameters))
        #assert all_zeroed_filters == all_zeroed_filters_copy

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
            if device_teacher is not None:
                data_device_teacher = data.to(device_teacher)
            else:
                data_device_teacher = data
            teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data_device_teacher)
            loss, loss_hinton_float, loss_attention_float, loss_fsp_float = calculate_loss_for_batch_CE(args, device, device_teacher,
                                                                                                        student_model,
                                                                                                        student_batch_outputs,
                                                                                                        teacher_model,
                                                                                                        teacher_batch_outputs,
                                                                                                        target, index,
                                                                                                        spatialWeights,
                                                                                                        maxDist, alpha,
                                                                                                        epoch,
                                                                                                        parameters,
                                                                                                        criterion,
                                                                                                        criterion_DICE,
                                                                                                        criterion_SL)

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
                logger.write('Epoch: {} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), loss.item()))

        # visualize total training loss
        plot_with_visdom(viz, win_loss, epoch, training_loss_sum, 'training loss')
        plot_with_visdom(viz, win_loss, epoch, training_hinton_loss_sum, 'training hinton loss')
        plot_with_visdom(viz, win_loss, epoch, training_attention_loss_sum, 'training attention loss')
        plot_with_visdom(viz, win_loss, epoch, training_fsp_loss_sum, 'training fsp loss')

        miou_train = conf_matrix_to_mIoU(args, conf_matrix_whole)

        s = 'Epoch: {}, Train mIoU: {}'.format(epoch, miou_train)
        print('Experiment: {}'.format(args.expname))
        logger.write(s)
        short_logger.write_silent(s)

        # *********VALIDATION*******************************************************************************************

        set_model_to_eval(student_model)  # for validation loss and also test afterwards
        set_model_to_eval(teacher_model)  # should already be in eval mode
        assert not student_model.training, 'Student must be in eval mode'
        if teacher_model is not None:
            assert not teacher_model.training, 'Teacher must be in eval mode'

        lossvalid, validation_loss_sum = validate(args, device, device_teacher, student_model, validloader, teacher_model, viz,
                                                  win_loss, win_iou, alpha, epoch, parameters, criterion,
                                                  criterion_DICE, criterion_SL)
        scheduler.step(lossvalid)

        # every epoch calculate test IoU
        evaluate_on_test_during_train(args, device, device_teacher, student_model, testloader, teacher_model, epoch, viz, win_iou)

        # save last checkpoint
        torch.save(student_model.state_dict(), LOGDIR/'models'/f'{args.model}_final.pkl')

        # save best model as full
        if validation_loss_sum < best_val_loss_so_far:
            logger.write(f"Saving new best model to {args.model}_best.pt")
            torch.save(student_model, LOGDIR/'models'/f'{args.model}_best.pt')
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
            logger.write("Validation loss hasn't improved for 10 epochs, stopping training")
            break

        epochs_since_best_loss += 1


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
                                transform=transform, n_classes=n_classes, **kwargs)
    valid = IrisDataset(filepath=Path2file, split='validation',
                        transform=transform, n_classes=n_classes, **kwargs)

    trainloader = DataLoader(train_dataset, batch_size=args.bs,
                             shuffle=True, num_workers=args.workers, drop_last=True)

    validloader = DataLoader(valid, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True)

    test = IrisDataset(filepath=Path2file, split='test',
                       transform=transform, n_classes=n_classes, **kwargs)

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
    global short_logger
    short_logger = Logger(LOGDIR/'short.log')


def load_student(args, device):
    model_name = args.resume #'logs/new_pruning_10filters_at_once_retrain_for_5epochs_distillation_attention_b0_001_resize_activations40_25_layer_limit75_block_limit75/models/ckpt_0.pt'

#    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
#    del checkpoint  # dereference seems crucial
#    torch.cuda.empty_cache()


    model = torch.load(model_name)
    model = model.to(device)
    return model


def main():
    args = parse_args()

    initialize_globals_train_with_pruning(args)
    initialize_globals(args)
    (LOGDIR/'models').mkdir(parents=True, exist_ok=True)
    (LOGDIR/'optimizers').mkdir(parents=True, exist_ok=True)
    (LOGDIR/'schedulers').mkdir(parents=True, exist_ok=True)


    used_gpus_list = args.pruningGpus#.split(',')
    if args.useGPU == 'True' or args.useGPU == 'true':
        logger.write('USE GPUs: {0}'.format(used_gpus_list))
        device = torch.device("cuda")
        # os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in used_gpus_list)  # This only works if it's set before torch is imported (or not?)
        # device = torch.device(f"cuda:{used_gpus_list[0]}")
        torch.cuda.manual_seed(7)
    else:
        logger.write('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(7)

    torch.backends.cudnn.deterministic = True


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
        'alpha_distillation': 0.0000,
        'T': 8,  # TODO 2, 4, 8, 16
        # ---------------ATTENTION------------------
        'beta': 0.00,
        # -------------FSP------------------
        'lambda': 0.000000
    }

    device_teacher = None
    teacher_model = None
    if (parameters['alpha_distillation'] != 0 or parameters['beta'] != 0 or parameters['lambda'] != 0) and len(
            used_gpus_list) != 2:
        print(used_gpus_list)
        raise Exception('Distillation requires two GPUs')

    if len(used_gpus_list) == 2:  # if distillation is used and we (must) have 2 gpus
        device = torch.device("cuda:{0}".format(used_gpus_list[0]))
        device_teacher = torch.device("cuda:{0}".format(used_gpus_list[1]))
        #args.teacher = 'logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl'
        logger.write('using teacher: {0}'.format(args.teacher))
        teacher_model = load_teacher(args, device_teacher)


    student_model = load_student(args, device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    """
    all_zeroed_filters_copy, all_used_filters_copy = count_zeroed_filters_for_model(student_model, device)
    logger.write('removed filters: {0}'.format(all_zeroed_filters_copy))
    logger.write('using model WITHOUT {0} filters'.format(len(all_zeroed_filters_copy)))

    # initialize weights for loaded model
    print('init weights..')
    student_model._initialize_weights()
    for filter in all_zeroed_filters_copy:
        # register hooks
        _ = disable_filter(device, student_model, filter)  # disable this filter - zero filter's weights

        layer_name, index = get_parameter_name_and_index_from_activations_dict_key(filter)
        block_name, layer_name_without_block = layer_name.split('.')
        block = getattr(student_model, block_name)
        model_layer = getattr(block, layer_name_without_block)
        model_layer_weight = getattr(model_layer, 'weight')
        model_layer_weight.register_hook(outer_hook(device, index))  # also zero gradients of this filter
    """
    from torchsummary import summary
    summary(student_model, input_size=(1, args.height, args.width))  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    #print(student_model)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True, n_classes=4 if 'sip' in args.dataset.lower() else 2)
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

    opt = vars(args)
    logger.write(str(opt))
    logger.write(str(parameters))

    trainloader, validloader, testloader = get_data_loaders(args)

    train(args, device, teacher_model, device_teacher, trainloader, optimizer, student_model, validloader, testloader, viz, win_loss,
          win_iou, parameters, scheduler, criterion, criterion_DICE, criterion_SL)


# destilacija z dvema gpujema:
# python train_pruned_model.py --expname train_with_attention_beta0_35_prunned_with_same_distillation_model_with_139k_parameters --resume logs/pruning_10filters_at_once_retrain_for_5epochs_distillation_attention_b0_35_resize_activations40_25_layer_limit70_block_limit70/models/model_without_700_filters.pkl --teacher logs/pruning_10filters_at_once_retrain_for_5epochs_distillation_attention_b0_35_resize_activations40_25_layer_limit70_block_limit70/models/model_without_0_filters.pkl  --pruningGpus 0 1



if __name__ == '__main__':
    main()


# python train_pruned_model.py --expname train_prunned_model_with_attention_beta0_001_prunned_with_same_distillation_model --resume logs/new_pruning_1Gflops_retrain_for_5epochs_pruneAway74_distillation_b0_001_resize_activations40_25_layer_limit75_block_limit75/models/ckpt_12.pt --teacher logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl  --pruningGpus 0 1


# python train_pruned_model.py --expname train_prunned_model_prunned_without_distillation --resume logs/new_pruning_1Gflops_retrain_for_5epochs_pruneAway74_resize_activations40_25_layer_limit75_block_limit75/models/ckpt_12.pt
