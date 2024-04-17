import copy

from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import IrisDataset, transform
import torch

from utils import CrossEntropyLoss2d, Logger, GeneralizedDiceLoss, SurfaceLoss, ResourceManager
import numpy as np
# from dataset import transform
from opt import parse_args
import os
import sys
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
from visdom import Visdom
from models import model_dict
from train_with_knowledge_distillation import initialize_globals as kd_script_initialize_globals
from train_with_knowledge_distillation import get_data_loaders, get_conf_matrix, \
    conf_matrix_to_mIoU, fetch_teacher_outputs, load_teacher, compare_models
from torchsummary import summary

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eyez.utils import EYEZ


# https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/4
def cross_entropy_with_soft_targets(device, student_prediction, teacher_prediction, T, size_average=True):
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

    if student_prediction.device.index != teacher_prediction.device.index:
        # get teacher data on student gpu
        teacher_prediction_device = teacher_prediction.to(device)
    else:
        teacher_prediction_device = teacher_prediction

    #print(student_prediction.shape)
    #print(teacher_prediction_device.shape)
    #print('--------------------')
    softmax = nn.Softmax(dim=1)
    target = softmax(teacher_prediction_device/T)
    #print(target.shape)
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(student_prediction/T), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(student_prediction/T), dim=1))


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

    return loss_original


def get_hinton_loss(args, device, student_model, teacher_model, indexes_with_lower_student_iou, student_batch_outputs, teacher_batch_outputs, parameters, device_teacher):
    if parameters['alpha_distillation'] == 0.0:
        return None

    assert len(indexes_with_lower_student_iou) == args.bs
    #if not indexes_with_lower_student_iou: # if empty, then return 0.0
    #    return None

    # ADD ALL INTERMEDIATE LAYERS
    #torch_indices_lower_student_iou = torch.tensor(indexes_with_lower_student_iou).to(device)
    #dim = torch.tensor(0).to(device)

    if len(indexes_with_lower_student_iou) > student_model.x1.shape[0]:
        logger.write('x1: len(indices) > x1 batch size')
    #s_x1 = torch.index_select(student_model.x1, dim, torch_indices_lower_student_iou)
    #t_x1 = torch.index_select(teacher_model.x1, dim, torch_indices_lower_student_iou)
    loss_ce_x1 = cross_entropy_with_soft_targets(device, student_model.x1, teacher_model.x1, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x3.shape[0]:
        logger.write('x3: len(indices) > x3 batch size')
    #s_x3 = torch.index_select(student_model.x3, dim, torch_indices_lower_student_iou)
    #t_x3 = torch.index_select(teacher_model.x3, dim, torch_indices_lower_student_iou)
    loss_ce_x3 = cross_entropy_with_soft_targets(device, student_model.x3, teacher_model.x3, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x5.shape[0]:
        logger.write('x5: len(indices) > x5 batch size')
    #s_x6 = torch.index_select(student_model.x6, dim, torch_indices_lower_student_iou)
    #t_x7 = torch.index_select(teacher_model.x7, dim, torch_indices_lower_student_iou)
    loss_ce_x7 = cross_entropy_with_soft_targets(device, student_model.x7, teacher_model.x7, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x7.shape[0]:
        logger.write('x7: len(indices) > x7 batch size')
    #s_x7 = torch.index_select(student_model.x7, dim, torch_indices_lower_student_iou)
    #t_x9 = torch.index_select(teacher_model.x9, dim, torch_indices_lower_student_iou)
    loss_ce_x9 = cross_entropy_with_soft_targets(device, student_model.x9, teacher_model.x9, parameters['T'])
    if len(indexes_with_lower_student_iou) > student_model.x9.shape[0]:
        logger.write('x9: len(indices) > x9 batch size')
    # add smallest layer:
    #s_x5 = torch.index_select(student_model.x5, dim, torch_indices_lower_student_iou)
    #t_x5 = torch.index_select(teacher_model.x5, dim, torch_indices_lower_student_iou)
    loss_ce_smallest_layer = cross_entropy_with_soft_targets(device, student_model.x5, teacher_model.x5, parameters['T'])

    #s_output = torch.index_select(student_batch_outputs, dim, torch_indices_lower_student_iou)
    #t_output = torch.index_select(teacher_batch_outputs, dim, torch_indices_lower_student_iou)
    loss_ce_output = cross_entropy_with_soft_targets(device, student_batch_outputs, teacher_batch_outputs, parameters['T'])
    loss_kd = loss_ce_smallest_layer + loss_ce_output + loss_ce_x1 + loss_ce_x3 + loss_ce_x7 + loss_ce_x9

    return loss_kd


def get_attention_loss(args, device, student_model, teacher_model, indexes_with_lower_student_iou, parameters):
    if parameters['beta'] == 0.0:
        return None

    assert len(indexes_with_lower_student_iou) == args.bs
    #if not indexes_with_lower_student_iou: # if empty, then return 0.0
    #    return None

    #torch_indices_lower_student_iou = torch.tensor(indexes_with_lower_student_iou).to(device)#.long()
    #dim = torch.tensor(0).to(device)

    # print(student_model.att_map_x1_sum.shape)  # torch.Size([4, 640, 400])

    # ATTENTION
    #s_att_map_x1 = torch.index_select(student_model.att_map_x1_sum, dim, torch_indices_lower_student_iou)
    #t_att_map_x1 = torch.index_select(teacher_model.att_map_x1_sum, dim, torch_indices_lower_student_iou)
    # TODO spremeni beta on the fly, beta=1000/(height*width*bs) -> w in h sta od attention mapa
    # zagoruyko paper, page 5, equation (2)

    if student_model.att_map_x1_sum.device.index != teacher_model.att_map_x1_sum.device.index:
        # models are on different gpus
        t_att_map_x1 = teacher_model.att_map_x1_sum.to(device)
        t_att_map_x3 = teacher_model.att_map_x3_sum.to(device)
        t_att_map_x5 = teacher_model.att_map_x5_sum.to(device)
        t_att_map_x7 = teacher_model.att_map_x7_sum.to(device)
        t_att_map_x9 = teacher_model.att_map_x9_sum.to(device)
    else:
        t_att_map_x1 = teacher_model.att_map_x1_sum
        t_att_map_x3 = teacher_model.att_map_x3_sum
        t_att_map_x5 = teacher_model.att_map_x5_sum
        t_att_map_x7 = teacher_model.att_map_x7_sum
        t_att_map_x9 = teacher_model.att_map_x9_sum

    loss_att_x1 = torch.norm(
        student_model.att_map_x1_sum / torch.norm(student_model.att_map_x1_sum, 2) - t_att_map_x1 / torch.norm(t_att_map_x1, 2), 2)

    #s_att_map_x3 = torch.index_select(student_model.att_map_x3_sum, dim, torch_indices_lower_student_iou)
    #t_att_map_x3 = torch.index_select(teacher_model.att_map_x3_sum, dim, torch_indices_lower_student_iou)
    loss_att_x3 = torch.norm(
        student_model.att_map_x3_sum / torch.norm(student_model.att_map_x3_sum, 2) - t_att_map_x3 / torch.norm(t_att_map_x3, 2), 2)

    #s_att_map_x5 = torch.index_select(student_model.att_map_x5_sum, dim, torch_indices_lower_student_iou)
    #t_att_map_x5 = torch.index_select(teacher_model.att_map_x5_sum, dim, torch_indices_lower_student_iou)
    loss_att_x5 = torch.norm(
        student_model.att_map_x5_sum / torch.norm(student_model.att_map_x5_sum, 2) - t_att_map_x5 / torch.norm(t_att_map_x5, 2), 2)

    #s_att_map_x6 = torch.index_select(student_model.att_map_x6_sum, dim, torch_indices_lower_student_iou)
    #t_att_map_x7 = torch.index_select(teacher_model.att_map_x7_sum, dim, torch_indices_lower_student_iou)
    loss_att_x6 = torch.norm(
        student_model.att_map_x7_sum / torch.norm(student_model.att_map_x7_sum, 2) - t_att_map_x7 / torch.norm(t_att_map_x7, 2), 2)

    #s_att_map_x7 = torch.index_select(student_model.att_map_x7_sum, dim, torch_indices_lower_student_iou)
    #t_att_map_x9 = torch.index_select(teacher_model.att_map_x9_sum, dim, torch_indices_lower_student_iou)
    loss_att_x7 = torch.norm(student_model.att_map_x9_sum / torch.norm(student_model.att_map_x9_sum, 2) - t_att_map_x9 / torch.norm(t_att_map_x9, 2), 2)

    beta_x1 = 1000 / (np.prod(list(t_att_map_x1.shape))) # 1000 / n_elements in list * batch_size
    beta_x3 = 1000 / (np.prod(list(t_att_map_x3.shape)))
    beta_x5 = 1000 / (np.prod(list(t_att_map_x5.shape)))
    beta_x6 = 1000 / (np.prod(list(t_att_map_x7.shape)))
    beta_x7 = 1000 / (np.prod(list(t_att_map_x9.shape)))
    loss_attention = beta_x1 * loss_att_x1 + beta_x3 * loss_att_x3 + beta_x5 * loss_att_x5 + beta_x6 * loss_att_x6 + beta_x7 * loss_att_x7

    return loss_attention

# TODO spremeni device
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


def calculate_loss_for_batch_CE(args, device, device_teacher, student_model, student_batch_outputs, teacher_model, teacher_batch_outputs, target, index, spatialWeights, maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL):
    # ORIGINAL LOSS
    loss_original = get_original_loss(device, student_batch_outputs, target, spatialWeights, maxDist, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL)

    indexes_with_lower_student_iou = [x for x in range(args.bs)]#get_indexes_with_lower_student_iou(args, student_batch_outputs, teacher_batch_outputs, target, epoch, index)

    # hinton loss
    loss_hinton = get_hinton_loss(args, device, student_model, teacher_model, indexes_with_lower_student_iou,
                                  student_batch_outputs, teacher_batch_outputs, parameters, device_teacher)

    # attention loss
    loss_attention = get_attention_loss(args, device, student_model, teacher_model, indexes_with_lower_student_iou, parameters)

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





def eval_model_and_populate_activations(args, device, loader, model):
    model.eval()
    logger.write('reset conv activations sum')
    model.reset_conv_activations_sum(device)
    conf_matrix_whole = np.zeros((args.numberOfClasses, args.numberOfClasses))
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(loader), total=len(loader)):
            img, label_tensor, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = label_tensor.to(device).long()
            output = model(data)
            predictions = get_predictions(output)
            conf_matrix_batch = get_conf_matrix(args, predictions, target)
            conf_matrix_whole += conf_matrix_batch

    mIoU = conf_matrix_to_mIoU(args, conf_matrix_whole)
    return mIoU

        # visualize test iou
        #plot_with_visdom(viz, win_iou, epoch, miou, 'test iou')
        #try:
        #    viz.line(
        #        X=np.array([epoch]),
        #        Y=np.array([loss_sum]),
        #        win=win_loss,
        #        name=description,
        #        update='append'
        #    )
        #except Exception:
        #    pass



def _get_sorted_filter_activations_dict_for_model(model):
    # ACTIVATIONS MUST BE CALCULATED BEFORE RUNNING THIS METHOD
    # AFTER DISABLING FILTER, THIS METHOD HAS OLD ACTIVATIONS - THEY NEED TO BE POPULATED AGAIN
    all_activations_sum_dict = {}
    for name, param in model.named_parameters():  # always weight and bias
        if name.endswith('_activations_sum'):
            block_name, attribute_name = name.split('.')
            block = getattr(model, block_name)
            curr_activations_sum_tensor = getattr(block, attribute_name)
            for index, activation_sum in enumerate(curr_activations_sum_tensor):
                name_without_activations_sum_suffix = name.replace('_activations_sum', '')
                dict_key = '{0}-{1}'.format(name_without_activations_sum_suffix, index) # conv1-15
                #print('adding key {0} to dict'.format(dict_key))
                all_activations_sum_dict[dict_key] = activation_sum  # cuda tensor

    all_activations_sum_dict_sorted = {k: v for k, v in
                                       sorted(all_activations_sum_dict.items(), key=lambda item: item[1])}
    return all_activations_sum_dict_sorted


def get_mIoU_on_train_and_populate_activations(args, device, trainloader, model):
    logger.write('evalvating on train set (populate activations)....')
    train_mIoU = eval_model_and_populate_activations(args, device, trainloader, model)
    # dict for saving all activations_sums
    # here model should have activations calculated
    #all_activations_sum_dict_sorted = _get_sorted_filter_activations_dict_for_model(model)
    return train_mIoU#, all_activations_sum_dict_sorted


def get_filter_with_minimal_activation(device, all_activations_sum_dict_sorted, layers_with_exceeded_limit, blocks_with_exceeded_limit):
    # we must return filter with nonzero activation, because zeroed filters are not used in network
    # ALSO set selected filter's activation to zero!
    # and skip layers in layers_with_exceeded_limit or blocks in blocks_with_exceeded_limit
    for key, value in all_activations_sum_dict_sorted.items():
        zero_tensor = torch.tensor(0, dtype=torch.float32).to(device)
        if torch.equal(value, zero_tensor):
            continue

        layer_name, _ = get_parameter_name_and_index_from_activations_dict_key(key)

        if layer_name in layers_with_exceeded_limit:
            #logger.write('layer {0} skipped because it has exceeded percent limit'.format(layer_name))
            continue
        block_name, _ = layer_name.split('.')
        if block_name in blocks_with_exceeded_limit:
            #logger.write('layer {0} skipped because block {1} has exceeded percent limit'.format(layer_name, block_name))
            continue

        all_activations_sum_dict_sorted[key] = zero_tensor
        return key, value#, all_activations_sum_dict_sorted

    print('tukaj sem..')


def disable_filter(device, model, name_index):
    #logger.write('disabling filter in layer {0}'.format(name_index))
    n_parameters_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    name, index = get_parameter_name_and_index_from_activations_dict_key(name_index)
    block_name, layer_name = name.split('.')
    block = getattr(model, block_name)
    layer = getattr(block, layer_name)
    #todo: layer = conv2d....

    new_conv = \
        torch.nn.Conv2d(in_channels=layer.in_channels, \
                        out_channels=layer.out_channels - 1,
                        kernel_size=layer.kernel_size, \
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=1,  # conv.groups,
                        bias=True
                        )

    if (layer.groups != 1):
        print('MAYBE THIS IS WRONG GROUPS != 1')
    layer.out_channels -= 1

    old_weights = layer.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()
    new_weights[: index, :, :, :] = old_weights[: index, :, :, :]
    new_weights[index:, :, :, :] = old_weights[index + 1:, :, :, :]

    # conv.weight.data = torch.from_numpy(new_weights).to(self.device)
    layer.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(device))
    layer.weight.grad = None

    if not layer.bias is None:
        bias_numpy = layer.bias.data.cpu().numpy()
        bias = np.zeros(shape=(bias_numpy.shape[0] - 1), dtype=np.float32)
        bias[:index] = bias_numpy[:index]
        bias[index:] = bias_numpy[index + 1:]
        # conv.bias.data = torch.from_numpy(bias).to(self.device)
        layer.bias = torch.nn.Parameter(torch.from_numpy(bias).to(device))
        layer.bias.grad = None


    # ALSO: change activations sum for this conv layer # todo: i dont update activations (only sum)
    layer_activations_sum = getattr(block, layer_name + '_activations_sum') # vektor dolzine toliko kolikor je filtrov, za vsak filter je ena stevilka
    layer_activations_sum = torch.cat([layer_activations_sum[0:index], layer_activations_sum[index+1:]])
    setattr(block, layer_name + '_activations_sum', torch.nn.Parameter(layer_activations_sum.to(device), requires_grad=False))

    layer_index = _get_layer_index(name)
    # prune next bn if nedded
    _prune_next_bn_if_needed(layer_index, index, index, 1, device, model)

    # surgery on chained convolution layers
    next_conv_idx_list = _get_next_conv_id_list_recursive(layer_index)
    for next_conv_id in next_conv_idx_list:
        #print(next_conv_id)
        _prune_next_layer(next_conv_id, index, index, 1, device, model)

    n_parameters_after_pruning = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return n_parameters_before - n_parameters_after_pruning


def _get_layer_index(name):
    if name == 'down_block1.conv1':
        return 0
    elif name == 'down_block1.conv21':
        return 1
    elif name == 'down_block1.conv22':
        return 2
    elif name == 'down_block1.conv31':
        return 3
    elif name == 'down_block1.conv32':
        return 4
    elif name == 'down_block2.conv1':
        return 5
    elif name == 'down_block2.conv21':
        return 6
    elif name == 'down_block2.conv22':
        return 7
    elif name == 'down_block2.conv31':
        return 8
    elif name == 'down_block2.conv32':
        return 9
    elif name == 'down_block3.conv1':
        return 10
    elif name == 'down_block3.conv21':
        return 11
    elif name == 'down_block3.conv22':
        return 12
    elif name == 'down_block3.conv31':
        return 13
    elif name == 'down_block3.conv32':
        return 14
    elif name == 'down_block4.conv1':
        return 15
    elif name == 'down_block4.conv21':
        return 16
    elif name == 'down_block4.conv22':
        return 17
    elif name == 'down_block4.conv31':
        return 18
    elif name == 'down_block4.conv32':
        return 19
    elif name == 'down_block5.conv1':
        return 20
    elif name == 'down_block5.conv21':
        return 21
    elif name == 'down_block5.conv22':
        return 22
    elif name == 'down_block5.conv31':
        return 23
    elif name == 'down_block5.conv32':
        return 24

    elif name == 'up_block1.conv11':
        return 25
    elif name == 'up_block1.conv12':
        return 26
    elif name == 'up_block1.conv21':
        return 27
    elif name == 'up_block1.conv22':
        return 28
    elif name == 'up_block2.conv11':
        return 29
    elif name == 'up_block2.conv12':
        return 30
    elif name == 'up_block2.conv21':
        return 31
    elif name == 'up_block2.conv22':
        return 32
    elif name == 'up_block3.conv11':
        return 33
    elif name == 'up_block3.conv12':
        return 34
    elif name == 'up_block3.conv21':
        return 35
    elif name == 'up_block3.conv22':
        return 36
    elif name == 'up_block4.conv11':
        return 37
    elif name == 'up_block4.conv12':
        return 38
    elif name == 'up_block4.conv21':
        return 39
    elif name == 'up_block4.conv22':
        return 40
    elif name.startswith('out_conv1'):
        return 41
    else:
        raise Exception('Neki je narobe pri layer index')

def _get_next_conv_id_list_recursive(layer_index):
    if layer_index == 0:
        next_conv_idx = [1, 3]
    elif layer_index == 1:
        next_conv_idx = [2]
    elif layer_index == 2:
        next_conv_idx = [3]
    elif layer_index == 3:
        next_conv_idx = [4]
    elif layer_index == 4:
        next_conv_idx = [5, 6, 8, 37, 39]
    elif layer_index == 5:
        next_conv_idx = [6, 8]
    elif layer_index == 6:
        next_conv_idx = [7]
    elif layer_index == 7:
        next_conv_idx = [8]
    elif layer_index == 8:
        next_conv_idx = [9]
    elif layer_index == 9:
        next_conv_idx = [10, 11, 13, 33, 35]
    elif layer_index == 10:
        next_conv_idx = [11, 13]
    elif layer_index == 11:
        next_conv_idx = [12]
    elif layer_index == 12:
        next_conv_idx = [13]
    elif layer_index == 13:
        next_conv_idx = [14]
    elif layer_index == 14:
        next_conv_idx = [15, 16, 18, 29, 31]
    elif layer_index == 15:
        next_conv_idx = [16, 18]
    elif layer_index == 16:
        next_conv_idx = [17]
    elif layer_index == 17:
        next_conv_idx = [18]
    elif layer_index == 18:
        next_conv_idx = [19]
    elif layer_index == 19:
        next_conv_idx = [20, 21, 23, 25, 27]
    elif layer_index == 20:
        next_conv_idx = [21, 23]
    elif layer_index == 21:
        next_conv_idx = [22]
    elif layer_index == 22:
        next_conv_idx = [23]
    elif layer_index == 23:
        next_conv_idx = [24]
    elif layer_index == 24:
        next_conv_idx = [25, 27]
    # UP BLOCKS:
    elif layer_index == 25:
        next_conv_idx = [26]
    elif layer_index == 26:
        next_conv_idx = [27]
    elif layer_index == 27:
        next_conv_idx = [28]
    elif layer_index == 28:
        next_conv_idx = [29, 31]
    elif layer_index == 29:
        next_conv_idx = [30]
    elif layer_index == 30:
        next_conv_idx = [31]
    elif layer_index == 31:
        next_conv_idx = [32]
    elif layer_index == 32:
        next_conv_idx = [33, 35]
    elif layer_index == 33:
        next_conv_idx = [34]
    elif layer_index == 34:
        next_conv_idx = [35]
    elif layer_index == 35:
        next_conv_idx = [36]
    elif layer_index == 36:
        next_conv_idx = [37, 39]
    elif layer_index == 37:
        next_conv_idx = [38]
    elif layer_index == 38:
        next_conv_idx = [39]
    elif layer_index == 39:
        next_conv_idx = [40]
    elif layer_index == 40:
        next_conv_idx = [41]
    elif layer_index == 41:
        next_conv_idx = []
    else:
        raise Exception("Error occured")

    # recursion call
    result_list = next_conv_idx.copy()
    # for id in next_conv_idx:
    #    result_list = result_list + self._get_next_conv_id_list_recursive(id)

    return result_list


def _get_bn_by_prev_conv_index(conv_index, model):
    # if conv index is for conv that is passed to bn, then return bn layer
    if conv_index == 4:
        block = getattr(model, 'down_block1')
    elif conv_index == 9:
        block = getattr(model, 'down_block2')
    elif conv_index == 14:
        block = getattr(model, 'down_block3')
    elif conv_index == 19:
        block = getattr(model, 'down_block4')
    elif conv_index == 24:
        block = getattr(model, 'down_block5')
    else:
        raise Exception('neki je narobe, ni prava vrednost?')

    return getattr(block, 'bn')

def _prune_next_bn_if_needed(layer_index, filters_begin, filters_end, pruned_filters, device, model):
    next_bn_index = None
    if layer_index == 4:  # layer_index == 0 or layer_index == 1 or layer_index == 2 or layer_index == 3 or layer_index == 4:
        next_bn_index = 4  # 4, 9, 14, 19, 24
    elif layer_index == 9:  # layer_index == 5 or layer_index == 6 or layer_index == 7 or layer_index == 8 or layer_index == 9:
        next_bn_index = 9  # 4, 9, 14, 19, 24
    elif layer_index == 14:  # layer_index == 10 or layer_index == 11 or layer_index == 12 or layer_index == 13 or layer_index == 14:
        next_bn_index = 14  # 4, 9, 14, 19, 24
    elif layer_index == 19:  # layer_index == 15 or layer_index == 16 or layer_index == 17 or layer_index == 18 or layer_index == 19:
        next_bn_index = 19  # 4, 9, 14, 19, 24
    elif layer_index == 24:  # layer_index == 20 or layer_index == 21 or layer_index == 22 or layer_index == 23 or layer_index == 24:
        next_bn_index = 24  # 4, 9, 14, 19, 24
    else:
        next_bn = None

    if next_bn_index is not None:
        next_bn = _get_bn_by_prev_conv_index(next_bn_index, model)

    # Surgery on next batchnorm layer
    if next_bn is not None:
        logger.write('additionally pruning batch norm with index {0}'.format(next_bn_index))
        logger.write('n features compressed from {0} to {1} '.format(next_bn.num_features, next_bn.num_features - pruned_filters))
        next_new_bn = \
            torch.nn.BatchNorm2d(num_features=next_bn.num_features - pruned_filters, \
                                 eps=next_bn.eps, \
                                 momentum=next_bn.momentum, \
                                 affine=next_bn.affine,
                                 track_running_stats=next_bn.track_running_stats)
        next_bn.num_features -= pruned_filters

        old_weights = next_bn.weight.data.cpu().numpy()
        new_weights = next_new_bn.weight.data.cpu().numpy()
        old_bias = next_bn.bias.data.cpu().numpy()
        new_bias = next_new_bn.bias.data.cpu().numpy()
        old_running_mean = next_bn.running_mean.data.cpu().numpy()
        new_running_mean = next_new_bn.running_mean.data.cpu().numpy()
        old_running_var = next_bn.running_var.data.cpu().numpy()
        new_running_var = next_new_bn.running_var.data.cpu().numpy()

        new_weights[: filters_begin] = old_weights[: filters_begin]
        new_weights[filters_begin:] = old_weights[filters_end + 1:]
        #next_bn.weight.data = torch.from_numpy(new_weights).to(device)
        next_bn.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(device))
        next_bn.weight.grad = None

        new_bias[: filters_begin] = old_bias[: filters_begin]
        new_bias[filters_begin:] = old_bias[filters_end + 1:]
        #next_bn.bias.data = torch.from_numpy(new_bias).to(device)
        next_bn.bias = torch.nn.Parameter(torch.from_numpy(new_bias).to(device))
        next_bn.bias.grad = None

        new_running_mean[: filters_begin] = old_running_mean[: filters_begin]
        new_running_mean[filters_begin:] = old_running_mean[filters_end + 1:]
        next_bn.running_mean.data = torch.from_numpy(new_running_mean).to(device)
        #next_bn.running_mean = torch.nn.Parameter(torch.from_numpy(new_running_mean).to(device))
        next_bn.running_mean.grad = None

        new_running_var[: filters_begin] = old_running_var[: filters_begin]
        new_running_var[filters_begin:] = old_running_var[filters_end + 1:]
        next_bn.running_var.data = torch.from_numpy(new_running_var).to(device)
        #next_bn.running_var = torch.nn.Parameter(torch.from_numpy(new_running_var).to(device))
        next_bn.running_var.grad = None



def _layer_index_to_conv(layer_index, model):
    if layer_index == 0:
        block_name = 'down_block1'
        conv_name = 'conv1'
    elif layer_index == 1:
        block_name = 'down_block1'
        conv_name = 'conv21'
    elif layer_index == 2:
        block_name = 'down_block1'
        conv_name = 'conv22'
    elif layer_index == 3:
        block_name = 'down_block1'
        conv_name = 'conv31'
    elif layer_index == 4:
        block_name = 'down_block1'
        conv_name = 'conv32'
    elif layer_index == 5:
        block_name = 'down_block2'
        conv_name = 'conv1'
    elif layer_index == 6:
        block_name = 'down_block2'
        conv_name = 'conv21'
    elif layer_index == 7:
        block_name = 'down_block2'
        conv_name = 'conv22'
    elif layer_index == 8:
        block_name = 'down_block2'
        conv_name = 'conv31'
    elif layer_index == 9:
        block_name = 'down_block2'
        conv_name = 'conv32'
    elif layer_index == 10:
        block_name = 'down_block3'
        conv_name = 'conv1'
    elif layer_index == 11:
        block_name = 'down_block3'
        conv_name = 'conv21'
    elif layer_index == 12:
        block_name = 'down_block3'
        conv_name = 'conv22'
    elif layer_index == 13:
        block_name = 'down_block3'
        conv_name = 'conv31'
    elif layer_index == 14:
        block_name = 'down_block3'
        conv_name = 'conv32'
    elif layer_index == 15:
        block_name = 'down_block4'
        conv_name = 'conv1'
    elif layer_index == 16:
        block_name = 'down_block4'
        conv_name = 'conv21'
    elif layer_index == 17:
        block_name = 'down_block4'
        conv_name = 'conv22'
    elif layer_index == 18:
        block_name = 'down_block4'
        conv_name = 'conv31'
    elif layer_index == 19:
        block_name = 'down_block4'
        conv_name = 'conv32'
    elif layer_index == 20:
        block_name = 'down_block5'
        conv_name = 'conv1'
    elif layer_index == 21:
        block_name = 'down_block5'
        conv_name = 'conv21'
    elif layer_index == 22:
        block_name = 'down_block5'
        conv_name = 'conv22'
    elif layer_index == 23:
        block_name = 'down_block5'
        conv_name = 'conv31'
    elif layer_index == 24:
        block_name = 'down_block5'
        conv_name = 'conv32'

    elif layer_index == 25:
        block_name = 'up_block1'
        conv_name = 'conv11'
    elif layer_index == 26:
        block_name = 'up_block1'
        conv_name = 'conv12'
    elif layer_index == 27:
        block_name = 'up_block1'
        conv_name = 'conv21'
    elif layer_index == 28:
        block_name = 'up_block1'
        conv_name = 'conv22'
    elif layer_index == 29:
        block_name = 'up_block2'
        conv_name = 'conv11'
    elif layer_index == 30:
        block_name = 'up_block2'
        conv_name = 'conv12'
    elif layer_index == 31:
        block_name = 'up_block2'
        conv_name = 'conv21'
    elif layer_index == 32:
        block_name = 'up_block2'
        conv_name = 'conv22'
    elif layer_index == 33:
        block_name = 'up_block3'
        conv_name = 'conv11'
    elif layer_index == 34:
        block_name = 'up_block3'
        conv_name = 'conv12'
    elif layer_index == 35:
        block_name = 'up_block3'
        conv_name = 'conv21'
    elif layer_index == 36:
        block_name = 'up_block3'
        conv_name = 'conv22'
    elif layer_index == 37:
        block_name = 'up_block4'
        conv_name = 'conv11'
    elif layer_index == 38:
        block_name = 'up_block4'
        conv_name = 'conv12'
    elif layer_index == 39:
        block_name = 'up_block4'
        conv_name = 'conv21'
    elif layer_index == 40:
        block_name = 'up_block4'
        conv_name = 'conv22'
    elif layer_index == 41:
        return getattr(model, 'out_conv1'), None, 'out_conv1'
    else:
        raise Exception('neki je narobe pri pridobivanju conv iz layer indexa')

    block = getattr(model, block_name)
    return getattr(block, conv_name), block, conv_name


def _prune_next_layer(next_conv_i, filters_begin, filters_end, pruned_filters, device, model):
    logger.write('Additionally pruning (next layer) conv with layer_id ' + str(next_conv_i))
    assert filters_begin == filters_end
    next_conv, block, layer_name = _layer_index_to_conv(next_conv_i, model)

    next_new_conv = \
        torch.nn.Conv2d(in_channels=next_conv.in_channels - pruned_filters, \
                        out_channels=next_conv.out_channels, \
                        kernel_size=next_conv.kernel_size, \
                        stride=next_conv.stride,
                        padding=next_conv.padding,
                        dilation=next_conv.dilation,
                        groups=1,  # next_conv.groups,
                        bias=True
                        )  # next_conv.bias)
    next_conv.in_channels -= pruned_filters

    old_weights = next_conv.weight.data.cpu().numpy()
    new_weights = next_new_conv.weight.data.cpu().numpy()

    new_weights[:, : filters_begin, :, :] = old_weights[:, : filters_begin, :, :]
    new_weights[:, filters_begin:, :, :] = old_weights[:, filters_end + 1:, :, :]

    next_conv.weight = torch.nn.Parameter(torch.from_numpy(new_weights).to(device))
    #        next_conv.weight.data = torch.from_numpy(new_weights).to(self.device)
    next_conv.weight.grad = None

    if next_conv_i == 41: # out conv: ne popravljam aktivacij, ker jih nimam z ato konvolucijo
        return

    index = filters_begin
    # ALSO: change activations sum for this conv layer # todo: i dont update activations
    layer_activations_sum = getattr(block,
                                    layer_name + '_activations_sum')  # vektor dolzine toliko kolikor je filtrov, za vsak filter je ena stevilka
    layer_activations_sum = torch.cat([layer_activations_sum[0:index], layer_activations_sum[index + 1:]])
    setattr(block, layer_name + '_activations_sum',
            torch.nn.Parameter(layer_activations_sum.to(device), requires_grad=False))



    """
    layer_weight = getattr(layer, 'weight')
    layer_bias = getattr(layer, 'bias')
    filter_weight = layer_weight[index]
    filter_bias = layer_bias[index]
    with torch.no_grad():
        filter_weight =  torch.zeros(filter_weight.shape).to(device)
        filter_bias =  torch.zeros(filter_bias.shape).to(device)
        layer_weight[index] = filter_weight
        layer_bias[index] = filter_bias
        #na tem mestu se ze dejansko pozna sprememba v modelu
        #model_layer.weight = layer_weight
        #setattr(model, name, model_layer)  # requires_grad stays True on weight (because of no_grad
        #print(model.conv1.weight)

    n_disabled_params_for_weight = np.prod(filter_weight.shape)
    n_disabled_params_for_bias = np.prod(filter_bias.shape)
    n_disabled_params = n_disabled_params_for_bias + n_disabled_params_for_weight
    #print('disabled {0} parameters'.format(n_disabled_params))
    
    return n_disabled_params
    """

def get_parameter_name_and_index_from_activations_dict_key(key):
    assert len(key.split('-')) == 2
    name, index = key.split('-')
    return name, int(index)


def outer_hook(device, filter_index):
    def hook_fn(grad):
        new_grad = grad.clone()  # remember that hooks should not modify their argument
        mask = torch.ones(new_grad.shape).to(device)
        mask[filter_index, :, :, :] = torch.zeros(new_grad.shape[1:]).to(device)
        new_grad_multiplied = new_grad.mul_(mask)
        return new_grad_multiplied
    return hook_fn



def get_curr_params_for_layer(device, model, name):
    block_name, layer_name = name.split('.')
    block = getattr(model, block_name)
    layer = getattr(block, layer_name)
    n_biases, n_weights, n_learnable_biases, n_learnable_weights = count_learnable_parameters_for_module(name,
                                                                                                         layer,
                                                                                                         device)
    return n_learnable_biases + n_learnable_weights

def get_curr_params_for_block(device, model, block_name):
    sum_params_for_block = 0
    for name, module in model.named_modules():  # always weight and bias
        if (name.startswith(block_name)):
            n_biases, n_weights, n_learnable_biases, n_learnable_weights = count_learnable_parameters_for_module(name,
                                                                                                             module,
                                                                                                             device)
            sum_params_for_block += n_learnable_biases + n_learnable_weights

    return sum_params_for_block



def remove_filters_and_retrain_model(args, device, trainloader, testloader, start_epoch, end_epoch, viz, win_train_mIoU_during_retraining,
                                     win_train_loss_during_retraining, model, optimizer, teacher_model, device_teacher, parameters, criterion, criterion_DICE, criterion_SL,
                                     all_parameters_layer_name_dict,
                                     all_parameters_for_block_dict, targeting_flops, before_step_flops, rm
                                     ):
    # 1. EVAL MODEL AND CALCULATE MODEL'S ACTIVATIONS - RETURN ALL ACTIVATIONS WITH CORESPONDING NAMES IN DICT
    train_mIoU = get_mIoU_on_train_and_populate_activations(args, device, trainloader, model)  # first eval on test, then rank on all training images
    logger.write('train mIoU: {0}'.format(train_mIoU))
    # check which filters or blocks should not have any more parameters removed
    layers_with_exceeded_limit = []
    blocks_with_exceeded_limit = []
    for layer_name, value in all_parameters_layer_name_dict.items():
        percent_removed_params = 100 - (100 * get_curr_params_for_layer(device, model, layer_name) / all_parameters_layer_name_dict[layer_name])
        if percent_removed_params >= parameters['layer_limit_in_percents']:
            layers_with_exceeded_limit.append(layer_name)
        assert len(layer_name.split('.')) == 2
        block_name, _ = layer_name.split('.')
        curr_params_for_block = get_curr_params_for_block(device, model, block_name)
        percent_removed_params = 100 - (100 * curr_params_for_block / all_parameters_for_block_dict[block_name])
        if percent_removed_params >= parameters['block_limit_in_percents']:
            blocks_with_exceeded_limit.append(block_name)

    n_removed_flops_for_this_step = 0
    # 2. GET FILTERS WITH MINIMAL ACTIVATIONS AND DISABLE IT'S WEIGHTS
    while n_removed_flops_for_this_step < parameters['n_removed_flops_for_this_step']:
        # here model should have activations calculated
        # this must be done on updated activations (after filters have been prunned)
        all_activations_sum_dict_sorted = _get_sorted_filter_activations_dict_for_model(model)

        # set weights of this filter to zero AND set it's gradients to zero
        key, value = get_filter_with_minimal_activation(device, all_activations_sum_dict_sorted, layers_with_exceeded_limit, blocks_with_exceeded_limit)
        logger.write('disabling filter ' + str(key) + ' with value ' + str(value))

        disable_filter(device, model, key)  # disable this filter - zero filter's weights

        #layer_name, index = get_parameter_name_and_index_from_activations_dict_key(key)
        #block_name, layer_name_without_block = layer_name.split('.')
        #block = getattr(model, block_name)
        #model_layer = getattr(block, layer_name_without_block)
        #model_layer_weight = getattr(model_layer, 'weight')
        #model_layer_weight.register_hook(outer_hook(device, index))  # also zero gradients of this filter

        # update parameters info for layer and block
        #zeroed_parameters_layer_name_dict[layer_name] = zeroed_parameters_layer_name_dict[layer_name] + n_disabled_params
        #zeroed_parameters_for_block_dict[block_name] = zeroed_parameters_for_block_dict[block_name] + n_disabled_params

        for layer_name, value in all_parameters_layer_name_dict.items():
            percent_removed_params = 100 - (100 * get_curr_params_for_layer(device, model, layer_name) / \
                                     all_parameters_layer_name_dict[layer_name])
            if percent_removed_params >= parameters['layer_limit_in_percents']:
                layers_with_exceeded_limit.append(layer_name)
            assert len(layer_name.split('.')) == 2
            block_name, _ = layer_name.split('.')
            percent_removed_params = 100 - (100 * get_curr_params_for_block(device, model, block_name) / \
                                     all_parameters_for_block_dict[block_name])
            if percent_removed_params >= parameters['block_limit_in_percents']:
                blocks_with_exceeded_limit.append(block_name)

        # update how many flops we removed
        model.reset_conv_activations_sum(device)
        rm.calculate_resources(torch.zeros((1, 1, args.height, args.width), device=device))
        n_removed_flops_for_this_step = before_step_flops - rm.cur_flops
        logger.write('in this step we removed {0} flops'.format(n_removed_flops_for_this_step))
        if (rm.cur_flops <= targeting_flops):
            logger.write('Reached {0} flops, targeting {1} flops'.format(rm.cur_flops, targeting_flops))
            return model  # do not train afterwards


    # 3. TRAIN MODEL WITHOUT REMOVED FILTER
    model.train()
    alpha = parameters['alpha']
    conf_matrix_whole = np.zeros((args.numberOfClasses, args.numberOfClasses))
    for epoch in range(start_epoch, end_epoch):
        training_loss_sum = 0.0
        training_hinton_loss_sum = 0.0
        training_attention_loss_sum = 0.0
        training_fsp_loss_sum = 0.0
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            if device_teacher is not None:
                data_device_teacher = data.to(device_teacher)
            else:
                data_device_teacher = data
            target = labels.to(device).long()
            optimizer.zero_grad()
            student_batch_outputs = model(data)

            teacher_batch_outputs = fetch_teacher_outputs(args, teacher_model, data_device_teacher)

            loss, loss_hinton_float, loss_attention_float, loss_fsp_float = \
                calculate_loss_for_batch_CE(args, device, device_teacher, model, student_batch_outputs, teacher_model, teacher_batch_outputs,
                                            target, index, spatialWeights, maxDist, alpha, epoch, parameters, criterion,
                                            criterion_DICE, criterion_SL)
            # 4. backprop only on student model
            loss.backward()  # this is merged loss (original and distillation)

            # performs updates using calculated gradients
            optimizer.step()

            training_loss_sum += loss.item()
            training_hinton_loss_sum += loss_hinton_float
            training_attention_loss_sum += loss_attention_float
            training_fsp_loss_sum += loss_fsp_float

            predictions = get_predictions(student_batch_outputs)
            conf_matrix_batch = get_conf_matrix(args, predictions, target)
            conf_matrix_whole += conf_matrix_batch

            if i % 100 == 0:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), loss.item()))

        train_mIoU = conf_matrix_to_mIoU(args, conf_matrix_whole)

        """
        try:
            viz.line(
                X=np.array([epoch]),
                Y=np.array([train_mIoU]),
                win=win_train_mIoU_during_retraining,
                name='train mIoU',
                update='append'
            )

            # merged loss (original and distillation)
            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_loss_sum]),
                win=win_train_loss_during_retraining,
                name='train loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_hinton_loss_sum]),
                win=win_train_loss_during_retraining,
                name='train hinton loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_attention_loss_sum]),
                win=win_train_loss_during_retraining,
                name='train attention loss',
                update='append'
            )

            viz.line(
                X=np.array([epoch]),
                Y=np.array([training_fsp_loss_sum]),
                win=win_train_loss_during_retraining,
                name='train fsp loss',
                update='append'
            )

        except Exception:
            pass
        
        
        if epoch == start_epoch:
            # create line to see where trainig without filter starts
            try:
                viz.line(
                    X=np.array([epoch]),
                    Y=np.array([0]),
                    win=win_train_mIoU_during_retraining,
                    update='append',
                    name='new filter ' + str(epoch),
                    opts=dict(
                        linecolor=np.array([[0, 0, 255]])
                    )
                )
                # and keep cursor on last mIoU
                viz.line(
                    X=np.array([epoch]),
                    Y=np.array([train_mIoU]),
                    win=win_train_mIoU_during_retraining,
                    update='append',
                    name='new filter ' + str(epoch),
                    opts=dict(
                        linecolor=np.array([[0, 0, 255]])
                    )
                )

                # do the same for loss plot
                viz.line(
                    X=np.array([epoch]),
                    Y=np.array([0]),
                    win=win_train_loss_during_retraining,
                    update='append',
                    name='new filter ' + str(epoch),
                    opts=dict(
                        linecolor=np.array([[0, 0, 255]])
                    )
                )

                viz.line(
                    X=np.array([epoch]),
                    Y=np.array([training_loss_sum]),
                    win=win_train_loss_during_retraining,
                    update='append',
                    name='new filter ' + str(epoch),
                    opts=dict(
                        linecolor=np.array([[0, 0, 255]])
                    )
                )
            except Exception as ex:
                print(ex)
                raise ex
        """

    return model


def count_zeroed_filters_for_layer_name(model, block_name, layer_name, device):
    block = getattr(model, block_name)
    layer = getattr(block, layer_name)
    return count_zeroed_filter_for_layer(layer, layer_name, device, block_name)


def count_zeroed_filter_for_layer(layer, layer_name, device, block_name=''):
    weight = getattr(layer, 'weight')
    bias = getattr(layer, 'bias')
    # should have gradients set to zero and also weights
    assert len(weight.shape) == 4
    assert len(bias.shape) == 1
    used_filters = []
    zeroed_filters = []
    zero_filter_3d = torch.zeros(weight.shape[1:]).to(device)
    zero_filter_1d = torch.zeros(bias.shape[1:]).to(device)
    if weight.grad is None:
        #for filter_index, filter_weight in enumerate(weight):  # bs
        #    filter_name = '{0}-{1}'.format(layer_name, filter_index)
        #    used_filters.append(filter_name)
        #return zeroed_filters, used_filters
        for filter_index, (filter_weight, filter_bias) in enumerate(zip(weight, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_bias, zero_filter_1d):
                filter_name = '{0}.{1}-{2}'.format(block_name, layer_name, filter_index)
                zeroed_filters.append(filter_name)
            else:
                filter_name = '{0}.{1}-{2}'.format(block_name, layer_name, filter_index)
                used_filters.append(filter_name)

        if len(zeroed_filters) > 0:
            logger.write('WARNING: zeroed weights are interpreted as disabled parameters!')
        return zeroed_filters, used_filters
    else:
        assert weight.grad is not None
        for filter_index, (filter_weight, filter_grad, filter_bias) in enumerate(zip(weight, weight.grad, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_grad, zero_filter_3d) and torch.equal(
                    filter_bias, zero_filter_1d):
                filter_name = '{0}.{1}-{2}'.format(block_name, layer_name, filter_index)
                zeroed_filters.append(filter_name)
            else:
                filter_name = '{0}.{1}-{2}'.format(block_name, layer_name, filter_index)
                used_filters.append(filter_name)

        return zeroed_filters, used_filters


def count_zeroed_filters_for_model(model, device):
    all_zeroed_filters = []
    all_used_filters = []
    for name, param in model.named_parameters():  # always weight and bias
        if name.endswith('_activations_sum'):
            block_name, layer_name = name.replace('_activations_sum', '').split('.')
            zeroed_filters, used_filters = count_zeroed_filters_for_layer_name(model, block_name, layer_name, device)
            all_zeroed_filters = all_zeroed_filters + zeroed_filters
            all_used_filters = all_used_filters + used_filters

    return all_zeroed_filters, all_used_filters


def count_learnable_parameters_for_module(name, module, device):
    # get number of paramters for network, exclude parameters for zeroed filters
    # https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
    """
    NUMBER OF PARAMETERS IN CONV LAYER
    W_c = Number of weights of the ConvLayer.
    B_c = Number of biases ofthe ConvLayer.
    P_c = Number of parameters   of    the   Conv Layer.
    K = Size(width) of kernels used in the ConvLayer.
    N = Number of kernels.
    C = Number of channels of the input image.

    Wc = K * K * C * N  (depth of every kernel is always equal to number of channels in inut image. Every kernel has
    K*K*C parameters and there are N such kernels.
    Bc = N
    # parameters = Wc + Bc

    NUMBER OF PARAMETERS IN BATCHNORM LAYER
    len(bias) + len(weight) --> vedno  1 dimenzionalni
    """

    n_biases = 0
    n_weights = 0
    n_learnable_biases = 0
    n_learnable_weights = 0
    # get each layer and check its instance
    if hasattr(module, 'weight') and module.weight is not None:  # module is learnable if it has .weight attribute (maybe also bias)
        # logger.write('{0} is learnable (has weight)'.format(module))
        if isinstance(module, nn.BatchNorm2d):
            if module.affine:
                assert len(module.bias.shape) == 1
                assert len(module.weight.shape) == 1
                n_biases = module.bias.shape[0]
                n_weights = module.weight.shape[0]
                n_learnable_biases = n_biases  # all batchnorms are trainiable (im not excluding parameters here)
                n_learnable_weights = n_weights  # all batchnorms are trainiable (im not excluding parameters here)

        elif isinstance(module, nn.Conv2d):
            assert module.groups == 1  # za CONV assert groups 1!! ker drugace ne vem ce prav racunam
            assert len(module.kernel_size) == 2
            assert module.kernel_size[0] == module.kernel_size[1]
            n_weights = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels
            n_biases = 0
            if module.bias is not None:
                n_biases = module.out_channels
            # do not count manually zeroed filters
            zeroed_filters, used_filters = count_zeroed_filter_for_layer(module, name, device)
            assert len(zeroed_filters) + len(used_filters) == module.out_channels

            if module.bias is None:
                # ker nevem ali so excludani filtri z biasom ali brez. Moj model ima pri vseh Conv2D bias zraven
                raise NotImplemented('counting paramters for conv2d without bias is not implemented')

            n_learnable_weights = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * len(used_filters)
            n_learnable_biases = len(used_filters)

        else:
            raise NotImplementedError()

    return n_biases, n_weights, n_learnable_biases, n_learnable_weights


def count_number_of_learnable_parameters(model, device):
    all_parameters = 0 # all learnable parameters with zeroed parameters
    learnable_parameters = 0 # learnable parameters excluding manually zeroed parameters

    for name, module in model.named_modules():  # always weight and bias
        n_biases, n_weights, n_learnable_biases, n_learnable_weights = count_learnable_parameters_for_module(name, module, device)
        all_parameters = all_parameters + n_biases + n_weights
        learnable_parameters = learnable_parameters + n_learnable_biases + n_learnable_weights

    return learnable_parameters, all_parameters



def get_zeroed_and_all_parameters_wrt_layer_for_model_and_zeroed_percent_for_layer_and_block(model, device):
    zeroed_parameters_layer_name_dict = dict()
    all_parameters_layer_name_dict = dict()
    zeroed_parameters_for_block_dict = dict()
    all_parameters_for_block_dict = dict()
    percent_zeroed_for_layer = dict()
    percent_zeroed_for_block = dict()
    curr_block_zeroed_params_sum = 0
    curr_block_all_params_sum = 0
    curr_block_name = None

    # nedded to plot number of zeroed filters for each layer
    for name, module in model.named_modules():  # always weight and bias
        n_biases, n_weights, n_learnable_biases, n_learnable_weights = count_learnable_parameters_for_module(name, module, device)
        if n_biases == 0 and n_weights == 0:
            continue

        if len(name.split('.')) != 2:
            print('skipping {0}'.format(name))
            continue # do not count out_conv on model itself

        # this layer is learnable
        n_zeroed_params = (n_biases - n_learnable_biases) + (n_weights - n_learnable_weights)
        n_all_params = n_biases + n_weights
        zeroed_parameters_layer_name_dict[name] = n_zeroed_params
        all_parameters_layer_name_dict[name] = n_all_params
        assert len(name.split('.')) == 2
        block_name, layer_name = name.split('.')

        if block_name != curr_block_name:
            if curr_block_name is not None:
                zeroed_parameters_for_block_dict[curr_block_name] = curr_block_zeroed_params_sum
                all_parameters_for_block_dict[curr_block_name] = curr_block_all_params_sum
                percent_zeroed = 100 * curr_block_zeroed_params_sum / curr_block_all_params_sum
                assert percent_zeroed <= 100
                percent_zeroed_for_block[curr_block_name] = percent_zeroed
            curr_block_name = block_name
            curr_block_zeroed_params_sum = 0
            curr_block_all_params_sum = 0

        percent_zeroed = 100 * n_zeroed_params / n_all_params
        assert percent_zeroed <= 100
        percent_zeroed_for_layer[name] = percent_zeroed
        curr_block_zeroed_params_sum += n_zeroed_params
        curr_block_all_params_sum += n_all_params

    # save last block
    if curr_block_name != None:
        zeroed_parameters_for_block_dict[curr_block_name] = curr_block_zeroed_params_sum
        all_parameters_for_block_dict[curr_block_name] = curr_block_all_params_sum
        percent_zeroed = 100 * curr_block_zeroed_params_sum / curr_block_all_params_sum
        assert percent_zeroed <= 100
        percent_zeroed_for_block[curr_block_name] = percent_zeroed

    return zeroed_parameters_layer_name_dict, all_parameters_layer_name_dict, zeroed_parameters_for_block_dict, \
           all_parameters_for_block_dict, percent_zeroed_for_layer, percent_zeroed_for_block


def load_student(args, device):
    student_model = model_dict['student']
    logger.write('using student model: ' + str(type(student_model)))
    student_model = student_model.to(device)

    if args.resume != '':
        logger.write("EXISTING STUDENT DICT from: {}".format(args.resume))
        student_state_dict = torch.load(args.resume)
        student_model.load_state_dict(student_state_dict, strict=False)
        print('USING LOAD_MODEL STRICT=FALSE')
        # student_model.eval() # not needed if training continues

    return student_model


def start_pruning(args, device, trainloader, model, validloader, testloader, teacher_model, device_teacher, viz, parameters, criterion, criterion_DICE, criterion_SL):
    start_epoch = 0 #parameters['start_epoch'] #args.startEpoch
    end_epoch = start_epoch + parameters['n_epochs_for_retraining'] #args.epochs

    teacher_model_copy = copy.deepcopy(teacher_model) # check that teacher is not changing

    win_test_miou_wrt_zeroed_filters = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Test mIoU wrt zeroed filters'
        )
    )

    win_number_of_parameters_wrt_zeroed_filters = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Number of parameters'
        )
    )

    win_train_mIoU_during_retraining = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Train mIoU during retraining'
        )
    )

    win_train_loss_during_retraining = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='Train loss during retraining'
        )
    )

    # just to create window
    win_layers_and_n_removed_parameters = viz.bar(
        X=[1,2],
        opts=dict(
            rownames=['test', 'test2'],
            title='Number of removed parameters for each layer',
            marginbottom=130,
            marginright=80
        )
    )

    # just to create window
    win_layers_and_percent_removed_parameters = viz.bar(
        X=[1,2],
        opts=dict(
            rownames=['test', 'test2'],
            title='Percent of removed parameters for each layer',
            marginbottom=130,
            marginright=80
        )
    )
    # just to create window
    win_blocks_and_percent_removed_parameters = viz.bar(
        X=[1,2],
        opts=dict(
            rownames=['test', 'test2'],
            title='Percent of removed parameters for each block',
            marginbottom=100,
            marginright=80
        )
    )

    win_flops = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(
            title='FLOPS'
        )
    )


    #n_filters_to_remove_at_once = parameters['n_filters_to_remove_at_once']

    rm = ResourceManager(model)
    rm.calculate_resources(torch.zeros((1, 1, args.height, args.width), device=device))
    original_flops = rm.cur_flops
    targeting_flops = (1 - (parameters['prune_away_percent']/100)) * original_flops

    all_parameters_for_block_dict = {
        'down_block1':  22080,
        'down_block2':  32992,
        'down_block3':  32992,
        'down_block4':  32992,
        'down_block5':  32992,
        'up_block1': 23680,
        'up_block2': 23680,
        'up_block3': 23680,
        'up_block4': 23680,
    }

    all_parameters_layer_name_dict = {
        'down_block1.conv1':  320,
        'down_block1.conv21':  1088,
        'down_block1.conv22':  9248,
        'down_block1.conv31':  2112,
        'down_block1.conv32':  9248,
        'down_block2.conv1':  9248,
        'down_block2.conv21':  2080,
        'down_block2.conv22':  9248,
        'down_block2.conv31':  3104,
        'down_block2.conv32':  9248,
        'down_block3.conv1':  9248,
        'down_block3.conv21':  2080,
        'down_block3.conv22':  9248,
        'down_block3.conv31':  3104,
        'down_block3.conv32':  9248,
        'down_block4.conv1': 9248,
        'down_block4.conv21': 2080,
        'down_block4.conv22': 9248,
        'down_block4.conv31': 3104,
        'down_block4.conv32': 9248,
        'down_block5.conv1': 9248,
        'down_block5.conv21': 2080,
        'down_block5.conv22': 9248,
        'down_block5.conv31': 3104,
        'down_block5.conv32': 9248,
        'up_block1.conv11': 2080,
        'up_block1.conv12': 9248,
        'up_block1.conv21': 3104,
        'up_block1.conv22': 9248,
        'up_block2.conv11': 2080,
        'up_block2.conv12': 9248,
        'up_block2.conv21': 3104,
        'up_block2.conv22': 9248,
        'up_block3.conv11': 2080,
        'up_block3.conv12': 9248,
        'up_block3.conv21': 3104,
        'up_block3.conv22': 9248,
        'up_block4.conv11': 2080,
        'up_block4.conv12': 9248,
        'up_block4.conv21': 3104,
        'up_block4.conv22': 9248,
    }


    i = 0 # for plotting
    while True:
        if teacher_model is not None:
            models_equal = compare_models(teacher_model_copy, teacher_model)
            if not models_equal:
                logger.write('TEACHER MODEL HAS CHANGED!')

        summary(model, input_size=(1, args.height, args.width))  # , batch_size=args.bs)  #  input_size=(channels, H, W)

        # calculate flops for current model
        rm.calculate_resources(torch.zeros((1, 1, args.height, args.width), device=device))
        test_mIoU = eval_model_and_populate_activations(args, device, testloader, model)
        #learnable_parameters, all_parameters = count_number_of_learnable_parameters(model, device)
        logger.write('test mIoU = {1}, flops: {2}'
                     .format(None, test_mIoU, rm.cur_flops))
        try:
            viz.line(
                X=np.array([i]), # * n_filters_to_remove_at_once]),
                Y=np.array([rm.cur_flops]),
                win=win_flops,
                name='current flops',
                update='append',
            )
        except Exception as ex:
            print(ex)
            raise ex

        #if (i*n_filters_to_remove_at_once != rm.n_removed_filters):
        #    print(i*n_filters_to_remove_at_once)
        #    print(rm.n_removed_filters)
        #    print('ERROR: number of removed filters is not consistent in resourceManager!')
        #    raise Exception()

        # save model without n filters
        #torch.save(model.state_dict(), '{}/models/model_without_{}_filters.pkl'.format(LOGDIR, i*n_filters_to_remove_at_once))
        #torch.save(optimizer.state_dict(), '{}/optimizers/model_without_{}_filters.pkl'.format(LOGDIR, i*n_filters_to_remove_at_once))
        #torch.save(scheduler, '{}/schedulers/model_without_{}_filters.pkl'.format(LOGDIR, i*n_filters_to_remove_at_once))
        torch.save(model, '{}/models/ckpt_{}.pt'.format(LOGDIR, i))#*n_filters_to_remove_at_once))

        #zeroed_parameters_layer_name_dict, all_parameters_layer_name_dict, zeroed_parameters_for_block_dict, \
        #all_parameters_for_block_dict, percent_zeroed_for_layer, percent_zeroed_for_block = \
        #    get_zeroed_and_all_parameters_wrt_layer_for_model_and_zeroed_percent_for_layer_and_block(model, device)

        """
        zeroed_parameters_X = []
        zeroed_parameters_Y = []
        for key, value in zeroed_parameters_layer_name_dict.items():
            if value > 0:
                zeroed_parameters_X.append(value)
                zeroed_parameters_Y.append(key)
        # visdom bar cannot be plotted with just one column - in this case we add tmp column with value zero
        if len(zeroed_parameters_X) == 1:
            zeroed_parameters_X.append(0)
            zeroed_parameters_Y.append('tmp')
        #print(zeroed_parameters_layer_name_dict)

        percent_zeroed_parameters_for_layer_X = []
        percent_zeroed_parameters_for_layer_Y = []
        for key, value in percent_zeroed_for_layer.items():
            if value > 0:
                percent_zeroed_parameters_for_layer_X.append(value)
                percent_zeroed_parameters_for_layer_Y.append(key)
        # visdom bar cannot be plotted with just one column - in this case we add tmp column with value zero
        if len(percent_zeroed_parameters_for_layer_X) == 1:
            percent_zeroed_parameters_for_layer_X.append(0)
            percent_zeroed_parameters_for_layer_Y.append('tmp')
        #print(percent_zeroed_for_layer)

        percent_zeroed_parameters_for_block_X = []
        percent_zeroed_parameters_for_block_Y = []
        for key, value in percent_zeroed_for_block.items():
            if value > 0:
                percent_zeroed_parameters_for_block_X.append(value)
                percent_zeroed_parameters_for_block_Y.append(key)
        # visdom bar cannot be plotted with just one column - in this case we add tmp column with value zero
        if len(percent_zeroed_parameters_for_block_X) == 1:
            percent_zeroed_parameters_for_block_X.append(0)
            percent_zeroed_parameters_for_block_Y.append('tmp')
        #print(percent_zeroed_for_block)
        """
        try:

            viz.line(
                X=np.array([i]),
                Y=np.array([test_mIoU]),
                win=win_test_miou_wrt_zeroed_filters,
                name='test mIoU',
                update='append',
            )
            """
            viz.line(
                X=np.array([i*n_filters_to_remove_at_once]),
                Y=np.array([learnable_parameters]),
                win=win_number_of_parameters_wrt_zeroed_filters,
                name='number of learnable parameters',
                update='append',
            )

            viz.bar(
                X=zeroed_parameters_X,
                win=win_layers_and_n_removed_parameters,
                opts=dict(
                    rownames=zeroed_parameters_Y,
                    title='Number of removed parameters for each layer',
                    marginbottom=130,
                    marginright=80
                )
            )
            viz.bar(
                X=percent_zeroed_parameters_for_layer_X,
                win=win_layers_and_percent_removed_parameters,
                opts=dict(
                    rownames=percent_zeroed_parameters_for_layer_Y,
                    title='Percent of removed parameters for each layer',
                    marginbottom=130,
                    marginright=80
                )
            )
            viz.bar(
                X=percent_zeroed_parameters_for_block_X,
                win=win_blocks_and_percent_removed_parameters,
                opts=dict(
                    rownames=percent_zeroed_parameters_for_block_Y,
                    title='Percent of removed parameters for each block',
                    marginbottom=100,
                    marginright=80
                )
            )
            """

        except Exception as ex:
            print(ex)
            raise ex

        #if learnable_parameters < parameters['learnable_parameters_min_limit']:
        #    logger.write('Reached {0} learnable parameters, exiting..'.format(learnable_parameters))
        #    break

        if rm.cur_flops < targeting_flops: # (1 - (parameters['prune_away_percent']/100)) * original_flops:
            logger.write('Reached {0} flops, exiting..'.format(rm.cur_flops))
            break
        else:
            logger.write('not yet: {0}/{1} FLOPS'.format(rm.cur_flops, original_flops))
        # =========================================================== NEW MODEL IS CREATED
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        model = remove_filters_and_retrain_model(args, device, trainloader, testloader, start_epoch, end_epoch, viz,
                                                 win_train_mIoU_during_retraining, win_train_loss_during_retraining,
                                                 model, optimizer, teacher_model, device_teacher, parameters, criterion, criterion_DICE, criterion_SL,
                                                 all_parameters_layer_name_dict,
                                                 all_parameters_for_block_dict, targeting_flops, rm.cur_flops, rm)
        # update epochs number
        start_epoch = end_epoch
        end_epoch = end_epoch + parameters['n_epochs_for_retraining']#args.epochs

        # reset activations in model (conv has changed)
        model.reset_conv_activations_sum(device)

        #all_zeroed_filters, all_used_filters = count_zeroed_filters_for_model(model, device)
        #learnable_parameters, all_parameters = count_number_of_learnable_parameters(model, device)
        #logger.write('zeroed filters ({0}): {1}'.format(len(all_zeroed_filters), all_zeroed_filters))
        #logger.write('used filters ({0})'.format(len(all_used_filters)))
        #logger.write('number of parameters: {0}/{1}'.format(learnable_parameters, all_parameters))

        n_learnable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.write('pytorch: number of learnable parameters: {0}'.format(n_learnable_parameters))
        i = i + 1

    learnable_parameters, all_parameters = count_number_of_learnable_parameters(model, device)
    #from torchsummary import summary
    #summary(model, input_size=(1, args.height, args.width))  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    # print(student_model)

    logger.write('learnable parameters: {0}/{1}'.format(learnable_parameters, all_parameters))


def initialize_globals(args):
    global LOGDIR
    LOGDIR = EYEZ/'Segmentation/Sclera/Results/Matic RITnet'/args.expname
    global logger
    logger = Logger(os.path.join(LOGDIR, 'logs.log'))


def main():
    args = parse_args()

    initialize_globals(args)
    (LOGDIR/'models').mkdir(parents=True, exist_ok=True)
    # os.makedirs(LOGDIR + '/optimizers', exist_ok=True)
    # os.makedirs(LOGDIR + '/schedulers', exist_ok=True)

    kd_script_initialize_globals(args)


    #todo delete
    #args.resume = 'logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl'

    if args.resume == '':
        raise Exception('resume argument must be present when training with pruning')

    used_gpus_list = args.pruningGpus#.split(',')
    if args.useGPU == 'True' or args.useGPU == 'true':
        logger.write('USE GPUs: {0}'.format(used_gpus_list))
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in used_gpus_list)
        torch.cuda.manual_seed(7)
    else:
        logger.write('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(7)

    torch.backends.cudnn.deterministic = True

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    criterion_SL = SurfaceLoss()

    # visdom
    # RUN python -m visdom.server
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

    # PARAMETERS:
    parameters = {
        #'learnable_parameters_min_limit': 74788,  # compressed student has this much parameters
        #'flops_min_limit': 0, # end when min flops are reached
        'prune_away_percent': 74, # How many percentage of constraints should be pruned away. E.g., 50 means 50% of FLOPs will be pruned away
        #'n_filters_to_remove_at_once': 10,
        'n_removed_flops_for_this_step': 1000000000,
        #'start_epoch': 0,
        'n_epochs_for_retraining': 5,
        'block_limit_in_percents': 75,  # how many parameters can be disabled in one block
        'layer_limit_in_percents': 75,  # how many parameters can be disabled in one layer
        'alpha': None,  # will take 0 as value
        'alpha_original': 1,
        # -------------------HINTON-------------------
        'alpha_distillation': 0.001,
        'T': 8,  # TODO 2, 4, 8, 16
        # ---------------ATTENTION------------------
        'beta': 0.00, # beta 0.25 in hinton 0.00005 sta priblizno 30 oba. in original je 30. Ideja je, da hoces vec destilacije kot originalnega lossa
        # -------------FSP------------------
        'lambda': 0.0
    }

    opt = vars(args)
    logger.write(str(opt))
    logger.write(str(parameters))

    device_teacher = None
    teacher_model = None
    if (parameters['alpha_distillation'] != 0 or parameters['beta'] != 0 or parameters['lambda'] != 0) and len(
            used_gpus_list) != 2:
        print(used_gpus_list)
        raise Exception('Distillation requires two GPUs')

    if len(used_gpus_list) == 2: # if distillation is used and we (must) have 2 gpus
        device = torch.device("cuda:{0}".format(used_gpus_list[0]))
        device_teacher = torch.device("cuda:{0}".format(used_gpus_list[1]))
        args.teacher = 'logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl'
        logger.write('using teacher: {0}'.format(args.teacher))
        teacher_model = load_teacher(args, device_teacher)

    model = load_student(args, device) # ignore optimizer and scheduler
    trainloader, validloader, testloader = get_data_loaders(args)

    start_pruning(args, device, trainloader, model, validloader, testloader, teacher_model, device_teacher, viz, parameters, criterion, criterion_DICE, criterion_SL)

    # todo: matematicna porazdelitev



# python train_with_pruning.py --expname pruning_10filters_at_once_retrain_for_10epochs_distillation_att_b0_001_resize_activations40_25_layer_limit70_block_limit70 --resume logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl --pruningGpus 0 1


# python train_with_pruning.py --expname pruning_test_flops --resume logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl --gpu 0

if __name__ == '__main__':
    main()
