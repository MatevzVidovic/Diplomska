import copy

from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset import IrisDataset, transform #, init_seed
import torch

from utils import mIoU, CrossEntropyLoss2d, total_metric, get_nparams, Logger, GeneralizedDiceLoss, SurfaceLoss, \
    mIoU_conf_matrix
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
from train_with_knowledge_distillation import calculate_loss_for_batch_CE, plot_with_visdom, conf_matrix_to_iou, validate, evaluate_on_test_during_train, initialize_globals


def set_model_to_train(model):
    if model is not None and not model.training:
        logger.write('setting model ' + str(type(model)) + ' to train()')
        model.train()

def set_model_to_eval(model):
    if model is not None and model.training:
        logger.write('setting model ' + str(type(model)) + ' to eval()')
        model.eval()




def train_teacher(args, device, teacher_model, teacher_optimizer, teacher_scheduler, trainloader, validloader, testloader, alpha,
                  epoch, teacher_parameters, criterion, criterion_DICE, criterion_SL, viz, teacher_win_loss, teacher_win_iou):
    teacher_conf_matrix_whole = np.zeros((args.numberOfClasses, args.numberOfClasses))
    teacher_training_loss_sum = 0.0
    for i, batchdata in enumerate(trainloader):
        if i == 10:
            logger.write('i==10: cuda memory allocated: ' + str(torch.cuda.memory_allocated(device)))
        #logger.write('total memory: ' + str(torch.cuda.get_device_properties(device).total_memory))
        #logger.write('cached memory: ' + str(torch.cuda.memory_cached(device)))

        img, labels, index, spatialWeights, maxDist = batchdata
        data = img.to(device)
        target = labels.to(device).long()

        # first train teacher
        teacher_optimizer.zero_grad()

        teacher_batch_outputs = teacher_model(data)

        # for teacher send custom parameters - use only original loss
        teacher_loss, loss_hinton_float_t, loss_attention_float_t, loss_fsp_float_t = calculate_loss_for_batch_CE(args,
                                                                                                                  device,
                                                                                                                  teacher_model,
                                                                                                                  teacher_batch_outputs,
                                                                                                                  None,
                                                                                                                  None,
                                                                                                                  target,
                                                                                                                  index,
                                                                                                                  spatialWeights,
                                                                                                                  maxDist,
                                                                                                                  alpha,
                                                                                                                  epoch,
                                                                                                                  teacher_parameters,
                                                                                                                  criterion,
                                                                                                                  criterion_DICE,
                                                                                                                  criterion_SL)

        assert loss_hinton_float_t == 0.0
        assert loss_attention_float_t == 0.0
        assert loss_fsp_float_t == 0.0

        # backpropagate on teacher
        teacher_loss.backward()

        # performs updates using calculated gradients
        teacher_optimizer.step()

        teacher_training_loss_sum += teacher_loss.item()

        teacher_predictions = get_predictions(teacher_batch_outputs)
        teacher_conf_matrix_batch = mIoU_conf_matrix(teacher_predictions, target)
        teacher_conf_matrix_whole += teacher_conf_matrix_batch

        if i % 10 == 0:
            logger.write(
                '[Teacher] Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), teacher_loss.item()))

        """
        # delete unneccesary parameters
        logger.write('deleting teacher_loss, data, target, teacher_batch_outputs....')
        del teacher_loss
        del data
        del target
        del teacher_batch_outputs
        """
    # visualize total training loss
    plot_with_visdom(viz, teacher_win_loss, epoch, teacher_training_loss_sum, 'teacher training loss')

    teacher_miou_train = conf_matrix_to_iou(args, teacher_conf_matrix_whole)
    logger.write('Epoch:{}, Train mIoU: {}'.format(epoch, teacher_miou_train))

    set_model_to_eval(teacher_model)  # for validation loss and also test afterwards
    assert not teacher_model.training, 'Teacher must be in eval mode'

    teacher_lossvalid, teacher_validation_loss_sum = validate(args, device, teacher_model, validloader, None, viz,
                                                              teacher_win_loss,
                                                              teacher_win_iou, alpha, epoch, teacher_parameters,
                                                              criterion, criterion_DICE, criterion_SL)
    teacher_scheduler.step(teacher_lossvalid)

    #del teacher_lossvalid

    # every epoch calculate test IoU
    evaluate_on_test_during_train(args, device, teacher_model, testloader, None, epoch, viz, teacher_win_iou,
                                  visualize_on_test_set=False)

    # SAVE MODEL
    if epoch % 5 == 0:
        torch.save(teacher_model.state_dict(), '{}/models_teacher/dense_net_{}.pkl'.format(LOGDIR, epoch))
        torch.save(teacher_optimizer.state_dict(), '{}/optimizers_teacher/dense_net_{}.pkl'.format(LOGDIR, epoch))
        torch.save(teacher_scheduler, '{}/schedulers_teacher/dense_net_{}.pkl'.format(LOGDIR, epoch))







def train_student(args, device, teacher_model, trainloader, validloader, testloader, student_model, student_optimizer,
                  student_scheduler, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL, viz, win_loss, win_iou):
    training_hinton_loss_sum = 0.0
    training_attention_loss_sum = 0.0
    training_fsp_loss_sum = 0.0
    conf_matrix_whole = np.zeros((args.numberOfClasses, args.numberOfClasses))
    training_loss_sum = 0.0

    for i, batchdata in enumerate(trainloader):
        img, labels, index, spatialWeights, maxDist = batchdata
        data = img.to(device)
        target = labels.to(device).long()

        student_optimizer.zero_grad()
        assert student_model.training, 'Student must be in training mode'
        student_batch_outputs = student_model(data)
        assert not teacher_model.training, 'Teacher must be in eval mode'

        with torch.no_grad():
            teacher_batch_outputs = teacher_model(data)

        loss, loss_hinton_float, loss_attention_float, loss_fsp_float = calculate_loss_for_batch_CE(args, device,
                                                                                                    student_model,
                                                                                                    student_batch_outputs,
                                                                                                    teacher_model,
                                                                                                    teacher_batch_outputs,
                                                                                                    target, index,
                                                                                                    spatialWeights,
                                                                                                    maxDist, alpha,
                                                                                                    epoch, parameters,
                                                                                                    criterion,
                                                                                                    criterion_DICE,
                                                                                                    criterion_SL)

        # 4. backprop only on student model
        loss.backward()

        # performs updates using calculated gradients
        student_optimizer.step()

        training_loss_sum += loss.item()

        training_hinton_loss_sum += loss_hinton_float
        training_attention_loss_sum += loss_attention_float
        training_fsp_loss_sum += loss_fsp_float

        predictions = get_predictions(student_batch_outputs)
        conf_matrix_batch = mIoU_conf_matrix(predictions, target)
        conf_matrix_whole += conf_matrix_batch

        if i % 10 == 0:
            logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(epoch, i, len(trainloader), loss.item()))

    # visualize total training loss
    plot_with_visdom(viz, win_loss, epoch, training_loss_sum, 'training loss')
    plot_with_visdom(viz, win_loss, epoch, training_hinton_loss_sum, 'training hinton loss')
    plot_with_visdom(viz, win_loss, epoch, training_attention_loss_sum, 'training attention loss')
    plot_with_visdom(viz, win_loss, epoch, training_fsp_loss_sum, 'training fsp loss')

    miou_train = conf_matrix_to_iou(args, conf_matrix_whole)
    logger.write('Epoch:{}, Train mIoU: {}'.format(epoch, miou_train))

    # *********VALIDATION*******************************************************************************************

    set_model_to_eval(student_model)  # for validation loss and also test afterwards
    set_model_to_eval(teacher_model)  # should already be in eval mode
    assert not student_model.training, 'Student must be in eval mode'
    if teacher_model is not None:
        assert not teacher_model.training, 'Teacher must be in eval mode'

    lossvalid, validation_loss_sum = validate(args, device, student_model, validloader, teacher_model, viz, win_loss,
                                              win_iou, alpha, epoch, parameters, criterion, criterion_DICE,
                                              criterion_SL)
    student_scheduler.step(lossvalid)

    # every epoch calculate test IoU
    evaluate_on_test_during_train(args, device, student_model, testloader, teacher_model, epoch, viz, win_iou,
                                  visualize_on_test_set=True)

    # SAVE MODEL
    if epoch % 5 == 0:
        torch.save(student_model.state_dict(), '{}/models/dense_net_{}.pkl'.format(LOGDIR, epoch))
        torch.save(student_optimizer.state_dict(), '{}/optimizers/dense_net_{}.pkl'.format(LOGDIR, epoch))
        torch.save(student_scheduler, '{}/schedulers/dense_net_{}.pkl'.format(LOGDIR, epoch))

    return validation_loss_sum



def train(args, device, teacher_model, teacher_optimizer, teacher_scheduler, student_model, student_optimizer, student_scheduler,
          trainloader, validloader, testloader, viz, win_loss, win_iou, parameters, criterion, criterion_DICE, criterion_SL):
    alpha = parameters['alpha']

    # PARAMETERS FOR TEACHER:
    teacher_parameters = {
        'alpha': alpha,
        'alpha_original': 1,
        # -------------------HINTON-------------------
        'alpha_distillation': 0.0,
        'T': 1,
        # ---------------ATTENTION------------------
        'beta': 0,
        # -------------FSP------------------
        'lambda': 0.00000
    }
    teacher_win_loss = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
    )
    teacher_win_iou = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
    )

    prev_prev_val_loss = 10000
    prev_val_loss = 10000
    for epoch in range(args.startEpoch, args.startEpoch + args.epochs):
        set_model_to_train(student_model)
        set_model_to_train(teacher_model)
        assert student_model.training, 'Student must be in training mode'
        assert teacher_model.training, 'Teacher must be in training mode'

        logger.write('cuda memory allocated: ' + str(torch.cuda.memory_allocated(device)))
        logger.write('total memory: ' + str(torch.cuda.get_device_properties(device).total_memory))
        logger.write('cached memory: ' + str(torch.cuda.memory_cached(device)))
        logger.write('allocated memory: ' + str(torch.cuda.memory_allocated(device)))


        train_teacher(args, device, teacher_model, teacher_optimizer, teacher_scheduler, trainloader, validloader,
                      testloader, alpha, epoch, teacher_parameters, criterion, criterion_DICE, criterion_SL, viz, teacher_win_loss,
                          teacher_win_iou)

        #logger.write('cuda memory allocated: ' + str(torch.cuda.memory_allocated(device)))
        #logger.write('total memory: ' + str(torch.cuda.get_device_properties(device).total_memory))
        #logger.write('cached memory: ' + str(torch.cuda.memory_cached(device)))
        #logger.write('allocated memory: ' + str(torch.cuda.memory_allocated(device)))

        #print("Emptying cache..")
        #torch.cuda.empty_cache()

        #logger.write('cuda memory allocated (after empty cache): ' + str(torch.cuda.memory_allocated(device)))


        validation_loss_sum = train_student(args, device, teacher_model, trainloader, validloader, testloader, student_model, student_optimizer,
                  student_scheduler, alpha, epoch, parameters, criterion, criterion_DICE, criterion_SL, viz, win_loss, win_iou)

        # stopping criteria
        if prev_prev_val_loss < prev_val_loss and prev_val_loss < validation_loss_sum:
            logger.write('validation loss increased two times in a row')
            break

        # save validation losses
        prev_prev_val_loss = prev_prev_val_loss
        prev_val_loss = validation_loss_sum


def get_data_loaders(args, ):
    kwargs = vars(args)
    Path2file = args.dataset
    logger.write('path to file: ' + str(Path2file))
    train_dataset = IrisDataset(filepath=Path2file, split='train',
                        transform=transform, **kwargs)
    logger.write('len: ' + str(train_dataset.__len__()))
    valid = IrisDataset(filepath=Path2file, split='validation',
                        transform=transform, **kwargs)

    trainloader = DataLoader(train_dataset, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True) # TODO shuffle??

    validloader = DataLoader(valid, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=True)

    test = IrisDataset(filepath=Path2file, split='test',
                       transform=transform, **kwargs)

    testloader = DataLoader(test, batch_size=args.bs,
                            shuffle=False, num_workers=args.workers)

    return trainloader, validloader, testloader


def load_teacher(args, device):
    teacher_model = model_dict['teacher']
    logger.write('using teacher model: ' + str(type(teacher_model)))
    teacher_model = teacher_model.to(device)

    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    return teacher_model, optimizer, scheduler


def load_student(args, device):
    student_model = model_dict['student']
    logger.write('using student model: ' + str(type(student_model)))
    student_model = student_model.to(device)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    """
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
    """
    return student_model, optimizer, scheduler


def initialize_globals_1(args):
    global LOGDIR
    LOGDIR = 'logs/{}'.format(args.expname)
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs(LOGDIR + '/models', exist_ok=True)
    os.makedirs(LOGDIR + '/models_teacher', exist_ok=True)
    os.makedirs(LOGDIR + '/optimizers', exist_ok=True)
    os.makedirs(LOGDIR + '/optimizers_teacher', exist_ok=True)
    os.makedirs(LOGDIR + '/schedulers', exist_ok=True)
    os.makedirs(LOGDIR + '/schedulers_teacher', exist_ok=True)
    global logger
    logger = Logger(os.path.join(LOGDIR, 'logs.log'))


def main():
    args = parse_args()

    initialize_globals(args)
    initialize_globals_1(args)

    if args.useGPU == 'True' or args.useGPU == 'true':
        logger.write('USE GPU')
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        torch.cuda.manual_seed(7)
    else:
        logger.write('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(7)

    torch.backends.cudnn.deterministic = True

    if args.teacher != '':
        print('Teacher is ignored!')

    teacher_model, teacher_optimizer, teacher_scheduler = load_teacher(args, device)
    student_model, student_optimizer, student_scheduler = load_student(args, device)

    # from torchsummary import summary
    # summary(student_model, input_size=(1, 192, 192))  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    # print(student_model)

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
    )
    win_iou = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
    )

    trainloader, validloader, testloader = get_data_loaders(args)

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
        'alpha_distillation': 0.00001,
        'T': 8, # TODO 2, 4, 8, 16
        # ---------------ATTENTION------------------
        'beta': 0.05,
        # -------------FSP------------------
        'lambda': 0.00000
    }
    if teacher_model is None and (parameters['alpha_distillation'] != 0.0 or parameters['beta'] != 0.0 or parameters['lambda'] != 0.0):
        logger.write('When using distillation methods, teacher model must be present!')
        return 0

    opt = vars(args)
    logger.write(str(opt))
    logger.write(str(parameters))

    train(args, device, teacher_model, teacher_optimizer, teacher_scheduler, student_model, student_optimizer, student_scheduler, trainloader, validloader, testloader, viz, win_loss, win_iou, parameters, criterion, criterion_DICE, criterion_SL)



# python train_kd_simultaneously.py --expname distillation_attention_simultaneously_beta0_01_lr0_001

if __name__ == '__main__':
    main()
