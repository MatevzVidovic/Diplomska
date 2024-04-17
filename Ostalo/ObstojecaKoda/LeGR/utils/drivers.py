import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataset.dataset import IrisDataset

from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import re

def get_dataloader(img_size, dataset, datapath, batch_size, no_val, workers, get_only_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    if 'eyes' in dataset:
        n_classes = 4 if 'sip' in dataset else 2
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])  # TODO ali dam to ven? ker v članku ne najdem tega
             ])
        Path2file = '../IPAD/' + dataset
        print('path to file: ' + str(Path2file))

        test = IrisDataset(filepath=Path2file, split='test', n_classes=n_classes,
                           transform=transform)

        testloader = DataLoader(test, batch_size=batch_size,
                                shuffle=False, num_workers=workers)

        if get_only_test:
            return testloader


        train = IrisDataset(filepath=Path2file, split='train', n_classes=n_classes,
                                    transform=transform)
        valid = IrisDataset(filepath=Path2file, split='validation', n_classes=n_classes,
                            transform=transform)

        trainloader = DataLoader(train, batch_size=batch_size,
                                 shuffle=False, num_workers=workers, drop_last=True)

        validloader = DataLoader(valid, batch_size=batch_size,
                                 shuffle=False, num_workers=workers, drop_last=True)

        return trainloader, validloader, testloader

    if img_size == 32:
        train_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
        val_set = eval(dataset)(datapath, True, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        num_train = len(train_set)
        indices = list(range(num_train))
        split = int(np.floor(0.1 * num_train))

        np.random.seed(int(time.time()))
        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        test_set = eval(dataset)(datapath, False, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        if no_val:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
    else:
        if dataset == 'ImageNetData':
            # ToTensor and Normalization is done inside SeqImageNetLoader
            train_loader = SeqImageNetLoader('train', batch_size=batch_size, num_workers=8, cuda=True, remainder=False,
                    transform=transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip()]))

            test_loader = SeqImageNetLoader('val', batch_size=32, num_workers=8, cuda=True, remainder=True,
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224)]))

            val_loader = test_loader
        else:
            train_set = eval(dataset)(datapath, True, transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
            val_set = eval(dataset)(datapath, True, transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

            num_train = len(train_set)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train))

            np.random.seed(98)
            np.random.shuffle(indices)

            np.random.seed(int(time.time()))

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            test_set = eval(dataset)(datapath, False, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]))

            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size, sampler=train_sampler,
                num_workers=0, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size, sampler=valid_sampler,
                num_workers=0, pin_memory=True
            )
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


import torch.nn as nn
class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        #print('outputs:')
        #print(outputs.shape)
        #print(torch.unique(outputs))
        #print('target:')
        #print(targets.shape)
        #print(torch.unique(targets))
        return self.loss(F.log_softmax(outputs,dim=1), targets)


class GeneralizedDiceLoss(nn.Module):
    # Author: Rakshit Kothari
    # Input: (B, C, ...)
    # Target: (B, C, ...)
    def __init__(self, epsilon=1e-5, weight=None, softmax=True, reduction=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = []
        self.reduction = reduction
        if softmax:
            self.norm = nn.Softmax(dim=1)
        else:
            self.norm = nn.Sigmoid()

    def forward(self, ip, target):

        # Rapid way to convert to one-hot. For future version, use functional
        Label = (np.arange(ip.shape[1]) == target.cpu().numpy()[..., None]).astype(np.uint8)
        target = torch.from_numpy(np.rollaxis(Label, 3, start=1)).cuda()

        assert ip.shape == target.shape
        ip = self.norm(ip)

        # Flatten for multidimensional data
        ip = torch.flatten(ip, start_dim=2, end_dim=-1).cuda().to(torch.float32)
        target = torch.flatten(target, start_dim=2, end_dim=-1).cuda().to(torch.float32)

        numerator = ip * target
        denominator = ip + target

        class_weights = 1. / (torch.sum(target, dim=2) ** 2).clamp(min=self.epsilon)

        A = class_weights * torch.sum(numerator, dim=2)
        B = class_weights * torch.sum(denominator, dim=2)

        dice_metric = 2. * torch.sum(A, dim=1) / torch.sum(B, dim=1)
        if self.reduction:
            return torch.mean(1. - dice_metric.clamp(min=self.epsilon))
        else:
            return 1. - dice_metric.clamp(min=self.epsilon)

class SurfaceLoss(nn.Module):
    # Author: Rakshit Kothari
    def __init__(self, epsilon=1e-5, softmax=True):
        super(SurfaceLoss, self).__init__()
        self.weight_map = []

    def forward(self, x, distmap):
        x = torch.softmax(x, dim=1)
        self.weight_map = distmap
        score = x.flatten(start_dim=2) * distmap.flatten(start_dim=2)
        score = torch.mean(score, dim=2)  # Mean between pixels per channel
        score = torch.mean(score, dim=1)  # Mean between channels
        return score


def get_ritnet_loss(device, batch_outputs, target, spatialWeights, maxDist, alpha, epoch):
    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    criterion_SL = SurfaceLoss()

    CE_loss = criterion(batch_outputs, target)
    loss = CE_loss * (
            torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device) + (
        spatialWeights).to(
        torch.float32).to(device))
    loss = torch.mean(loss).to(torch.float32).to(device)
    loss_dice = criterion_DICE(batch_outputs, target)
    loss_sl = torch.mean(criterion_SL(batch_outputs.to(device), (maxDist).to(device)))
    if alpha is None:
        loss_original = (1 - 0) * loss_sl + 0 * (loss_dice) + loss  # for testing take alpha=0
    else:
        loss_original = (1 - alpha[epoch]) * loss_sl + alpha[epoch] * (loss_dice) + loss
    return loss_original


def get_predictions(output):
    bs, c, h, w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices


def get_conf_matrix(predictions, targets, num_classes):
    predictions_np = predictions.detach().cpu().long().numpy()
    targets_np = targets.cpu().long().numpy()
    # for batch of predictions
    # if len(np.unique(targets)) != 2:
    #    print(len(np.unique(targets)))
    assert (predictions.shape == targets.shape)
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


def conf_matrix_to_mIoU(confusion_matrix, print_per_class_IoU=False):
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
    # if confusion_matrix.shape != (4, 4):
    #     print(confusion_matrix.shape)
    #     raise NotImplementedError()

    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))

    if print_per_class_IoU:
        print("per-class mIOU: " + str(MIoU))

    if len(confusion_matrix) == 2:
        return MIoU.item(1) # only IoU for sclera (not background)
    else:
        return np.mean(MIoU)


def test(model, data_loader, device='cuda', get_loss=False, n_img=-1):

    if hasattr(data_loader.dataset, 'filepath') and 'IPAD' in data_loader.dataset.filepath:

        """
        model.eval()
        conf_matrix_whole = np.zeros((4, 4))
        with torch.no_grad():
            for i, batchdata in tqdm(enumerate(data_loader), total=len(data_loader)):
                img, label_tensor, index, spatialWeights, maxDist = batchdata
                data = img.to(device)
                target = label_tensor.to(device).long()
                output = model(data)
                predictions = get_predictions(output)
                conf_matrix_batch = get_conf_matrix(predictions, target)
                conf_matrix_whole += conf_matrix_batch

        mIoU = conf_matrix_to_mIoU(conf_matrix_whole)
        return mIoU
        """

        #model.to(device)
        model.eval()
        epoch_loss = []
        n_classes = data_loader.dataset.classes
        conf_matrix_whole = np.zeros((n_classes, n_classes))
        with torch.no_grad():
            for i, batchdata in enumerate(data_loader):
                img, labels, index, spatialWeights, maxDist = batchdata
                data = img.to(device)
                target = labels.to(device).long()

                batch_outputs = model(data)
                loss_original = get_ritnet_loss(device, batch_outputs, target, spatialWeights, maxDist, alpha=None, epoch=None)

                epoch_loss.append(loss_original.item())
                predict = get_predictions(batch_outputs)
                conf_matrix_batch = get_conf_matrix(predict, target, n_classes)
                conf_matrix_whole += conf_matrix_batch

        average_val_iou = conf_matrix_to_mIoU(conf_matrix_whole)
        return average_val_iou, epoch_loss





    else:
        model.to(device)
        model.eval()
        correct = 0
        total = 0

        """
        if hasattr(data_loader.dataset, 'filepath') and 'IPAD' in data_loader.dataset.filepath:
            if get_loss:
                criterion = CrossEntropyLoss2d()
                loss = np.zeros(0)

            total_len = len(data_loader)
            if n_img > 0 and total_len > int(np.ceil(float(n_img) / data_loader.batch_size)):
                total_len = int(np.ceil(float(n_img) / data_loader.batch_size))


            for i, batchdata in enumerate(data_loader):

                print('==============={0}: memory allocated: {1:.3f}/{2:.3f} GB'.format(
                    'new batch data',
                    torch.cuda.memory_allocated(device=torch.cuda.current_device()) / np.power(1024, 3),
                    torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / np.power(1024, 3)))


                if i >= total_len:
                    break

                img, labels, index, spatialWeights, maxDist = batchdata
                data = img.to(device)
                target = labels.to(device).long()

                output = model(data)

                tmp = np.array([criterion(output, target).data.cpu().numpy()])
                if get_loss:
                    loss = np.concatenate((loss, tmp))
                pred = output.data.max(1)[1]
                correct += pred.eq(target).sum()
                total += target.size(0)

            if get_loss:
                return float(correct)/total*100, loss
            else:
                return float(correct)/total*100

        else:
        """

        if get_loss:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            loss = np.zeros(0)

        total_len = len(data_loader)
        if n_img > 0 and total_len > int(np.ceil(float(n_img) / data_loader.batch_size)):
            total_len = int(np.ceil(float(n_img) / data_loader.batch_size))
        for i, (batch, label) in enumerate(data_loader):
            if i >= total_len:
                break
            batch, label = batch.to(device), label.to(device)
            output = model(batch)
            if get_loss:
                loss = np.concatenate((loss, criterion(output, label).detach().cpu().numpy()))
            pred = output.detach().max(1)[1]
            correct += pred.eq(label).sum()
            total += label.size(0)

        if get_loss:
            return float(correct)/total*100, loss
        else:
            return float(correct)/total*100



def outer_hook(name, param_itself):
    def hook_fn(grad):
        print(name + " " + str(grad.shape))
        #print(param_itself.shape)
        #new_grad = grad.clone()  # remember that hooks should not modify their argument
        #mask = torch.ones(new_grad.shape).to(device)
        #mask[filter_index, :, :, :] = torch.zeros(new_grad.shape[1:]).to(device)
        #new_grad_multiplied = new_grad.mul_(mask)
        #return new_grad_multiplied
    return hook_fn



#from pytorchviz.torchviz.dot import make_dot

def classify_to_group(filename):
    if 'i_' in filename:
        return 'indoor'
    elif 'n_' in filename:
        return 'normal'
    elif 'p_' in filename:
        return 'poor'
    else:
        return 'sbvpi'

def classify_to_phone(filename):
    x = re.search(".*_([123])[inp]_.*", filename)
    return x.group(1)

def evaluate_on_test_set(device, model, testloader):
    model.eval()
    n_classes = testloader.dataset.classes
    conf_matrix_whole = np.zeros((n_classes, n_classes))
    list_of_test_mIoUs = []  # for calculating distribution and standard deviation
    conf_matrix_sbvpi = np.zeros((n_classes, n_classes))
    sbvpi_count = 0
    list_of_test_sbvpi_mIoUs = []
    conf_matrix_mobius_poor = np.zeros((n_classes, n_classes))
    mobius_poor_count = 0
    list_of_test_poor_mIoUs = []
    conf_matrix_mobius_normal = np.zeros((n_classes, n_classes))
    mobius_normal_count = 0
    list_of_test_normal_mIoUs = []
    conf_matrix_mobius_indoor = np.zeros((n_classes, n_classes))
    mobius_indoor_count = 0
    list_of_test_indoor_mIoUs = []
    # phone groups
    list_of_test_mious_phone1 = []
    list_of_test_mious_phone2 = []
    list_of_test_mious_phone3 = []
    with torch.no_grad():
        for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
            img, label_tensor, filename, x, y = batchdata
            data = img.to(device)
            output = model(data)
            target = label_tensor.to(device).long()
            predictions = get_predictions(output)

            # iou_b = mIoU(predictions, labels)
            # ious_all.append(iou_b)
            conf_matrix_batch = get_conf_matrix(predictions, target, n_classes)
            conf_matrix_whole += conf_matrix_batch
            for idx, (prediction, label) in enumerate(zip(predictions, target)):
                conf_matrix = get_conf_matrix(prediction, label, n_classes)
                # get mIoU for each image in batch
                image_mIoU = conf_matrix_to_mIoU(conf_matrix, False)  # vrne mIoU, mean od vseh stirih IoU-jev
                # print(image_mIoU)
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

                phone_label = classify_to_phone(filename[idx])
                if phone_label == '1':
                    list_of_test_mious_phone1.append(image_mIoU)
                elif phone_label == '2':
                    list_of_test_mious_phone2.append(image_mIoU)
                elif phone_label == '3':
                    list_of_test_mious_phone3.append(image_mIoU)
                else:
                    print('ERROR: phone label {0} not recognized'.format(phone_label))

        # check group conf matrices and whole matrix == prve štiri bi se mogle seštet v zadnjo.. [OK]
        # print(conf_matrix_mobius_poor)
        # print(conf_matrix_mobius_normal)
        # print(conf_matrix_mobius_indoor)
        # print(conf_matrix_sbvpi)
        # print('===========================')
        # print(conf_matrix_whole)

        # conf matrix to iou

        print('Statistics:')
        # ious_mobius_poor = conf_matrix_to_mIoU(args, conf_matrix_mobius_poor)
        # print('# images in poor group: ' + str(mobius_poor_count) + ', mIoU (for all 4 classes): ' + str(ious_mobius_poor))
        print(
            'poor category (len {0}): mean: mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_poor_mIoUs),
                                                                                      np.mean(
                                                                                          list_of_test_poor_mIoUs),
                                                                                      np.std(
                                                                                          list_of_test_poor_mIoUs)))
        # ious_mobius_normal = conf_matrix_to_mIoU(args, conf_matrix_mobius_normal)
        # print('# images in normal group: ' + str(mobius_normal_count) + ', mIoU (for all 4 classes): ' + str(ious_mobius_normal))
        print('normal category (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_normal_mIoUs),
                                                                                    np.mean(
                                                                                        list_of_test_normal_mIoUs),
                                                                                    np.std(
                                                                                        list_of_test_normal_mIoUs)))
        # ious_mobius_indoor = conf_matrix_to_mIoU(args, conf_matrix_mobius_indoor)
        # print('# images in indoor group: ' + str(mobius_indoor_count) + ', mIoU (for all 4 classes): ' + str(ious_mobius_indoor))
        print('indoor category (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_indoor_mIoUs),
                                                                                    np.mean(
                                                                                        list_of_test_indoor_mIoUs),
                                                                                    np.std(
                                                                                        list_of_test_indoor_mIoUs)))
        if sbvpi_count != 0:
            # ious_sbvpi = conf_matrix_to_mIoU(args, conf_matrix_sbvpi)
            # print('# images in sbvpi group: ' + str(sbvpi_count) + ', mIoU (for all 4 classes): ' + str(ious_sbvpi))
            print(
                'sbvpi category (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_sbvpi_mIoUs),
                                                                                     np.mean(
                                                                                         list_of_test_sbvpi_mIoUs),
                                                                                     np.std(
                                                                                         list_of_test_sbvpi_mIoUs)))
        print('Test mIoU: {:0.5f}'.format(conf_matrix_to_mIoU(conf_matrix_whole, False)))
        print('Test mIoU (len {0}) mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_mIoUs),
                                                                             np.mean(list_of_test_mIoUs),
                                                                             np.std(list_of_test_mIoUs)))

        # phone statistics
        print('phone1 (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_mious_phone1),
                                                                           np.mean(list_of_test_mious_phone1),
                                                                           np.std(list_of_test_mious_phone1)))
        print('phone2 (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_mious_phone2),
                                                                           np.mean(list_of_test_mious_phone2),
                                                                           np.std(list_of_test_mious_phone2)))
        print('phone3 (len {0}): mean +- std: {1:0.5f} +- {2:0.5f}'.format(len(list_of_test_mious_phone3),
                                                                           np.mean(list_of_test_mious_phone3),
                                                                           np.std(list_of_test_mious_phone3)))


# create object for calculating model's flops
class ResourceManager():
    def __init__(self, model):
        self.model = model

    def _get_num_gen(self, gen):
        return sum(1 for x in gen)

    def _is_leaf(self, model):
        return self._get_num_gen(model.children()) == 0

    def trace_layer(self, layer, x):
        y = layer.old_forward(x)
        if isinstance(layer, nn.Conv2d):
            h = y.shape[2]
            w = y.shape[3]
            self.cur_flops += h * w * layer.weight.size(0) * layer.weight.size(1) * layer.weight.size(
                2) * layer.weight.size(3)
            #print('size0={0}, size2={1}, size3={2}, size4={3}: '.format(layer.weight.size(0),layer.weight.size(1),layer.weight.size(2),layer.weight.size(3)))

        elif isinstance(layer, nn.BatchNorm2d):
            pass

        elif isinstance(layer, nn.Linear):
            self.cur_flops += np.prod(layer.weight.shape)

        return y

    def calculate_resources(self, x):
        # tale ubistvu spremeni forward tako, da poklice trace_layer na vsakem. V trace nardis dejansko forward, poleg tega pa se
        # izracunas stevilo flopov.

        self.cur_flops = 0

        def modify_forward(model):
            for child in model.children():
                if self._is_leaf(child):
                    def new_forward(m):
                        def lambda_forward(x):
                            return self.trace_layer(m, x)

                        return lambda_forward

                    child.old_forward = child.forward
                    child.forward = new_forward(child)
                else:
                    modify_forward(child)

        def restore_forward(model):
            for child in model.children():
                # leaf node
                if self._is_leaf(child) and hasattr(child, 'old_forward'):
                    child.forward = child.old_forward
                    child.old_forward = None
                else:
                    restore_forward(child)

        modify_forward(self.model)
        y = self.model.forward(x)
        restore_forward(self.model)

        # also calculate trainable parameters
        self.cur_n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)



def retrain_model(device, model, optimizer, scheduler, trainloader, validloader, retrain_epochs, viz=None, win_loss=None, win_iou=None):
    # 3. TRAIN MODEL WITHOUT REMOVED FILTER
    #print('RETRAINING MODEL FOR {0} EPOCHS'.format(retrain_epochs))
    model.to(device)
    model.train()

    alpha = np.zeros(retrain_epochs)
    alpha[0:np.min([125, retrain_epochs])] = 1 - np.arange(1, np.min([125, retrain_epochs]) + 1) / np.min([125, retrain_epochs])
    if retrain_epochs > 125:
        alpha[125:] = 0
    #print(alpha)

    #print('registering hooks...')
    #for name, param_w_or_b in model.named_parameters():  # always weight and bias
        #print(name)
        #print(param_w_or_b.shape)
        #param_w_or_b.register_hook(outer_hook(name, param_w_or_b))  # also zero gradients of this filter

    for epoch in trange(retrain_epochs, desc=f"Retraining model for {retrain_epochs} epochs"):
        training_loss_sum = 0.0
        with trange(len(trainloader), desc="Training", leave=False) as pbar:
            for i, batchdata in enumerate(trainloader):
                img, labels, index, spatialWeights, maxDist = batchdata
                data = img.to(device)
                target = labels.to(device).long()
                optimizer.zero_grad()
                batch_outputs = model(data)
                loss = get_ritnet_loss(device, batch_outputs, target, spatialWeights, maxDist, alpha, epoch)

                # 4. backprop only on student model
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                training_loss_sum += loss.item()

                #predictions = get_predictions(batch_outputs)
                #conf_matrix_batch = get_conf_matrix(predictions, target)
                #conf_matrix_whole += conf_matrix_batch

                pbar.update()
                if not i % 10:
                    pbar.set_postfix(Loss=f"{loss.item():.3f}")

            #train_mIoU = conf_matrix_to_mIoU(conf_matrix_whole)

            if viz is not None and win_loss is not None:
                plot_with_visdom(viz, win_loss, epoch, training_loss_sum, 'training loss')


        # calculate validation loss
        val_loss_list = []
        n_classes = validloader.dataset.classes
        conf_matrix_whole = np.zeros((n_classes, n_classes))
        with torch.no_grad(), trange(len(validloader), desc="Validating", leave=False) as pbar:
            for i, batchdata in enumerate(validloader):
                img, labels, index, spatialWeights, maxDist = batchdata
                data = img.to(device)
                target = labels.to(device).long()
                batch_outputs = model(data)

                loss = get_ritnet_loss(device, batch_outputs, target, spatialWeights, maxDist, alpha, epoch)

                val_loss_list.append(loss.item())
                predict = get_predictions(batch_outputs)
                conf_matrix_batch = get_conf_matrix(predict, target, n_classes)
                conf_matrix_whole += conf_matrix_batch

                pbar.update()
                if not i % 10:
                    pbar.set_postfix(Loss=f"{loss.item():.3f}")

        average_val_iou = conf_matrix_to_mIoU(conf_matrix_whole)

        if viz is not None and win_loss is not None:
            plot_with_visdom(viz, win_loss, epoch, np.sum(val_loss_list), 'validation loss')

        if viz is not None and win_iou is not None:
            plot_with_visdom(viz, win_iou, epoch, average_val_iou, 'validation mIoU')

        scheduler.step(np.average(val_loss_list))

    return model


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


def train(model, train_loader, val_loader, optimizer=None, epochs=10, steps=None, scheduler=None, run_test=True, name='', device='cuda'):
    model.to(device)
    if optimizer is None:
        optimizer = optim.SGD(model.classifier.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)
    if scheduler is None:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Use number of steps as unit instead of epochs
    if steps:
        epochs = int(steps / len(train_loader)) + 1
        if epochs > 1:
            steps = steps % len(train_loader)

    best_acc = 0
    for i in range(epochs):
        print('Epoch: {}'.format(i))
        if i == epochs - 1:
            loss = train_epoch(model, train_loader, optimizer, steps=steps, device=device)
        else:
            loss = train_epoch(model, train_loader, optimizer, device=device)
        scheduler.step()

        if run_test:
            acc = test(model, val_loader)
            print('Testing Accuracy {:.2f}'.format(acc))
            if i and best_acc < acc:
                best_acc = acc
                torch.save(model, os.path.join('ckpt', '{}_best.t7'.format(name)))

def train_epoch(model, train_loader, optimizer=None, steps=None, device='cuda', distillation=None):
    model.to(device)
    model.train()
    losses = np.zeros(0)
    total_loss = 0
    data_t = 0
    train_t = 0
    criterion = torch.nn.CrossEntropyLoss()
    s = time.time()
    for i, (batch, label) in enumerate(train_loader):
        batch, label = batch.to(device), label.to(device)
        data_t += time.time() - s
        s = time.time()

        model.zero_grad()
        output = model(batch)

        if distillation:
            t_out = distillation.teacher(batch)
            soft_target = F.softmax(t_out/distillation.T, dim=1)
            logp = F.log_softmax(output/distillation.T, dim=1)
            soft_loss = -torch.mean(torch.sum(soft_target)*logp, dim=1)
            soft_loss = soft_loss + distillation.T * distillation.T

            loss = criterion(output, label) + distillation.alpha * soft_loss
            loss.backward()
        else:
            loss = criterion(output, label)
            loss.backward()
        optimizer.step()

        total_loss += loss
        losses = np.concatenate([losses, np.array([loss.item()])])

        train_t += time.time() - s
        length = steps if steps and steps < len(train_loader) else len(train_loader)

        if (i % 100 == 0) or (i == length-1):
            print('Training | Batch ({}/{}) | Loss {:.4f} ({:.4f}) | (PerBatchProfile) Data: {:.3f}s, Net: {:.3f}s'.format(i+1, length, total_loss/(i+1), loss, data_t/(i+1), train_t/(i+1)))
        if i == length-1:
            break
        s = time.time()
    return np.mean(losses)
