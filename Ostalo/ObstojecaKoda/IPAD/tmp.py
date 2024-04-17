import torch
import math
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import IrisDataset
from opt import parse_args
from train_with_knowledge_distillation import initialize_globals
from dataset import transform
from utils import CrossEntropyLoss2d

# original
class Net(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=False, prob=0):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.LeakyReLU()
        self.dropout = dropout
        self.dropout1 = nn.Dropout(p=prob)
        self.bn = torch.nn.BatchNorm2d(num_features=output_channels)
        self.output_channels = output_channels
        self.reset_conv_activations_sum()

    def forward(self, x):
        if self.dropout:
            self.conv1_activations = self.conv1(x)
            out = self.relu(self.dropout1(self.conv1_activations))
        else:
            self.conv1_activations = self.conv1(x)
            self.set_activations_for_removed_filters_to_zero(self.conv1, self.conv1_activations)
            self.conv1_activations_sum += self.calculate_activation_sum_for_layer(self.conv1_activations)
            out = self.relu(self.conv1_activations)

        return self.bn(out)

    def set_activations_for_removed_filters_to_zero(self, layer, layer_activations):
        # manualy set activations for removed filters to zero (otherwise we run into some small numbers)
        index_of_removed_filters = self.get_index_of_removed_filters_for_weight(layer)
        layer_activations[:, index_of_removed_filters, :, :] = torch.zeros(layer_activations.shape[0],
                                                                           len(index_of_removed_filters),
                                                                           layer_activations.shape[2],
                                                                           layer_activations.shape[3]).cuda()
        return layer_activations

    def calculate_activation_sum_for_layer(self, activations_for_layer):
        activations_for_layer_detached = activations_for_layer.detach()
        activations_for_layer_detached = activations_for_layer_detached.pow(2)
        n_summed_elements = activations_for_layer_detached.shape[0] * activations_for_layer_detached.shape[2] * \
                            activations_for_layer_detached.shape[3]
        activations_sum_for_layer = activations_for_layer_detached.sum(dim=[0, 2, 3]) / n_summed_elements

        return activations_sum_for_layer

    def get_index_of_removed_filters_for_weight(self, layer):
        weight = getattr(layer, 'weight')
        bias = getattr(layer, 'bias')
        # should have gradients set to zero and also weights and bias
        assert len(weight.shape) == 4
        assert len(bias.shape) == 1
        index_removed = []
        zero_filter_3d = torch.zeros(weight.shape[1:]).cuda()
        zero_filter_1d = torch.zeros(bias.shape[1:]).cuda()
        for index, (filter_weight, filter_bias) in enumerate(zip(weight, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_bias, zero_filter_1d):
                index_removed.append(index)
        return index_removed

    def reset_conv_activations_sum(self, device=None):
        print('resetting activations sum for all layers..')
        if device:
            self.conv1_activations_sum = torch.nn.Parameter(torch.zeros(self.output_channels).to(device), requires_grad=False)
        else:
            self.conv1_activations_sum = torch.nn.Parameter(torch.zeros(self.output_channels), requires_grad=False) # 1, output_c, 640, 400



def outer_hook(device, filter_index):
    def hook_fn(grad):
        new_grad = grad.clone()  # remember that hooks should not modify their argument
        mask = torch.ones(new_grad.shape).to(device)
        mask[filter_index, :, :, :] = torch.zeros(new_grad.shape[1:]).to(device)
        new_grad_multiplied = new_grad.mul_(mask)
        return new_grad_multiplied
    return hook_fn


def get_mIoU_and_rank_filters_activations(device, loader, model):
    eval_model_and_populate_activations(device, loader, model)
    # dict for saving all activations_sums
    # here model should have activations calculated
    all_activations_sum_dict_sorted = _get_sorted_filter_activations_dict_for_model(model)
    # todo return mIoU
    return all_activations_sum_dict_sorted

def eval_model_and_populate_activations(device, loader, model):
    model.eval()
    model.reset_conv_activations_sum(device)
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            output = model(data)

            # todo get predictions and calculate mIoU

    #return model

def _get_sorted_filter_activations_dict_for_model(model):
    # ACTIVATIONS MUST BE CALCULATED BEFORE RUNNING THIS METHOD
    # AFTER DISABLING FILTER, THIS METHOD HAS OLD ACTIVATIONS - THEY NEED TO BE POPULATED AGAIN
    all_activations_sum_dict = {}
    for name, param in model.named_parameters():  # always weight and bias
        if name.endswith('_activations_sum'):
            curr_activations_sum_tensor = getattr(model, name)
            for index, activation_sum in enumerate(curr_activations_sum_tensor):
                name_without_activations_sum_suffix = name.replace('_activations_sum', '')
                dict_key = '{0}-{1}'.format(name_without_activations_sum_suffix, index)
                all_activations_sum_dict[dict_key] = activation_sum  # cuda tensor

    all_activations_sum_dict_sorted = {k: v for k, v in
                                       sorted(all_activations_sum_dict.items(), key=lambda item: item[1])}
    return all_activations_sum_dict_sorted


def get_filter_with_minimal_activation(device, all_activations_sum_dict_sorted):
    # we must return filter with nonzero activation, because zeroed filters are not used in network
    for key, value in  all_activations_sum_dict_sorted.items():
        zero_tensor = torch.tensor(0, dtype=torch.float32).to(device)
        if torch.equal(value, zero_tensor):
            continue

        print('filter with minimal activation: ' + str(value))
        return key, value



def get_zeroed_parameters_wrt_layer_for_model(model, device):
    zeroed_parameters_layer_name_dict = dict()
    # nedded to plot number of zeroed filters for each layer
    for name, module in model.named_modules():  # always weight and bias
        n_biases, n_weights, n_learnable_biases, n_learnable_weights = count_learnable_parameters_for_module(name, module, device)
        if n_learnable_biases + n_learnable_weights > 0:
            # this layer is learnable
            if n_biases - n_learnable_biases > 0 or n_weights - n_learnable_weights > 0:
                # module has some zeroed biases
                zeroed_parameters_layer_name_dict[name] = (n_biases - n_learnable_biases) + n_weights - n_learnable_weights

    return zeroed_parameters_layer_name_dict


def remove_filter_and_retrain_model(device, trainloader, model, optimizer, criterion):
    # 1. EVAL MODEL AND CALCULATE MODEL'S ACTIVATIONS - RETURN ALL ACTIVATIONS WITH CORESPONDING NAMES IN DICT
    all_activations_sum_dict_sorted = get_mIoU_and_rank_filters_activations(device, trainloader,
                                                                        model)  # rank on all training images
    print(all_activations_sum_dict_sorted)
    # 2. GET FILTER WITH MINIMAL ACTIVATIONS AND DISABLE IT'S WEIGHTS
    key_min, value_min = get_filter_with_minimal_activation(device, all_activations_sum_dict_sorted)
    print('disabling filter ' + str(key_min) + ' with value ' + str(value_min))
    disable_filter(device, model, key_min)  # disable this filter

    name, index = get_parameter_name_and_index_from_activations_dict_key(key_min)
    model_layer = getattr(model, name)
    model_layer_weight = getattr(model_layer, 'weight')
    model_layer_weight.register_hook(outer_hook(device, index))

    # 3. TRAIN MODEL WITHOUT REMOVED FILTER
    print('training model without filter {0}...'.format(key_min))
    model.train()
    for epoch in range(1, 3):
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            #count_zeroed_filters(model.conv1.weight, device)

    return model


def main():
    args = parse_args()
    kwargs = vars(args)

    initialize_globals(args)

    if args.useGPU == 'True' or args.useGPU == 'true':
        print('USE GPU')
        device = torch.device("cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        torch.cuda.manual_seed(7)
    else:
        print('USE CPU')
        device = torch.device("cpu")
        torch.manual_seed(7)

    torch.backends.cudnn.deterministic = True

    """
    from train_with_pruning import count_zeroed_filters_for_model, load_student, \
        initialize_globals as init_globals_pruning, \
        eval_model_and_populate_activations as eval_model_and_populate_activations_1
    init_globals_pruning(args)
    args.resume = 'logs/pruning_10filters_at_once_retrain_for_5epochs_resize_activations40_25_filter_limit70_block_limit70/models/model_without_0_filters.pkl'
    model = load_student(args, device) # ignore optimizer and scheduler
    all_zeroed_filters, all_used_filters = count_zeroed_filters_for_model(model, device)
    print('all zeroed filters: {0}'.format(len(all_zeroed_filters)))
    print('all used filters: {0}'.format(len(all_used_filters)))

    #Path2file = args.dataset
    #test = IrisDataset(filepath=Path2file, split='test',
    #                   transform=transform, **kwargs)
    #testloader = DataLoader(test, batch_size=args.bs,
    #                        shuffle=False, num_workers=args.workers)
    #test_mIoU = eval_model_and_populate_activations_1(args, device, testloader, model)
    #print('model without {0} filters: test mIoU = {1}'.format(len(all_zeroed_filters), test_mIoU))

    """
    print('models loaded correctly...')
    args.bs = 1 # todo
    Path2file = "eyes_tmp"
    print('path to file: ' + str(Path2file))
    train = IrisDataset(filepath=Path2file, split='train',
                        transform=transform, **kwargs)
    print('len: ' + str(train.__len__()))

    trainloader = DataLoader(train, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=False)

    print('datasets made... ' + str(len(trainloader)))

    model = Net(input_channels=1, output_channels=4)
    model = model.to(device)

    previous_model = model

    os.makedirs('logs/TMP', exist_ok=True)
    os.makedirs('logs/TMP/models', exist_ok=True)

    for i in range(3):
        torch.save(model.state_dict(), 'logs/TMP/models/model_without_{}_filters.pkl'.format(i))
        print('model saved to logs/TMP/models/model_without_{}_filters.pkl'.format(i))
        optimizer = torch.optim.Adam(previous_model.parameters(), 0.001)
        criterion = CrossEntropyLoss2d()
        previous_model = remove_filter_and_retrain_model(device, trainloader, model, optimizer, criterion)
        print('model statistics:')
        count_zeroed_filters_for_model(model, device)
        zeroed_parameters_layer_name_dict = get_zeroed_parameters_wrt_layer_for_model(model, device)
        print(zeroed_parameters_layer_name_dict)

    learnable_parameters, all_parameters = count_number_of_learnable_parameters(model, device)
    from torchsummary import summary
    summary(model, input_size=(1, args.height, args.width), batch_size=1)  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    print(model)
    print('learnable parameters: {0}/{1}'.format(learnable_parameters, all_parameters))


def get_parameter_name_and_index_from_activations_dict_key(key):
    assert len(key.split('-')) == 2
    name, index = key.split('-')
    return name, int(index)

def disable_filter(device, model, name_index):
    name, index = get_parameter_name_and_index_from_activations_dict_key(name_index)
    model_layer = getattr(model, name)
    layer_weight = getattr(model_layer, 'weight')
    layer_bias = getattr(model_layer, 'bias')
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

def count_zeroed_filters_for_model(model, device):
    all_zeroed_filters = []
    all_used_filters = []
    for name, param in model.named_parameters():  # always weight and bias
        if name.endswith('_activations_sum'):
            layer_name = name.replace('_activations_sum', '')
            zeroed_filters, used_filters = count_zeroed_filters_for_layer_name(model, layer_name, device)
            all_zeroed_filters = all_zeroed_filters + zeroed_filters
            all_used_filters = all_used_filters + used_filters

    print('zeroed filters ({0}): {1}'.format(len(zeroed_filters), zeroed_filters))
    print('used filters ({0}): {1}'.format(len(used_filters), used_filters))
    return all_zeroed_filters, all_used_filters

def count_zeroed_filters_for_layer_name(model, layer_name, device):
    model_layer = getattr(model, layer_name)
    return count_zeroed_filter_for_layer(model_layer, layer_name, device)


def count_zeroed_filter_for_layer(layer, layer_name, device):
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
        for filter_index, (filter_weight, filter_bias) in enumerate(zip(weight, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_bias, zero_filter_1d):
                filter_name = '{0}-{1}'.format(layer_name, filter_index)
                zeroed_filters.append(filter_name)
                # TODO: na zacetku dodam hook, da se gradient ne bo izracunaval
                #print('registering hook...')
                #weight.register_hook(outer_hook(device, filter_index))
            else:
                filter_name = '{0}-{1}'.format(layer_name, filter_index)
                used_filters.append(filter_name)

        if len(zeroed_filters) > 0:
            print('WARNING: zeroed weights are interpreted as disabled parameters!')
        return zeroed_filters, used_filters

    else:
        assert weight.grad is not None
        for filter_index, (filter_weight, filter_grad, filter_bias) in enumerate(zip(weight, weight.grad, bias)):  # bs
            if torch.equal(filter_weight, zero_filter_3d) and torch.equal(filter_grad, zero_filter_3d) and torch.equal(
                    filter_bias, zero_filter_1d):
                filter_name = '{0}-{1}'.format(layer_name, filter_index)
                zeroed_filters.append(filter_name)
            else:
                filter_name = '{0}-{1}'.format(layer_name, filter_index)
                used_filters.append(filter_name)

        return zeroed_filters, used_filters


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
    """

    n_biases = 0
    n_weights = 0
    n_learnable_biases = 0
    n_learnable_weights = 0
    if hasattr(module, 'weight') and module.weight is not None:  # module is learnable if it has .weight attribute (maybe also bias)
        print('{0} is learnable (has weight)'.format(module))
        if isinstance(module, nn.BatchNorm2d):
            if module.affine:
                assert len(module.bias.shape) == 1
                assert len(module.weight.shape) == 1
                n_biases = module.bias.shape[0]
                n_weights = module.weight.shape[0]
                n_learnable_biases = n_biases
                n_learnable_weights = n_weights

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
        # get each layer and check its instance
        n_biases, n_weights, n_learnable_biases, n_learnable_weights = count_learnable_parameters_for_module(name, module, device)
        all_parameters = all_parameters + n_weights + n_biases
        learnable_parameters = learnable_parameters + n_learnable_weights + n_learnable_biases

    return learnable_parameters, all_parameters


if __name__ == '__main__':
    #main()

    """
    import numpy as np
    import matplotlib.pyplot as plt

    # python test.py --load logs/pruning_training_reference_model_216k/models/dense_net_200.pkl
    reference_model_216_mIoUs = [0.8501279164965186, 0.3545839120606826, 0.863931752520277, 0.450170357368169, 0.8033686196847755, 0.7571687357671024, 0.7974542520542317, 0.8745097694759206, 0.5718282598096079, 0.4531044177916168, 0.7323516730275134, 0.7655525539821784, 0.8591875377266918, 0.8742981245268721, 0.5985536890650309, 0.5195397251949294, 0.8626734184805596, 0.9298132622449858, 0.8532319712024634, 0.8956769014642529, 0.6910271584086443, 0.6725988224591172, 0.9249974958925246, 0.8429443485826618, 0.8654992288276873, 0.6943260981016403, 0.5715087198658373, 0.9088261659203255, 0.9204138476370696, 0.7819822174986264, 0.6843622634672435, 0.7733045615788996, 0.9303763178214813, 0.9104323794261115, 0.8465531089612336, 0.8082625125519154, 0.8868861695866229, 0.9103035020085686, 0.8428062096645843, 0.9306739928417418, 0.7431718435048027, 0.6921190610661065, 0.8983166381204396, 0.8241218295906487, 0.8979587072349624, 0.47331637356065037, 0.5577853984019896, 0.7477655803613793, 0.9170046196120504, 0.9496081806299846, 0.7997454912030177, 0.7946109791809692, 0.8811848183033595, 0.741209469281108, 0.8052705435937013, 0.733340472437016, 0.6781042782876006, 0.7200444579045793, 0.918775186037624, 0.8387295544421831, 0.778300903062016, 0.7927275288490814, 0.686805273812054, 0.8963105970897631, 0.897478545533434, 0.863852709549594, 0.8683909146654296, 0.6128432367700118, 0.6565175226347608, 0.8323823472888469, 0.915745884987911, 0.881156018269315, 0.8797689544833776, 0.8665725352695761, 0.7493235568599005, 0.8607012311343063, 0.8183622941570765, 0.8569021209452851, 0.7573562709998215, 0.7116556607809804, 0.9124359399243189, 0.8586893343203901, 0.5462501182912527, 0.8187720894935189, 0.5928036191504235, 0.8003453430464353, 0.5858889009505086, 0.8714320119604027, 0.8476533046296315, 0.8267787958319367, 0.6465714202695398, 0.7790517236231538, 0.9112316903423223, 0.8759931903375742, 0.8982087342444639, 0.8296046429810955, 0.8479417763719905, 0.6042084235917878, 0.4728947838187175, 0.8616870919465938, 0.9231977401762923, 0.8122965782256528, 0.8129994430754413, 0.7238923446571469, 0.7317604005395493, 0.9011034128098558, 0.8848496821547506, 0.8446752358021112, 0.7215654199105277, 0.7841973599163932, 0.9218594593666872, 0.9004623164549191, 0.80638282499, 0.7190226304267016, 0.8319842933467697, 0.7132810302804438, 0.7844529865585175, 0.8011330258981978, 0.7457010605272156, 0.47897989645573985, 0.8568512730783087, 0.8442601162727159, 0.7098546575609882, 0.7625240885205555, 0.7704552755804359, 0.7066568391074024, 0.8322008675420387, 0.8255817662345275, 0.6027625033948757, 0.6745049497451259, 0.8773955519793546, 0.7407015790424091, 0.6792909249745085, 0.8723890789050283, 0.8173660353975721, 0.8124485724936414, 0.8473679177979108, 0.8516837702459306, 0.7140662659541426, 0.8980924727364568, 0.8542012058688511, 0.8447658803373196, 0.7999130793635147, 0.7872583351545854, 0.9289633687397503, 0.9459370792416693, 0.9349045540396448, 0.6987966177328655, 0.9410335474173567, 0.835183938156722, 0.8914037188469266, 0.7480461028800357, 0.7653043871738398, 0.9434804865882892, 0.8670399375105158, 0.8341900073326222, 0.5458861711116388, 0.6611006677489126, 0.8194824215242815, 0.7377435016083784, 0.6763923916284065, 0.8003204863596847, 0.6716169607294639, 0.9080272619431258, 0.8946878209045525, 0.8740332540369096, 0.6605687669800483, 0.4180622780094586, 0.7758597504819338, 0.7316603791053063, 0.7366592069960325, 0.5869191092052432, 0.7570761593288377, 0.881668964705685, 0.8511141980956276, 0.8907810570001872, 0.8964046284259035, 0.6795968359661886, 0.6749767771860908, 0.8971652548773277, 0.8883511478903254, 0.8704546709312648, 0.4526766854922359, 0.4639495846596526, 0.8653688799005599, 0.9206726310379113, 0.8796359308403288, 0.6272255604055359, 0.8378403238278456, 0.8125589806851452, 0.6892266978617385, 0.7078752816455309, 0.9327722352347255, 0.848720028415513, 0.8770953883807562, 0.6085679405220269, 0.6694647821102236, 0.8533546346484773, 0.8917266104393478, 0.8495921861780361, 0.8006156499155259, 0.7296793371949548, 0.9214997082274088, 0.8814566103277971, 0.9209370846464598, 0.779002331119138, 0.7800344525262208, 0.9076100958543823, 0.8128437204499972, 0.6788795070807453, 0.9207771537894164, 0.7791318625633504, 0.8601010593496228, 0.7698728446010206, 0.5718394898322103, 0.8190059533377293, 0.9067743388726077, 0.7952864166621043, 0.817341592554523, 0.7126108379850171, 0.868455478638197, 0.8391086042952641, 0.7946815805425096, 0.8264864486940722, 0.9140003433411393, 0.876516869520125, 0.9086176719125948, 0.798555076241212, 0.8487676854050393, 0.8625835962050945, 0.8174578357807007, 0.8426441059675496, 0.86140916341655, 0.7411990251392994, 0.7291866431587201, 0.9149577172817447, 0.9160770549028043, 0.9289043233949903, 0.6287875151790898, 0.9289783486934655, 0.9083407277646045, 0.8658383661119031, 0.861231286995954, 0.7823669540028937, 0.6411734457036271, 0.7845172048082621, 0.6683071709554969, 0.8401585926598004, 0.5885051527660584, 0.6127445371694571, 0.8457892368750068, 0.8863303627199094, 0.8473264558311735, 0.7409505964875791, 0.8312622321111777, 0.8944738726142538, 0.8572996781672915, 0.5805923042691552, 0.8513077610468712, 0.7448586098393496, 0.9127478357065922, 0.8079583788117595, 0.8711984554177761, 0.7301800108113515, 0.9011215164142274, 0.33347938875358557, 0.9101191675328013, 0.6967788740354361, 0.8512504316652793, 0.43261522002588243, 0.47668170464390275, 0.8534670151080609, 0.873477371403266, 0.8210401168622707, 0.8829499244278461, 0.8833154634088671, 0.89970955903381, 0.8419415645895036, 0.5532401959046476, 0.7868609747996289, 0.8251236533575471, 0.5590770395097279, 0.8683444591571808, 0.70646584857086, 0.7586511415878798, 0.7755904259631061, 0.7966406094350078, 0.8006823542618511, 0.7754626374576308, 0.7645395720892183, 0.8854497549857121, 0.6552161923521728, 0.8440715422049476, 0.7543123298170212, 0.6684953435810987, 0.7736107895684228, 0.6703418751482138, 0.7831726250782842, 0.740937875738736, 0.6450326258034214, 0.6878537865345027, 0.845330600523062, 0.827337663348679, 0.8621388644197496, 0.8359239658652855, 0.8559002291545295, 0.8709317055642539, 0.7285654361002329, 0.8171814625898343, 0.6848674920142663, 0.7155588687813385, 0.9245792868417493, 0.91321271944688, 0.865090667550447, 0.5651858747179864, 0.5482901058637815, 0.9176521509231963, 0.7299300767576837, 0.8207361917881679, 0.9088352239142017, 0.9068075134204802, 0.8648475464945719, 0.7669069194164504, 0.7351200641827562, 0.8362536676963066, 0.6108951698262872, 0.7815806662210862, 0.4290497230639246, 0.5175360936312292, 0.8328220407115723, 0.6704574065186863, 0.8985856279789981, 0.7678898691437703, 0.8300140418135687, 0.6104455104617166, 0.896303576348424, 0.8731753325421376, 0.7874022817529769, 0.7533400446934604, 0.6928535879004294, 0.8102357198379468, 0.7779655662433367, 0.8778256158840025, 0.7829298063252595, 0.5535199400285686, 0.5448150906602488, 0.9273309545576358, 0.8986632711868989, 0.8073360599020714, 0.882990098328266, 0.6432183645547137]
    reference_model_188_mIoUs = [0.8165344836193812, 0.40947462891943565, 0.8397629562802807, 0.49604722600667733, 0.7540609268407816, 0.7390369943743156, 0.7629952497525682, 0.8355230056191087, 0.5679853071404214, 0.43451301252262836, 0.7944538328894443, 0.6962391162325976, 0.7880632351289251, 0.8185823828869673, 0.5871089813892355, 0.39287014162069384, 0.8545128008000487, 0.9063771716178638, 0.7704887924345047, 0.906295843795675, 0.8098252510929409, 0.5847514355521609, 0.9482267492455434, 0.8439146141877016, 0.9066595444501131, 0.7285656558766759, 0.5973791140519578, 0.8790573504803276, 0.9030535789235433, 0.8468776960088917, 0.4970728861091712, 0.7658099249437363, 0.9512605962561373, 0.9122455042490225, 0.7737439481623808, 0.8285027369325659, 0.8595307970388963, 0.9022569680930275, 0.8568953299037918, 0.932680815769392, 0.7481850141439713, 0.7321994046584522, 0.8695331239361361, 0.8733181036762337, 0.8385862824030428, 0.3698481212991312, 0.6001655571670009, 0.7222755489492583, 0.8261196711685693, 0.9313083671387342, 0.755379226830194, 0.8273484546286208, 0.8827856097960112, 0.723642126652272, 0.721144902302886, 0.7302515107929506, 0.6010773407824206, 0.7099979537612358, 0.8783343311132291, 0.7833335005360336, 0.6998083359994477, 0.8086280501314658, 0.6534615808843109, 0.9096696767271073, 0.8528120821949254, 0.8563601366426041, 0.8522966827465435, 0.6119061464935399, 0.531343862674589, 0.7356353573870521, 0.8831352550291756, 0.88557758767353, 0.8509760386263479, 0.7778075551394974, 0.6559684528217741, 0.8330974678405683, 0.7869396065118001, 0.8701735677423387, 0.7215323964782276, 0.7611425063541541, 0.8875760198479066, 0.8930573424149851, 0.5786846205950219, 0.8861863349762413, 0.5822355058148339, 0.816434191939078, 0.5497107390454619, 0.8407910388314497, 0.8014778754499308, 0.783470789564036, 0.6343062368635497, 0.6443850245596151, 0.9108049399809413, 0.8748928737695001, 0.8692776697460453, 0.7786886622201019, 0.8019546833720523, 0.5593103560717838, 0.4752252036609561, 0.86754869699514, 0.8899434699190574, 0.7692739864217889, 0.766557855862392, 0.7254411396385185, 0.7582026463698291, 0.8491170901035965, 0.8564736939246662, 0.8493226570591027, 0.7115408762795299, 0.7530028987970951, 0.9233159851008289, 0.8966501591051953, 0.8798308021451713, 0.7025254192657416, 0.7579654944614354, 0.6612317301964387, 0.7035415210532708, 0.7314487577164033, 0.7986576403835702, 0.583189229467404, 0.8353152333030973, 0.8618151010969074, 0.7283176986257491, 0.7534402610661042, 0.7817590468531613, 0.8093835065505062, 0.8334473506567572, 0.8267479126604074, 0.5632070574641018, 0.5849099931051914, 0.8017755772498952, 0.7215558787178706, 0.607522068448334, 0.8565635407573841, 0.782912734599166, 0.8133144087013018, 0.8260540315836347, 0.7087567932533958, 0.6708318145241898, 0.8435001716256901, 0.8704499568770787, 0.8130533791849388, 0.7488940380725877, 0.7727487943385687, 0.9050954162359076, 0.9276347638990563, 0.9388694300423723, 0.6734005352859591, 0.9423944005173649, 0.8670493775175256, 0.8560796827886823, 0.7207651176853556, 0.6749496894385264, 0.8953637019297866, 0.8219776853893771, 0.8475122172378406, 0.41333389683147764, 0.6628107327550293, 0.8180452232715989, 0.6399376960379984, 0.6469826768249695, 0.7513993668120059, 0.615424616568256, 0.8606716557842959, 0.8837940846070473, 0.8856900963303885, 0.6572297817668524, 0.4295355187477968, 0.7856639499081125, 0.7425369039552436, 0.7632713084584429, 0.6607059044381377, 0.71092554811477, 0.889195682388155, 0.8808249044096161, 0.8805123396304232, 0.879735585180306, 0.7153898729603709, 0.6743149500404986, 0.9132041716047076, 0.8855081177049964, 0.8451131246095789, 0.3642289048619188, 0.3940101654892785, 0.8781502749571155, 0.9221980969071539, 0.8859218249512035, 0.6903542500729232, 0.8589771506459799, 0.8779368207773031, 0.7553829526164836, 0.734287694310369, 0.9111813947073616, 0.8281312773761844, 0.8988254757438079, 0.5998201275762387, 0.5845492114477525, 0.7586575958560068, 0.8534429927145961, 0.8343546813821044, 0.767263483644709, 0.6647645076406914, 0.913459171466255, 0.8345397718597778, 0.8919145263320722, 0.7643706809358412, 0.6814366092160484, 0.8539137422347723, 0.7596013968125461, 0.6770559155334908, 0.904391913965918, 0.7672418811470009, 0.8574262897082843, 0.6831176182349082, 0.4527028698068072, 0.8158525444585936, 0.9187239449001756, 0.7502622623877097, 0.7779471047858615, 0.7954797757523028, 0.8599700257563152, 0.7988530149379175, 0.8712986637390279, 0.8215110480869698, 0.8996799746829388, 0.8678099729189267, 0.9073971118104994, 0.7648182921975508, 0.8310335576832286, 0.8304426100925948, 0.7565855061103293, 0.797201285008639, 0.8886307262127641, 0.5941812153919842, 0.6559012037119978, 0.9115044639879298, 0.9080825051912125, 0.9226720368784586, 0.5825841494203375, 0.9405902721609289, 0.8766219987076866, 0.812949394233892, 0.7713691855136582, 0.7401420811367384, 0.6734882404248709, 0.7238485692645207, 0.5766649350191753, 0.7503908040360769, 0.5195895862051448, 0.6039017709711634, 0.8284875168791141, 0.8833707279004296, 0.850585258619581, 0.7029345328872117, 0.8550902120156672, 0.8922005080372948, 0.8523859394093518, 0.5925024427869432, 0.812059360150822, 0.8132147169157173, 0.8603954104677627, 0.7625028221587018, 0.8296374370182282, 0.7944139833459054, 0.8750599420685496, 0.32465783923595914, 0.8606944950283526, 0.6866358751765514, 0.851840534453988, 0.39246830074577266, 0.452148519794885, 0.7564609170835364, 0.9045298602512062, 0.8233444100722327, 0.8655649172279959, 0.8177231999535177, 0.8451836447467587, 0.8818038045650665, 0.5175480375720918, 0.7349660822569888, 0.8113960820208642, 0.545129163224768, 0.828593887117529, 0.6424726152176369, 0.6656672158153221, 0.702509362810584, 0.7292885771818172, 0.7929427465787782, 0.6497678648411523, 0.8221048439757677, 0.8524033751635556, 0.7376386643761021, 0.7862453973965607, 0.8140062644391122, 0.6196054073374817, 0.8099548925802025, 0.624812439607682, 0.7768008907523605, 0.8570281052270748, 0.5488962193981952, 0.689623652588663, 0.779510727171608, 0.8461402337501427, 0.8660378939868572, 0.8923439387228705, 0.8789954580873789, 0.8521299524576982, 0.6535961141462603, 0.7718482856702447, 0.6243771605476542, 0.7141630480597292, 0.9074082537766499, 0.9017206887820604, 0.8715113761392994, 0.5181454910543258, 0.43107851938320335, 0.8969696063342479, 0.7899225139027118, 0.7175776653772744, 0.9008194050551321, 0.9064703601136255, 0.887007126642985, 0.7157257772826866, 0.7021029652289463, 0.7892074352299366, 0.6125588024612274, 0.6855403797434869, 0.3656979217115888, 0.5256243328243385, 0.7999506269048715, 0.7530374634070797, 0.8504842015781624, 0.8551117419222379, 0.7977647999353914, 0.5974117493656061, 0.8996887014665611, 0.8564000727350746, 0.768075214397784, 0.6981755376909314, 0.725274069192876, 0.7907129927486252, 0.7605281712530234, 0.8779652075854283, 0.7487393454759069, 0.542044744871354, 0.5970725157556371, 0.9352848741174113, 0.8697073603780974, 0.7832639089050222, 0.8505180849895349, 0.6122673108691626]
    reference_model_139_mIoUs = [0.803080957757194, 0.47104149209311025, 0.8052912788804105, 0.4101354263577273, 0.8072907287621592, 0.7600380421321961, 0.759938134102859, 0.8161346664421805, 0.655857280446545, 0.40705121159741975, 0.781715461685772, 0.6516307171234914, 0.817818261068175, 0.8129370041525942, 0.5988995716243757, 0.4231166824034701, 0.8267304184608445, 0.9072535584501398, 0.7858952945378161, 0.8905142667694388, 0.6958336489624435, 0.6001512112718734, 0.9269749034817982, 0.8579129596641892, 0.8454176770951531, 0.6973644939256776, 0.6078098754602718, 0.8747317342876165, 0.8879461151998514, 0.7920423744295506, 0.5459409879749267, 0.7299641050392697, 0.9094493369980312, 0.8890559130481475, 0.8383666365168182, 0.8165314229812042, 0.8564476221667529, 0.918884615659199, 0.8561802448799681, 0.907273438879737, 0.7298404666359133, 0.7348828497784343, 0.8604247451479932, 0.7854439933950759, 0.8511067901810595, 0.4042711632692269, 0.6528329154149163, 0.7296328542240671, 0.8605039060654182, 0.9258931743836455, 0.737337517211911, 0.800771152993834, 0.8785892727921334, 0.7098891688713826, 0.7727392930617016, 0.7109542745289742, 0.6017830792189407, 0.7219349987774486, 0.8932483844329724, 0.8342093147378172, 0.7226023560483225, 0.7616374511015089, 0.7030163488597032, 0.8269665429404824, 0.8411024014717771, 0.7793248918986206, 0.7861623540277489, 0.5674906754978782, 0.6322804868324516, 0.684899778294219, 0.8875064998214273, 0.7942137879965856, 0.8515445500732681, 0.8523836336059356, 0.7360559699641797, 0.8612886757664727, 0.8019289633487323, 0.8597056538143377, 0.6867978713648593, 0.7359147639875604, 0.9169951452608345, 0.8341037749737978, 0.5495172460278211, 0.8027070367024774, 0.5751629714201957, 0.7938448176868931, 0.5663733635945521, 0.8823289185373919, 0.807352891264853, 0.7384695801122484, 0.6888183961577125, 0.7186303491389662, 0.9183208707989917, 0.8604455820962063, 0.8896151855378339, 0.7722447700571904, 0.7501265587987296, 0.5889732486032099, 0.4180845373972319, 0.8321406068790074, 0.9066129941190928, 0.8135827502985042, 0.8281162070658036, 0.7122447878633302, 0.7490296363100989, 0.8740175598279929, 0.8389971348227006, 0.882233996490099, 0.7672887536510276, 0.763387797155405, 0.9260849054750686, 0.853619941981444, 0.8358356599501584, 0.6991118193536701, 0.7481396921028619, 0.7488850070764012, 0.7655827984549777, 0.7327657889012846, 0.7563291264258786, 0.5821126016661193, 0.8848182737995907, 0.829193483680864, 0.7241980686725691, 0.7816736487269784, 0.7859871341502982, 0.8258991311983258, 0.8599028852950663, 0.8302468869166006, 0.5794843105625606, 0.5718769636701321, 0.836337093100296, 0.706469442147164, 0.5931347287486711, 0.8801691106195978, 0.8239398050213618, 0.8364945144559257, 0.8178746756684034, 0.7776809533984762, 0.7201768585882247, 0.841127353832575, 0.8211592596426609, 0.8463470819479603, 0.7900946140755012, 0.7717542107148849, 0.9109829537106037, 0.9204489855775322, 0.9291908414343146, 0.729296416741167, 0.9200791368415041, 0.8542444093544682, 0.8577262934160266, 0.7803491413735789, 0.689610539585813, 0.8935318534044995, 0.8269946460910224, 0.8379852970115417, 0.5378168453004575, 0.6144939474885802, 0.8255414542965961, 0.669990043355452, 0.6670779039309205, 0.8335754933881341, 0.7139211291505927, 0.8788653173814857, 0.8890389283846798, 0.8681068235189643, 0.6628443679194446, 0.4133526530506606, 0.7725670502816102, 0.7097409954833749, 0.7450216142381425, 0.521195451336755, 0.7548964463334654, 0.8630602889213239, 0.8537773229523585, 0.8844503062730257, 0.9028902023716571, 0.6343611787362226, 0.6390607736694859, 0.8975419144889941, 0.8695999239217735, 0.8580772875894762, 0.39044578301902727, 0.3862870342831555, 0.8698824295471755, 0.9027585595902844, 0.8527211423377248, 0.6484090195260009, 0.8371024382224642, 0.7772991265346215, 0.6734199220480975, 0.6908426587645283, 0.8995726001672436, 0.8691233898180628, 0.8541905670546202, 0.651953679496258, 0.7242585570038881, 0.8424158429138353, 0.821228201337987, 0.7692199518898133, 0.7994882291728277, 0.7006119202371169, 0.912289447030338, 0.7964420965015704, 0.9037324379005813, 0.8271086677569589, 0.7305364768324116, 0.8935308349045, 0.808004251242459, 0.6999494756511982, 0.9346399886343129, 0.7347843166843974, 0.8565849070632205, 0.7391302405428501, 0.4652697749733545, 0.8182057543889787, 0.9216945731795745, 0.8216913356654636, 0.8523214160158055, 0.7281494382688214, 0.8817659127085806, 0.8704720343225111, 0.8227729148195917, 0.7824185746574362, 0.913416481051201, 0.8793400364386816, 0.8737339017361575, 0.8245418892465443, 0.7850915762756546, 0.8487306088563784, 0.8071336306833022, 0.8413031003615462, 0.8452348919777445, 0.6297811548048151, 0.7304486467153459, 0.9202104746807153, 0.8503695848150568, 0.9023460395363057, 0.6125017768063509, 0.9296578533111164, 0.8611805100594823, 0.8237339865107618, 0.8218507028349165, 0.7810095480232919, 0.6547464347530194, 0.7447090705357171, 0.597101309806367, 0.7813707473587866, 0.6024556055810353, 0.5316284510497951, 0.8769947810587748, 0.8720151337912223, 0.8606240208116026, 0.7230427125123162, 0.8453063952033645, 0.9087878426490901, 0.8807323449597706, 0.5261045351227548, 0.8389131054799461, 0.8281615529777164, 0.8634050194479359, 0.7972017140028579, 0.8859853112161605, 0.7414665242214514, 0.901731086156453, 0.3429188014833612, 0.8314787072768663, 0.5925368028202774, 0.8534490508016842, 0.4230695537326968, 0.4375677527961769, 0.7754900116548122, 0.8474263513239345, 0.8581615643170044, 0.8822755320097091, 0.8354774628242491, 0.8825526253004354, 0.8738848254102929, 0.5705709639247305, 0.737192270242135, 0.7928636111590482, 0.5867884972458506, 0.8320077875813324, 0.7617250830546918, 0.7936326457608066, 0.7960619703601554, 0.7616576808894394, 0.7874741493220685, 0.8093735244096363, 0.7492714311365083, 0.9026844234181991, 0.6682587853082352, 0.7849456055195426, 0.8362030613217996, 0.6321454986262635, 0.7315712202676682, 0.6868876324465317, 0.7919529804396546, 0.8077158222922125, 0.5615050449198199, 0.6139584554561469, 0.7946372498618746, 0.7711396357973308, 0.8309434583774995, 0.8152270300907508, 0.8479474439155368, 0.8385160231897371, 0.7850164975015189, 0.8009074390340426, 0.6207980632761239, 0.7216991406193836, 0.9182244600561466, 0.8759972087496333, 0.8453495159614038, 0.49273650243518485, 0.5714634434539335, 0.8783715914569306, 0.7803681053232577, 0.7925236533665794, 0.8949202061711122, 0.894720042901323, 0.8251752246507664, 0.7140034895058636, 0.7176013498303215, 0.8323923690505413, 0.603090823784835, 0.7189303914460876, 0.466555756260517, 0.5435179710370734, 0.8466984265927191, 0.7515788180316834, 0.8464532970831653, 0.7718905088066966, 0.7893094379174631, 0.6985032542642147, 0.8767566359526212, 0.8668991921597545, 0.7940267523123257, 0.7309882804157274, 0.7232057423311833, 0.8147397305595929, 0.7487347696231724, 0.8816377052143027, 0.7619031388761472, 0.5222986840787535, 0.6021548762898257, 0.9281655662521654, 0.8845232248328009, 0.8051375233041035, 0.8754994625626189, 0.5935408502010955]

    prunned_with_distillation_retrain_without_distillation_216_mIoUs = [0.8516697846198333, 0.44617809739568604, 0.8236543012423256, 0.4783099541731487, 0.7944145760182191, 0.7432317886675419, 0.7761232827024287, 0.8674579925546542, 0.5994926753461072, 0.38486304028559437, 0.7938734658929233, 0.7504666630919006, 0.8547222673009782, 0.8716196041388706, 0.579725461151023, 0.4903893528393194, 0.8639869017664907, 0.9314450863593936, 0.8262025558492564, 0.8933167028127251, 0.7195510802198564, 0.6445696672165049, 0.949860494995528, 0.8838707500777392, 0.8907075791578954, 0.6940770695517948, 0.5935933349950446, 0.8991010326169059, 0.8868639205366902, 0.8098422496786755, 0.629506945046214, 0.804815910928991, 0.9014918108600769, 0.8992570689425504, 0.8018493250603607, 0.8157722111215286, 0.8316412659418896, 0.9205517853882133, 0.863536356724204, 0.9142537809098945, 0.7322537710862114, 0.7833351709616473, 0.8914172751298346, 0.8436541149349711, 0.8717253580793004, 0.629799016203436, 0.6587624217458514, 0.8155822196743183, 0.9159717287488973, 0.9345927587974052, 0.7695844736865025, 0.803264830952059, 0.8720237425752793, 0.7686509605470915, 0.7641689583698296, 0.7195978658916848, 0.5770045427159701, 0.7922232157474731, 0.8960826207653315, 0.8253891604222633, 0.767191778686856, 0.8130457542461058, 0.68639757146298, 0.8834903296125263, 0.8833834875827815, 0.8693688844816237, 0.8936020782771462, 0.6510840529434203, 0.6093293170181018, 0.817697302279494, 0.8838291360425856, 0.8503004313243228, 0.8990831749901801, 0.8549802355329639, 0.7596447216383421, 0.8624468048312521, 0.7782794969057291, 0.8656116467665473, 0.7396979150053037, 0.7905398523564757, 0.9050982652238415, 0.8849376262965272, 0.5403287230016693, 0.815280331129009, 0.5876500207249575, 0.8177003041595154, 0.6058737481842719, 0.8797367021387065, 0.8407881033996514, 0.7502193122308786, 0.7006071359918529, 0.8122486015349163, 0.9049798881937914, 0.8768513121495392, 0.8843041839352983, 0.7701231032110403, 0.8241896608552105, 0.5926383036621493, 0.5835621028617234, 0.8537567473672264, 0.850396550771338, 0.8071951025297663, 0.8261771054877897, 0.7282477447491256, 0.7704509703650471, 0.9119057882275123, 0.8892855326073887, 0.9094022894819727, 0.7400175220731215, 0.8269426082509068, 0.9091349997664224, 0.9003631444583668, 0.8614346349980454, 0.7637662909183954, 0.7835051463701495, 0.7018332195811506, 0.7320532647219957, 0.7493200868806301, 0.7899563132636892, 0.5043888711351384, 0.8526966219155258, 0.8593070281029296, 0.7046154138318204, 0.8058989050232127, 0.7888770241194744, 0.7658960583950117, 0.8468906107904645, 0.8407929197794806, 0.5849174030982341, 0.6399216666031182, 0.8778669934787476, 0.6921490766938951, 0.596819531775002, 0.8819723090294161, 0.8315814535904177, 0.822616908982514, 0.868271260964999, 0.8244563062935025, 0.6746502516997018, 0.9125578723675893, 0.8542744204450569, 0.8315707396674535, 0.7887969642382259, 0.806993037547139, 0.9240153830263409, 0.9395581630477532, 0.9370885528191839, 0.7395972208385422, 0.9301283872391575, 0.8807557700349224, 0.904404931664266, 0.6655211787023194, 0.7218321158797733, 0.9167514400269666, 0.8897533126524422, 0.8127458496611798, 0.5063854731849967, 0.6835632159855141, 0.829552984591464, 0.6801665054159042, 0.6436286808150924, 0.7471901016148708, 0.6487411329957992, 0.8920523560046194, 0.8699661276420346, 0.8959457708508534, 0.6496766664864594, 0.48099119607378493, 0.7918069995142831, 0.7863951306166492, 0.7431396488458966, 0.6001693484402356, 0.7932780958905684, 0.8862122799545484, 0.8621202390645285, 0.8706470053912378, 0.9377951337782302, 0.6976438425704399, 0.6998106761200751, 0.9224938898780707, 0.9029806851989421, 0.85860301830779, 0.43061913987712924, 0.533677688402381, 0.8814336016840247, 0.9318510161355281, 0.8816179571008382, 0.7041193202080779, 0.8524185220731206, 0.8358099118953327, 0.7233806663477749, 0.7208028746382631, 0.9028031144942863, 0.8726867371203189, 0.8683137723718425, 0.6530087125800934, 0.6254539431144176, 0.884190198480267, 0.8871556276177658, 0.9037568255974469, 0.7932406018468847, 0.6931218368768994, 0.9016812985467366, 0.8439932749273488, 0.9030185136025786, 0.8305496663397259, 0.7118615530588491, 0.9220737183256728, 0.8421833599596924, 0.6724637000499698, 0.9465399842451525, 0.7657373497741128, 0.8659493681595016, 0.7623569105224559, 0.5192816617896031, 0.875087407508949, 0.9179146275743904, 0.8102058482945328, 0.788668648573869, 0.7951812186553164, 0.8716450276140646, 0.8031169726764633, 0.8288494862855955, 0.8003358056275873, 0.9264974044730255, 0.8973335263109781, 0.9024090051108362, 0.8288624016628636, 0.8218276176154655, 0.8531381929526578, 0.8013416129613794, 0.8539699543600809, 0.8733394532167682, 0.6604625306091457, 0.658721914060647, 0.9171624399716866, 0.903654098292154, 0.9353068423555039, 0.6016273638511687, 0.9432181741616658, 0.9048443290593227, 0.825494259950999, 0.8379854379833898, 0.7647139213890148, 0.7097764844263064, 0.7632554373911367, 0.6584111915423194, 0.7885763408258474, 0.6002606808016155, 0.6420601414634745, 0.8448505102109758, 0.8793857749669503, 0.8592440316995434, 0.730692855629145, 0.8292959977969147, 0.9066247401086112, 0.8627663964864909, 0.5282410247223878, 0.8571612218971959, 0.8069974789017363, 0.9107851386585657, 0.7938787928471535, 0.8909796476076491, 0.742867308561608, 0.8741811539770379, 0.3388503156296611, 0.8882593383547567, 0.652188564347794, 0.8462442321220094, 0.43021814767143673, 0.5666549750181022, 0.7593342933862703, 0.9050540645949521, 0.8774896156875565, 0.9152812172454915, 0.874043956666074, 0.8920996917301998, 0.8706972933278885, 0.5470958244678675, 0.7620702062115616, 0.8268403734092202, 0.4843237190375438, 0.8320610730431215, 0.8021428489793672, 0.7828878421088202, 0.7370078597438082, 0.7347412169439831, 0.8209677156234383, 0.7996947795031689, 0.8131958324343265, 0.8868509130254842, 0.7043993305769977, 0.8064216498124193, 0.8503840729506598, 0.66954140202151, 0.7172874276750083, 0.7343557707752499, 0.7838652893410611, 0.8643964706972787, 0.6280567399603586, 0.7484802904001385, 0.8712093596613697, 0.8384621484904551, 0.8318918195515593, 0.8515558054925189, 0.8756119140093861, 0.8964193698085213, 0.7554167642312645, 0.8565994638842034, 0.6487763680268205, 0.7302804844713854, 0.9457366602386467, 0.8771541435384169, 0.8476300042670608, 0.5834137247034443, 0.5106363209313365, 0.9162152492361894, 0.7731682046433215, 0.8077145824716113, 0.9011862756888576, 0.883478949928327, 0.8942460975635818, 0.6998530490603287, 0.7143181057262898, 0.8204195076058213, 0.6119502439259343, 0.7146008645747483, 0.45450217304245444, 0.6165324032581896, 0.8269067439011121, 0.715268072924869, 0.9074993389835669, 0.7912335274634166, 0.8279671528512222, 0.7590487389008664, 0.9055633536558738, 0.8496008612933477, 0.8046131349140523, 0.7462537355776571, 0.7366372644317117, 0.8092267321210252, 0.7684887564003507, 0.8888970992627353, 0.7748865141655146, 0.5408262616880304, 0.6188847295694636, 0.929884041966761, 0.8935102046616707, 0.8139195922495934, 0.8774213962609119, 0.6535073896233825]
    prunned_with_distillation_retrain_without_distillation_188_mIoUs = [0.7894311895687826, 0.4893765670739733, 0.8632945949964146, 0.4675390640519307, 0.7500719761152177, 0.7530532226148783, 0.7773321185993178, 0.8397787597091447, 0.6256228639502847, 0.43179814423472684, 0.7741940612604435, 0.7517755030266875, 0.8319161541619308, 0.8775836847154539, 0.6004718216132818, 0.46021265569177766, 0.8682190793907667, 0.9072918936940647, 0.7911448238790005, 0.900917906876281, 0.798499366072483, 0.6166283158530985, 0.9442523883086515, 0.9016292224760858, 0.8895882913276182, 0.7112376360446915, 0.5188563942437909, 0.862303937143724, 0.8837809705804455, 0.835824095876217, 0.600799333087886, 0.7371829420450816, 0.9126031921917888, 0.9184682064970375, 0.7896273265947711, 0.8353762149944597, 0.8567831557903227, 0.9141714841763297, 0.8367064145579928, 0.9243687231799202, 0.7669169696952023, 0.7872564048963281, 0.8617852544582506, 0.8641414500544911, 0.8874372913805232, 0.5228951881432908, 0.6384324433252014, 0.7684262351928528, 0.8756191702521177, 0.9293860428251508, 0.6755379867071327, 0.8311519255874849, 0.8730201722113816, 0.7449084012196379, 0.7365557497583032, 0.7269257196663893, 0.6506640164099063, 0.7821457797208408, 0.8954402494085447, 0.7924111233886308, 0.7581975647179378, 0.7551893866081412, 0.7133686648432689, 0.907782150095351, 0.8873781686284364, 0.821891474103708, 0.8630678755527902, 0.6640623439164279, 0.5905567739820794, 0.8134412065647719, 0.9128551647460481, 0.8372848430834379, 0.884150314208469, 0.8571524942621518, 0.7198805776807532, 0.864782328791813, 0.8074570185084948, 0.8709485719671012, 0.7228410320311369, 0.7347930565727184, 0.9163877724417255, 0.8634496642399647, 0.5675167732661884, 0.7985741965283633, 0.6045442176795361, 0.7661466110928578, 0.5937796321105773, 0.8490583970955493, 0.8519430777835955, 0.7705022455451478, 0.5574556448869256, 0.6797100770081448, 0.9175248587967655, 0.8686668200765142, 0.8751686842638968, 0.8220972154604249, 0.7566527833511362, 0.5613493772129593, 0.4375619740282825, 0.8710964776849336, 0.8871340022620681, 0.7964303103202657, 0.8375050161323198, 0.7401720418423039, 0.7506502016213676, 0.8667898579646091, 0.8727159347850885, 0.8832847772175756, 0.7933507573502727, 0.7568548768426069, 0.9289842420622931, 0.8974802412983516, 0.8645897401661118, 0.7660389850087009, 0.7462185642706725, 0.7004071323987568, 0.7122717383295465, 0.7429442897355342, 0.7755478746256015, 0.5099886866971062, 0.8632683205971081, 0.8345567211933681, 0.7354102711337954, 0.7997438131407215, 0.7765021572296804, 0.7516478431282649, 0.8687171020541266, 0.8096636495501394, 0.6099511083896993, 0.5887974671621823, 0.8555329309740302, 0.6826839986773117, 0.5884769441589619, 0.8584971103799109, 0.8203439434673339, 0.831757634890583, 0.8365372820782, 0.8036053794539976, 0.6351373436281024, 0.9014741528981385, 0.8452333157104346, 0.8471086407182936, 0.8082653970460273, 0.7997565124158594, 0.9274993929390964, 0.9279344387125916, 0.9411812592833656, 0.7113920682573679, 0.9190658889860099, 0.8481940429844593, 0.8718415118237104, 0.7696205123526796, 0.740818178498309, 0.9287266055861904, 0.8520210446249157, 0.8546177271878876, 0.53890755558972, 0.6568640313309195, 0.8144278506355915, 0.6905129851231846, 0.6751644093389486, 0.7441221313492494, 0.6400869114992631, 0.8800048237422261, 0.8937643652500299, 0.8660329367964068, 0.6444322437502932, 0.41656851340409695, 0.7512434853546338, 0.7256206529084324, 0.7413745684747733, 0.53336614509341, 0.7416724526104423, 0.8920470866689052, 0.8526033578511653, 0.8975533636190287, 0.8776005902268271, 0.7795001346315034, 0.6663315641299966, 0.9056867789251078, 0.8892772015707734, 0.8646117573927965, 0.40785769915015757, 0.4339421267398376, 0.8925961504619806, 0.9136656344460645, 0.8562421198998527, 0.7166107572466827, 0.8540069907681671, 0.8538447218254526, 0.7364491169921868, 0.7263428664447831, 0.9223283990739003, 0.8481273493143212, 0.8864137165738379, 0.6411284863413091, 0.645318542667219, 0.8698393658023539, 0.8497046152031409, 0.8552786793227101, 0.792004850281907, 0.6891295924074449, 0.8881350683568328, 0.812848764660885, 0.8870777355819105, 0.8218071453142974, 0.742215756273326, 0.8850898779210059, 0.8386998051026997, 0.6859690810869024, 0.9202658861546337, 0.7723229897757975, 0.8847539308715334, 0.7367588403269492, 0.4875007626120854, 0.8913072831655977, 0.9310016279321699, 0.8321956060412616, 0.8022861139207973, 0.7239876729148546, 0.8687743614143232, 0.8617784485994087, 0.830526676286306, 0.8221796008964473, 0.9113148362504602, 0.8875824889925484, 0.8995020206990912, 0.7834115324257092, 0.7509393310150922, 0.8567106496013053, 0.7816157504935478, 0.8266850433464311, 0.880014730806447, 0.6318323219983456, 0.69059456084868, 0.9320263692234888, 0.9126588476539932, 0.9173245224332146, 0.6631555161281495, 0.9459354628648964, 0.8761252490265568, 0.8354082841254488, 0.7788920635636205, 0.7517323866173538, 0.6804417811050744, 0.7458137806083208, 0.6648962474094086, 0.7996567571604221, 0.5312786409155917, 0.690314919183111, 0.8714215332905602, 0.8880968958157214, 0.853095927407753, 0.7298969945124445, 0.8608032256943173, 0.8987278946723054, 0.8878257377997167, 0.5875152894767985, 0.831496289225018, 0.7335137753730223, 0.8901178945435687, 0.8155578162064158, 0.8821772943367752, 0.7657352080814526, 0.8954403912427363, 0.33084428058094695, 0.891114827353577, 0.7016613635550912, 0.848021699974354, 0.44346618793712583, 0.46811313489930245, 0.7493552456277901, 0.8943186526453821, 0.8551432676199573, 0.8957952186954924, 0.8305926589854868, 0.8576048245866459, 0.8924191570935637, 0.5411041885853457, 0.758654288946795, 0.7818343542423503, 0.5392374688613446, 0.8208818047197967, 0.8051335287516924, 0.7780935015372139, 0.7579124288493011, 0.7889515353450629, 0.7860111391696318, 0.717587608369433, 0.7908778895758994, 0.8599015816189481, 0.6071345680792201, 0.7701959945629567, 0.7699258803943518, 0.6492630954725735, 0.8398895975302008, 0.6616440070057419, 0.8329983236313026, 0.7742562315534723, 0.6383628153063815, 0.6909853004268611, 0.8320856581970417, 0.9131323476211968, 0.8423371368870813, 0.8080490075219373, 0.8744335684207936, 0.8585473913043192, 0.7465282534282862, 0.7955509634020266, 0.6181894910331716, 0.7204038894186018, 0.9038409376682561, 0.8953414007125416, 0.82554320268058, 0.48382109048768096, 0.48618152373755924, 0.8919567370475487, 0.772338302223814, 0.8320014401953675, 0.9091078286865145, 0.909251727615368, 0.8351183742066715, 0.6894981981129578, 0.7240781933354902, 0.8075218004119967, 0.6172652567349646, 0.7676607129255633, 0.4574200051171601, 0.4682476482400597, 0.8281897837685306, 0.6987304485628367, 0.8854614004001901, 0.7646496419861561, 0.7654338431321396, 0.7107489768994747, 0.8957713592382313, 0.8486184102230302, 0.7778013028641027, 0.7500393411255852, 0.721517448136791, 0.8065229918660277, 0.7478277429135076, 0.8606081748917644, 0.7736405754440827, 0.6092977622640912, 0.5974105350128726, 0.9332718961699842, 0.8948099982051074, 0.7972691793401998, 0.8742673066597701, 0.6178928148105425]
    prunned_with_distillation_retrain_without_distillation_139_mIoUs = [0.8139680394188571, 0.3865587618584263, 0.8755951746707679, 0.5107331174272401, 0.7788257558121509, 0.7557682616365105, 0.7797187316750698, 0.8445370362594147, 0.5593178811740209, 0.4860875562049138, 0.8156945234249189, 0.7240971355598147, 0.8382651580965803, 0.852850410666695, 0.6231259562223911, 0.4820964017469814, 0.8167652992531067, 0.9029320188331487, 0.7479122870351215, 0.8828450396303095, 0.6668108285040523, 0.6738327756001705, 0.9333374319593439, 0.8866381986802205, 0.8418423155342485, 0.7585887759121994, 0.6194272454268013, 0.8965757431800836, 0.8530634784679055, 0.7883940718648498, 0.6748839598476957, 0.7944175129062014, 0.9010911638339142, 0.8628905917329575, 0.8555137846236732, 0.8199723608160093, 0.8867801543121555, 0.8972017722244994, 0.8627042668048028, 0.8893803593522831, 0.7452476602850613, 0.7582071330798227, 0.8599107948210535, 0.8041849844097215, 0.8332272007715296, 0.5505132513734194, 0.6685137735693957, 0.7200790601075061, 0.9117985572080224, 0.920765462844931, 0.7291167215376476, 0.8160440079134939, 0.877139481571476, 0.7733379749617367, 0.7789136997014338, 0.7353195494231312, 0.5817782388038383, 0.7846604172125613, 0.8919631607970444, 0.8262348563035938, 0.8189447755166734, 0.7928049874388555, 0.7514700908008002, 0.8032780295101687, 0.8356080112508198, 0.8007005831297941, 0.7994262931512659, 0.6303634237983694, 0.6122966977760733, 0.8106446302900583, 0.8632289783800373, 0.829540600043092, 0.884432128500323, 0.8286345960560486, 0.7476825906296632, 0.8683025825627939, 0.7709126130200418, 0.8259592071453629, 0.7273716075412724, 0.7715947402164576, 0.9118233588810325, 0.8037160981454616, 0.5719133246816884, 0.8607789749496826, 0.5992764868710466, 0.8583044371355187, 0.5981077306884933, 0.8281505162215166, 0.8295192453869641, 0.7209455633208123, 0.6186215640792222, 0.7086130544497482, 0.9042849387883444, 0.8750587415603559, 0.9014568399609274, 0.7379304467861583, 0.7133428317350317, 0.5765447892443697, 0.5179817978360212, 0.8353355379366094, 0.9089758447254331, 0.8293591810894793, 0.8849426054820797, 0.7523644547163, 0.7551690030367268, 0.8918814315219008, 0.8859523746406542, 0.8821153225305428, 0.7751539660193476, 0.7847972342805414, 0.9225556708783166, 0.8615180766411923, 0.8458139390495659, 0.7523952658657211, 0.7800985691537361, 0.7136139262174782, 0.7493239304445773, 0.766136517392051, 0.776778214091504, 0.5006497236022228, 0.8859400227840325, 0.8094855381796977, 0.7794595548622816, 0.735113983295629, 0.7520338734521292, 0.740237259296548, 0.8330312442347592, 0.8086937013896969, 0.5828928588394081, 0.6300963247714164, 0.8145809270621949, 0.7190550495290854, 0.5780901205312233, 0.8679351746255344, 0.8378808225093208, 0.8451585453092638, 0.8302146907621435, 0.8439731436043125, 0.7637486651010943, 0.9043275125214986, 0.8116089791652076, 0.8334254333741192, 0.781824696677943, 0.7827465241046129, 0.9245599504510105, 0.9361027433398162, 0.9022362134787749, 0.7152491051853986, 0.892120564710392, 0.7892442598397122, 0.8422700155844618, 0.728683450370671, 0.7383524700303348, 0.9129621091540726, 0.8882957272664064, 0.814203612125066, 0.537737848481566, 0.6477439569912723, 0.8198659825359625, 0.6708802647373233, 0.7030027586232382, 0.812823695088707, 0.6800031599094315, 0.8956827748439262, 0.8986300216929669, 0.8652718617433856, 0.6412866707048216, 0.40458436468082853, 0.7584136256955478, 0.717210437236207, 0.7339084922573371, 0.6398080378220846, 0.7304918413131829, 0.8921655955386901, 0.8315064448721539, 0.889978260640637, 0.9260191255423635, 0.7537455427342198, 0.663472342327981, 0.8953204207871224, 0.8703816536699367, 0.865265569882987, 0.4569111913154891, 0.46775323429440147, 0.87858011368965, 0.9002791979754382, 0.853563050693735, 0.7076256596254333, 0.8154828580822722, 0.7688884636982102, 0.6891949143356307, 0.6970153686944407, 0.924188544329261, 0.8616392225731702, 0.8100737260172681, 0.6704511285385231, 0.7008591628449807, 0.8488842789129638, 0.860698546443824, 0.7369678909410029, 0.7878168509884118, 0.7415593576328094, 0.9014895001407415, 0.8768603728695551, 0.9103280564455412, 0.8287343974668503, 0.7421726749729735, 0.8972993051669181, 0.8309145171614749, 0.677592646197824, 0.9328926174512773, 0.7562124706588861, 0.8336460245667464, 0.7285576225779354, 0.5389691095091345, 0.8251795286138756, 0.8890868903455865, 0.8070950470695791, 0.8582548570492899, 0.7947630215728672, 0.8812213030529246, 0.8285437456299531, 0.7561880145191221, 0.7714197860939555, 0.906418018804816, 0.8942906362803135, 0.8520448830742176, 0.8343936793762634, 0.7924317743597632, 0.8522894611909885, 0.7719596149871865, 0.8111394868816273, 0.8032549998810095, 0.7255637544354844, 0.7532871889373062, 0.935769330859324, 0.880179463395731, 0.8925152689620727, 0.5818149083685465, 0.9297990683000228, 0.8650084186352638, 0.8630996917449876, 0.8122536161819234, 0.782774459065423, 0.7397850314006186, 0.7458499694523143, 0.6791107677258166, 0.8058847292946996, 0.5565197560895262, 0.6500254711269808, 0.8630073666562544, 0.8769355156309723, 0.8413261178677751, 0.6894038379242253, 0.8194235743579927, 0.9026827467724484, 0.8786952480049137, 0.5804693436771866, 0.8352154500584312, 0.7836567114382964, 0.9033872136326568, 0.8001008201887047, 0.8653991351467112, 0.767300330163413, 0.8851267311117375, 0.32990609281527933, 0.8673781562709019, 0.6994644500018837, 0.8658853839202983, 0.4538833033527737, 0.502161721511787, 0.8343515221283253, 0.7885158049337808, 0.8799188169259444, 0.8827160298614454, 0.8444096787892029, 0.8294978657999537, 0.83359255018818, 0.5373139334393848, 0.749469933053919, 0.7735109075530368, 0.5414920912885943, 0.8779019694404124, 0.7326340967944377, 0.8152420097828297, 0.7875856928152355, 0.7877358807580368, 0.8154304623953115, 0.7339909753385014, 0.7982865663152163, 0.9049645775313014, 0.642540822522939, 0.8210843195478822, 0.709129370287128, 0.6767842260440633, 0.7222254711891793, 0.6758141474313185, 0.801981498217792, 0.7004078433022578, 0.62261154668886, 0.695840809716302, 0.8421273522106825, 0.8346518016740919, 0.7672742860156146, 0.8178703168457387, 0.843363939074208, 0.8770083026757453, 0.7850265851057233, 0.8398873527114401, 0.7623106228036477, 0.754257129307511, 0.8943332992191065, 0.8188307752828025, 0.7936642909708078, 0.5361446217065661, 0.6848096681909922, 0.8870635625785104, 0.7524203278034695, 0.8121233589815133, 0.8736978888525321, 0.8784534565110069, 0.8526385034081057, 0.7200084816670427, 0.7489056250845295, 0.8088472068939311, 0.5975578103529615, 0.7571053513063113, 0.532724206266018, 0.5559235344201189, 0.8468578744847216, 0.7045665900674721, 0.8869287842086848, 0.7778816867185762, 0.7756046130271514, 0.6279957702703598, 0.8835462664399061, 0.8716231004816862, 0.8164412172409854, 0.7618179409669763, 0.669186357660954, 0.7930854832929657, 0.7333002866864934, 0.8581155052794848, 0.7877862915293101, 0.5344713417256025, 0.6159233437555673, 0.9213348278243985, 0.8866256517702255, 0.7675572505969467, 0.8931027072317237, 0.6519292657504107]
    prunned_with_distillation_retrain_without_distillation_248_mIoUs = [0.8500254731004455, 0.45225670709890947, 0.8484980734223835, 0.48323434083621664, 0.7395550769906978, 0.7747117623216676, 0.7704345239493327, 0.8594089210412288, 0.6389098157931637, 0.5652963546548289, 0.7835013374858217, 0.7419848614919371, 0.8205339158037634, 0.8580764853058075, 0.5637229047531068, 0.4292048806246636, 0.8615147390300241, 0.9274978705157981, 0.7556435578041657, 0.9013441409298995, 0.751032421399199, 0.6642916659375732, 0.9465587037326388, 0.8827155257786253, 0.889581299690166, 0.7173466610496961, 0.6078082870602748, 0.8943859837283507, 0.921217785128387, 0.853496300615726, 0.5880638973493522, 0.7181860058962928, 0.931100125470701, 0.9252320413833603, 0.821089759730565, 0.845079473699367, 0.8871825712895142, 0.9163582725330702, 0.8816673806811404, 0.9181835942212387, 0.7554254330643023, 0.7234550049765281, 0.8920214412039807, 0.8878474659843913, 0.8754789290774232, 0.5470340496822971, 0.6777375841098809, 0.7853719406745818, 0.9045815617864035, 0.9299236636397639, 0.7193634992602158, 0.8278984279788392, 0.8870337773235525, 0.7926340191902564, 0.7450829818485301, 0.694117921575777, 0.5909797229588424, 0.7792969808604627, 0.8999391725967788, 0.8038780339253232, 0.7557622699975975, 0.7968595821356013, 0.697698698664518, 0.9061171449296427, 0.8726303693874962, 0.8178257531814521, 0.8335435249939677, 0.6159709048857929, 0.6023049661747846, 0.8113715809967655, 0.8736822802389116, 0.9161662621923098, 0.9229294926903826, 0.8714299791450697, 0.7391863402590113, 0.8334326264141074, 0.7924705306646123, 0.8734257944206743, 0.6985294044218724, 0.7030672518386967, 0.922287646514742, 0.8768753616105724, 0.5911378813602773, 0.8180955974843837, 0.6238906090504817, 0.7472413009096077, 0.5914096422613514, 0.8729220785276681, 0.8490867405323794, 0.7783314630365683, 0.7308754716767711, 0.7232977815452475, 0.9113125818885692, 0.8652356961553271, 0.9029024675389113, 0.8698085450910159, 0.815487973199259, 0.5752054412225283, 0.5136431258680966, 0.8480598308135372, 0.8996984817088213, 0.768960904593792, 0.7943289890685212, 0.7415699954259607, 0.7671107629533143, 0.8720245357953977, 0.8948758579826204, 0.9003936703400494, 0.7317643034777518, 0.8000301200706526, 0.9134858513717393, 0.8873051627489652, 0.8824173311689714, 0.6978686578805643, 0.7311211027765543, 0.6976497968559204, 0.7404462168046249, 0.7449396637370104, 0.7556526257967588, 0.5735671588557456, 0.8676561429597351, 0.8744107973047357, 0.772688234512923, 0.8283436560709049, 0.8066730806541746, 0.8095086184392051, 0.8396454766166589, 0.8247514259889082, 0.55920217550104, 0.6398964117227488, 0.8800779299596122, 0.7322986119550662, 0.5948361186930968, 0.8672651296789464, 0.8185914717349855, 0.8587598046917168, 0.8411335298339853, 0.8416106073591385, 0.6501100070773808, 0.9167856299859447, 0.869681413024002, 0.8419731100010694, 0.7599786527972894, 0.7986234175590418, 0.9145539677851039, 0.9323727877068154, 0.9428441952333781, 0.6827315933872499, 0.9491785395359875, 0.8347150978299119, 0.9002496276215415, 0.7711796591612243, 0.6979831161081671, 0.9243566078318889, 0.8845503459334478, 0.8593135103568394, 0.5181352640797351, 0.6276577940662657, 0.8156830742960484, 0.6567670558714925, 0.682896521518625, 0.7477782716093523, 0.6343632416892917, 0.8844956453409984, 0.8934363647447852, 0.8931049507705788, 0.6607114787438607, 0.5241669111391246, 0.7266146116735406, 0.6959711072856787, 0.7593877590476211, 0.5955051509476739, 0.744188060895581, 0.9052515111542212, 0.850375817764878, 0.8830354036186805, 0.9367329769191322, 0.7234710568603695, 0.6657417401926814, 0.9161233409818846, 0.8986981892262246, 0.8609403821221444, 0.33639885390843105, 0.4098187376990983, 0.8847266870784192, 0.9243120429757168, 0.8726740318796007, 0.7400141984945088, 0.8363917136596377, 0.8592014009976561, 0.6966150933684405, 0.7498789318725998, 0.9069147522642175, 0.8476947805565856, 0.9112389575304524, 0.6568311686824906, 0.6908465836661849, 0.8242401307036532, 0.876871746737393, 0.8842096470421399, 0.7912360046747614, 0.6865890075727337, 0.9056791745005882, 0.8686983083522565, 0.8725920219062109, 0.8283363633303535, 0.7878640981708234, 0.8927044568664564, 0.8669857408017174, 0.6251487807555458, 0.9191399246166319, 0.7714968113215137, 0.8944436859977644, 0.7851802525276901, 0.5855918754851673, 0.8599320928329144, 0.9085517754344392, 0.7862607754602904, 0.7822476965271503, 0.7307639541079466, 0.8707920781924865, 0.8432184813603592, 0.8100079191809822, 0.8042132271883373, 0.9060635279736949, 0.9014991131316544, 0.9049635760451352, 0.7831923943185688, 0.8683573188446254, 0.8620768323145909, 0.7611098438367989, 0.8247554668715333, 0.8844094978610332, 0.6808857714272593, 0.7044065376391143, 0.9359033514598886, 0.8903642806669145, 0.9343578884199435, 0.6097787888429944, 0.9348056564138412, 0.8899200838364633, 0.8331561089843916, 0.8341960124672719, 0.7675977431715939, 0.6928600815798003, 0.7473880569165908, 0.6530701999118409, 0.7887456724782255, 0.5964768866564792, 0.6780539359035747, 0.8334356498764895, 0.8988910232881666, 0.8438891433799323, 0.6809708866369324, 0.8526550414959586, 0.9107276594327307, 0.8426580525387908, 0.5388939133469896, 0.8328838060254683, 0.8157482084065634, 0.8875744582645277, 0.7880761620369646, 0.8729845940237316, 0.7568039963971822, 0.8936173410275294, 0.3450003001949249, 0.8778111592567972, 0.7633014366360849, 0.8777100751070919, 0.47226752767259417, 0.545266804476726, 0.721209319812819, 0.8960664978076689, 0.8591038655931964, 0.9179875505557262, 0.8367100436505731, 0.8563620174788118, 0.8455176097368899, 0.5133625512911234, 0.7454349851157809, 0.8344881949186129, 0.49378327418710244, 0.8729414823833921, 0.7958584879903708, 0.8568765200607501, 0.7642540516042173, 0.7932030069493772, 0.8198807495690597, 0.7863319255411961, 0.7602864441214168, 0.8941986772620051, 0.720194113579858, 0.7096271202106724, 0.8346233212958651, 0.6473953359439173, 0.7145556498374913, 0.6451330145161135, 0.8140189922871197, 0.7978210593343112, 0.6668424482150642, 0.7332394562279696, 0.8346070492857771, 0.8484774397871665, 0.8335413422671991, 0.854046242965753, 0.834750027469577, 0.8496741959657372, 0.7050516321891896, 0.8472169410472034, 0.6500971560673595, 0.7622805133662293, 0.9128505076868274, 0.8945197740679398, 0.8643348016915829, 0.544192751089034, 0.6362601465624271, 0.9140324942644726, 0.7461525573071529, 0.686538837800638, 0.9010440470581536, 0.9092970358497099, 0.8990971953280831, 0.7041958284477735, 0.7302853162471523, 0.8211134423455975, 0.6508802903873704, 0.7959964979313433, 0.4894249732236093, 0.5971210137352041, 0.811851332029732, 0.7028410708287198, 0.8692659298500753, 0.781067207170324, 0.8017995662068521, 0.6464332575007371, 0.8592633465278438, 0.8711590255064938, 0.7940467317721622, 0.7415380657876448, 0.6994730708929762, 0.7853597000446522, 0.7653594531805733, 0.8713834103107684, 0.7670928070657216, 0.5605415901525914, 0.5818863117522135, 0.9293970927877311, 0.887198562869592, 0.808727865812182, 0.8564028268502138, 0.612501388042979]

    prunned_without_distillation_139_mIoUs = [0.8204326883450288, 0.3775493811297503, 0.8737774511740612, 0.5403619440348973, 0.8359335174188647, 0.7666194277122034, 0.7771732862462665, 0.8292735277437149, 0.5023532105263369, 0.36594399817495266, 0.7261327781684311, 0.7109367002631082, 0.7920398350045557, 0.8487296922517658, 0.5948436714851647, 0.4526873482995932, 0.8391511746600709, 0.9245017255603024, 0.828396254028348, 0.8939556495481055, 0.7053451172569565, 0.6400628158765413, 0.9319990677473838, 0.8809800505065578, 0.830409186680935, 0.7149397305205508, 0.5830394016405063, 0.8812584459528701, 0.8847511373922841, 0.841551937395829, 0.6806012341798424, 0.7915913591921052, 0.916890258274181, 0.8894452146827864, 0.8342327184286972, 0.8426558729509402, 0.8636861754793458, 0.9052989379399541, 0.8248627116324836, 0.8983676238830701, 0.7595760600991642, 0.7170099656201631, 0.8860399868876474, 0.8370461033732158, 0.8332681252002344, 0.521644396333759, 0.6210669608647046, 0.6962734761134133, 0.8638517171754109, 0.9295811626119908, 0.7411568972842821, 0.815848525392509, 0.8677459470737732, 0.7443876854092979, 0.6999874002860013, 0.7330213386529469, 0.7064082614024014, 0.8180889151780423, 0.8952205631949095, 0.8438082935796033, 0.7811576559653881, 0.7488848644411551, 0.7259280535723182, 0.8859143922611046, 0.8912084381350516, 0.8032834547053584, 0.8030870215346387, 0.63535814523325, 0.606462005775433, 0.7626116594557582, 0.8680991304545759, 0.8744268939192305, 0.8552406746203631, 0.8109410810335309, 0.7232178416140655, 0.8768279242603527, 0.8148297354120522, 0.8023568141835284, 0.709946609970788, 0.7379094940285374, 0.9167696967219592, 0.7806625673346544, 0.5210938896635492, 0.8482023348082918, 0.6043027489411226, 0.8561873166898067, 0.6026736388724145, 0.8266588677318731, 0.8181142634654939, 0.7548818610775117, 0.7531297999111388, 0.7493857406090851, 0.8568368096652179, 0.8867603156900236, 0.9079844896884701, 0.7686151971357151, 0.7541319103284527, 0.581506495131807, 0.47641587850592193, 0.843090996073154, 0.9077876528512291, 0.8047477400179348, 0.8231885480088058, 0.7290678959073107, 0.7711901561274974, 0.8744815237604479, 0.8859864722686116, 0.868662488350973, 0.7698071007009092, 0.7639781597386267, 0.9311569776548563, 0.8949156588852832, 0.8380157937872175, 0.7632771502129402, 0.773062913360465, 0.6906974316929448, 0.8053181552097524, 0.7629522644273515, 0.7628305012491655, 0.5115809464139348, 0.8678190199561748, 0.8300750575205886, 0.6713165187961562, 0.7308678940275791, 0.7572020453299405, 0.7317801072196307, 0.8213578333146492, 0.8071279099410376, 0.5809792513960099, 0.5979667787665227, 0.8248836267546433, 0.771923672759968, 0.629933969507271, 0.8784017042276919, 0.8398909071215942, 0.8403755877142801, 0.8665289544436043, 0.8272942712714548, 0.6981723068630593, 0.9061042048129867, 0.8138871699346516, 0.8436098846996505, 0.8183786493956932, 0.7993178888762051, 0.9305259357212485, 0.9274926521660021, 0.9228386084634066, 0.7128062771895539, 0.9120478099907596, 0.8226690255152729, 0.86727026386169, 0.7450959367435943, 0.7216889249023317, 0.9160127760599137, 0.8259600360441303, 0.8028754514638514, 0.478571118292938, 0.6386121168133587, 0.8453688228874341, 0.720725304105786, 0.6937837656475323, 0.8337815369456165, 0.6759819129982173, 0.8889873883318247, 0.8963274970252282, 0.8583464336461237, 0.6436233097766985, 0.4436783759770157, 0.7278424697179781, 0.6165209167118256, 0.7299603003332678, 0.5460889002081105, 0.7173473239935806, 0.8872978390434917, 0.8448740295413174, 0.876327216468267, 0.9325954609575674, 0.6852782579002128, 0.6862801372411872, 0.901649301551491, 0.8979053462137583, 0.8615709357935373, 0.3500943283073503, 0.4340727623806103, 0.8812149842814566, 0.8926589850561433, 0.8587100120054921, 0.6684775119570269, 0.8283627303423718, 0.7902128919264474, 0.6810131693719599, 0.6903842133668006, 0.8668387951031469, 0.8373824641619299, 0.812835466550267, 0.6661960532487723, 0.690915737236648, 0.9126444036482129, 0.9031311965685229, 0.7508598592185035, 0.8049563228460725, 0.723851054976739, 0.9119912111886453, 0.8669672664778337, 0.917607641236878, 0.8175787177614604, 0.7596301420237462, 0.891969224460403, 0.7692025447156866, 0.6834116095536082, 0.9310317432459523, 0.7528400876774685, 0.8542624829702128, 0.7181888254648973, 0.5175324707439284, 0.8182227425231267, 0.8895483373022788, 0.8182623106107153, 0.8271669938505332, 0.7308235751283044, 0.879655691550522, 0.810835127039784, 0.7644788348258722, 0.7641355086414542, 0.894878746256804, 0.9038691203187429, 0.8859637343467394, 0.8417601021580097, 0.7910168196385556, 0.8661644028525953, 0.7707915308327945, 0.8221317032457012, 0.842235225398154, 0.6933241100073482, 0.7105868658867105, 0.8928594444224639, 0.8967058370269181, 0.9214951697689676, 0.5995708528913154, 0.9364654486902599, 0.8646594317405852, 0.8802232916267304, 0.8325561072162226, 0.7748923497169008, 0.6660823376878315, 0.7546838721773528, 0.6866374434018809, 0.8183421334756803, 0.5755233212277466, 0.6642005925232295, 0.9000906864333527, 0.9066861309463364, 0.8358148151096696, 0.7273209762966026, 0.8285218419724448, 0.9139589136029198, 0.8388786592049104, 0.5870581761218792, 0.8498125125715105, 0.7656962241318697, 0.9054331264448043, 0.7899944604941793, 0.8695467402366804, 0.7498436167422038, 0.8865472224409345, 0.33316122327996783, 0.9003492738804683, 0.7291411485642874, 0.8416391548017098, 0.4346538490243827, 0.5457840572242328, 0.8052062507597001, 0.839247880526059, 0.8464103160114271, 0.9030134132249295, 0.8692617809495159, 0.8531471066520157, 0.8387798792263828, 0.5718741138449129, 0.7568124186252083, 0.7996438272852195, 0.5032664445334117, 0.8645743020664858, 0.7203444794720928, 0.7953250786823038, 0.7853588934254487, 0.7921965419910568, 0.7804044139956963, 0.7684564251853734, 0.766906583968307, 0.8713310717071037, 0.7372073555125946, 0.8143953268245857, 0.7820914855872902, 0.7359914894512242, 0.7518103998192318, 0.6900398197427697, 0.8035700263781214, 0.7750686755900164, 0.628892732218728, 0.6572384752660108, 0.849260398825703, 0.8367027854609852, 0.8364859871764371, 0.7971002057918168, 0.8533265921668077, 0.8362748170889942, 0.724025027882353, 0.8131948161089344, 0.7407853699980916, 0.7346183500700433, 0.9179792207535846, 0.854461429569876, 0.8353003031060147, 0.6024015319658159, 0.6511082340773414, 0.893617645319942, 0.7420618790118721, 0.8447709982631745, 0.8994541748504064, 0.8903806096621083, 0.8472811530006759, 0.699871990545855, 0.737929458338867, 0.8151851306721039, 0.6404131543296426, 0.789373555270026, 0.4150012442671475, 0.47969183933886433, 0.8326582676632166, 0.7620850203054294, 0.8983644540775104, 0.7803109350901178, 0.8203316058640706, 0.7342808353307899, 0.8722503956427787, 0.8510475749333888, 0.7981591525487316, 0.693495461956414, 0.7475870026109102, 0.7833158643876668, 0.7390838436945497, 0.8613016753299655, 0.7788554471109929, 0.5699969273981644, 0.5466845785667487, 0.9177852131695309, 0.8683208635357316, 0.7913126538094266, 0.8753422724936897, 0.6277070224897187]
    prunned_without_distillation_188_mIoUs = [0.7624094400846674, 0.4131760577822034, 0.8125440782474209, 0.5295650547434464, 0.7721264223788953, 0.7150119470603746, 0.7660081855071976, 0.874336333373261, 0.6071696793633857, 0.6312082205810204, 0.7330842164311203, 0.7194155720448329, 0.8394184238720143, 0.8836924761856891, 0.6120044095993156, 0.43750825308483, 0.8641551641581308, 0.9359846182077503, 0.8702343485884431, 0.9086810873280411, 0.765328074567891, 0.6907349006779926, 0.9314552808258016, 0.8762815002485767, 0.8556352360440109, 0.7389452987335308, 0.5113549488425647, 0.8957696722130472, 0.8630982876769943, 0.8025640561689111, 0.6424608642101246, 0.7594182697713052, 0.9225282855773049, 0.8911216084333451, 0.796014477022968, 0.8102130698724084, 0.883536545740012, 0.9168833562955393, 0.8332948460651814, 0.9124514137994963, 0.7447410787695463, 0.7164080329123778, 0.8913148532106396, 0.874468529651716, 0.8312531095378577, 0.5516922836545622, 0.6836594815099252, 0.726777247799232, 0.8926532151510495, 0.9307810689946645, 0.7694031304204787, 0.8435094989881053, 0.8771863483676224, 0.7325548067025536, 0.7329769442302158, 0.7220641269202759, 0.49455366104593884, 0.7819716740525199, 0.8921811926775876, 0.8178393873502576, 0.7793382668863624, 0.8169295274905941, 0.6942541094497116, 0.9055791715642726, 0.8841329978506628, 0.8115448647526575, 0.8071120019884643, 0.6525350056368, 0.6546947514610504, 0.8286886180774804, 0.8023998577742649, 0.8444173163222937, 0.8777060103345852, 0.8381600255930859, 0.6964645747958104, 0.8482979036434762, 0.7779083449959848, 0.864720546934609, 0.7621279855911802, 0.782951713476393, 0.9054517430526122, 0.856003139721069, 0.6413444368087858, 0.7934716350331984, 0.6019479302979266, 0.8853132411716976, 0.5596624285307432, 0.8349731275819301, 0.8385265151700938, 0.7327718831647616, 0.7249474881017056, 0.6864927699970661, 0.8939420708675845, 0.8824021479817231, 0.9111114799259779, 0.7777036483334607, 0.8611230958633267, 0.5659514997193656, 0.5124078894503888, 0.8672029711420909, 0.9141406726674886, 0.7741744282642405, 0.8130938536571068, 0.7552519548608401, 0.7820601203141002, 0.9021557436510402, 0.8939948865904703, 0.8943470610098758, 0.7673792954510325, 0.7862705200538099, 0.9126305311845411, 0.880755309910535, 0.7888374399177276, 0.7281658384449875, 0.6917602354494671, 0.7272525215280483, 0.7453557819402753, 0.7442791210464238, 0.7610964199656045, 0.5480332570587061, 0.9077703393987366, 0.8619993649322313, 0.752250911746049, 0.7894311941524393, 0.8102615825309399, 0.7608188664298812, 0.8486104474109153, 0.8407861115770666, 0.5652482237810554, 0.6491738650081018, 0.8274836570536463, 0.6495040917747494, 0.5557563549146223, 0.8800939388584743, 0.8198013345078448, 0.8397437421030619, 0.8184340488827305, 0.8030492069365099, 0.6219303318698365, 0.9117714857957722, 0.8490115352533639, 0.8212865464162281, 0.7528844196591564, 0.7671285920245248, 0.9214221464038876, 0.9313504497992559, 0.9388622755762774, 0.7026348144662328, 0.9159936555357326, 0.8076404633467013, 0.8875727454037634, 0.7725905026589541, 0.7087002291670197, 0.8885325852383825, 0.8811458634662565, 0.8040514700332725, 0.4846145720429077, 0.6635120214527662, 0.8166314369151413, 0.6445568883415173, 0.6367405864382275, 0.7380135345790437, 0.6274888731078992, 0.88957757361584, 0.8846114315810195, 0.8798962920851621, 0.6260926700350612, 0.4289868464716895, 0.7498657217330092, 0.7124839990880004, 0.7744875568146234, 0.6240528525801875, 0.6957094385852571, 0.8878246642180962, 0.8786363923618841, 0.8835651787297902, 0.9303999356016404, 0.7653320447619634, 0.6593574213023911, 0.9099758161075676, 0.8905752838737395, 0.8682320440227194, 0.2583876909535764, 0.4599750286533646, 0.8795868313831304, 0.8893389688075571, 0.8728913761378383, 0.7123155142128106, 0.8595633215964193, 0.8219818065266071, 0.7329954837994075, 0.6973740038179342, 0.903234128563702, 0.8525797638766949, 0.8558348441797485, 0.6107037315170215, 0.6563426386347456, 0.7700129299686721, 0.884428448801356, 0.8558255250044846, 0.7747766058513115, 0.7224491153680046, 0.9098661068531236, 0.8481742224848879, 0.9113373236250859, 0.8024928002632028, 0.7431507715779598, 0.8729451610458959, 0.849985322230388, 0.6870187967434443, 0.9308469624261988, 0.7859768501765532, 0.8501813632419455, 0.7083425268289897, 0.5748086405932344, 0.8433868989361678, 0.9201554680175845, 0.805995089639804, 0.8081138305879049, 0.7392348310523651, 0.8716319607244521, 0.8554984601692212, 0.797328917970465, 0.7638320565583196, 0.9216807821198363, 0.8938267567156231, 0.8985949559360212, 0.8234547786352856, 0.8500858384535124, 0.8669782092593475, 0.7585165452727383, 0.8156399929421426, 0.8507870908241855, 0.6637968610204795, 0.7147181558946447, 0.9050865817561841, 0.8896124824383801, 0.9231803152779131, 0.6555864576303561, 0.9440747221360017, 0.89997830284393, 0.85125032743923, 0.8081128985934076, 0.7521685731914293, 0.7072598946489154, 0.7118494064168386, 0.6329988514510184, 0.7880836797629671, 0.5730165644372727, 0.6657569604583192, 0.8608453053113974, 0.9061615498780471, 0.8520674925100017, 0.6696725343189134, 0.8467805845282064, 0.9284598923635126, 0.8683422026613523, 0.6071999903057534, 0.8379476754546173, 0.7386566707106788, 0.9038229692726834, 0.8016996019229363, 0.8589997388767256, 0.7326630249139193, 0.8995474752574407, 0.34952838772639877, 0.8842418743443177, 0.7392656108823483, 0.8400481829483732, 0.47569765639339473, 0.5629454107284727, 0.7741494054140657, 0.8409878289331162, 0.831103641976426, 0.8941768420661012, 0.8764887712558249, 0.8854383167814921, 0.8228794148918558, 0.5150785813687724, 0.7538019434503552, 0.8085758847024392, 0.5073688558684831, 0.8403767578351204, 0.7310118007786507, 0.8127873881587194, 0.7530804771532157, 0.7496220629372272, 0.7935993566826907, 0.7917560319971428, 0.7912455327421456, 0.8949131270794984, 0.6544840438197383, 0.8116016447853123, 0.7959215727696027, 0.6644296103150624, 0.8057050727746001, 0.7200854173378083, 0.8123979059170774, 0.7630892370494927, 0.6539294119646502, 0.7060884329252791, 0.8168599408824206, 0.84245807016634, 0.8245138063643525, 0.9048286282774003, 0.8463964647658067, 0.8117080742190906, 0.7502116672531141, 0.8816125790220191, 0.6418950975075276, 0.7188223687118728, 0.9261059288542214, 0.8792511061121342, 0.8226359390136002, 0.5109640245795877, 0.46593173512223446, 0.8957582032777662, 0.7944117466435072, 0.8597605670768497, 0.8975001803212758, 0.8880325449762074, 0.8885848645126064, 0.7014053373813802, 0.7316811360809784, 0.7915300935323811, 0.6141482290654044, 0.7210157149635139, 0.48717616264732083, 0.5014523695073684, 0.8289399578115648, 0.6911957247919308, 0.906335466322486, 0.7777634955533346, 0.7970618250279, 0.6349080062904207, 0.8921331445178574, 0.8690448503278194, 0.7914838327123893, 0.7396610152894844, 0.6773637344490525, 0.7864561368833619, 0.7704396196051215, 0.8699701496001676, 0.7674634253036321, 0.5330763150736632, 0.6164994574064971, 0.9193477068461458, 0.8718836890469106, 0.7992953790498425, 0.8591298271776933, 0.6286724118868785]
    prunned_without_distillation_216_mIoUs = [0.8508003815276441, 0.4391330781195371, 0.8487128551984314, 0.5130035349453687, 0.8186347082322784, 0.77209450921614, 0.7678309133929295, 0.809551771241405, 0.5627245650560926, 0.5321474634788379, 0.7544863145113208, 0.7396981866258245, 0.8343687651728773, 0.8450618417323659, 0.556188720582209, 0.42709986612307993, 0.8495837720523302, 0.9274037762167062, 0.7732500800783028, 0.8688280587347981, 0.789474738948376, 0.6367594760741734, 0.9032341784376134, 0.8932180095697431, 0.8649798330268701, 0.7446822976339227, 0.5748704781254829, 0.9072901400782013, 0.9187003690252762, 0.8218806707657466, 0.6794157526010409, 0.7823250042405503, 0.921283519039593, 0.9134999448620138, 0.8465586781536663, 0.8287883651300003, 0.8662442673892301, 0.9195499214142339, 0.8675385778342919, 0.9195800385645951, 0.7344428001427583, 0.804010521434467, 0.8894354857818381, 0.7832117305774989, 0.8671611252103397, 0.5116644319360512, 0.6422769627155358, 0.7382614956109836, 0.8937919922305749, 0.931461250810683, 0.7683093458588269, 0.8191783473187597, 0.8903623739436519, 0.7775296422810105, 0.8471083717677512, 0.6900382278844648, 0.6018460240648607, 0.8306585946337528, 0.9053548589033604, 0.8137590663214008, 0.7872740784035716, 0.8161580062152514, 0.7394735104545134, 0.8311599774051877, 0.8501295625715188, 0.8500931508787649, 0.8458782872889465, 0.6719188919249608, 0.6048385933545918, 0.7763056420439223, 0.8720511600150301, 0.8323762277768154, 0.8679356040489373, 0.8391309508936299, 0.7554561403928358, 0.8560757868528596, 0.8139118798863535, 0.8615686801877701, 0.7311836674308443, 0.7323294042537418, 0.9122731245809781, 0.8182803124653177, 0.5795433290709986, 0.8053150385723015, 0.6053892739612898, 0.8461882793715657, 0.5823226241706667, 0.8814460607866369, 0.8582604850658423, 0.8033774292496781, 0.6437749316528756, 0.6781148927493423, 0.8898563994711991, 0.8761469346705724, 0.904651174499061, 0.8238734803390371, 0.8153780227917813, 0.5912517994863582, 0.5047119149060885, 0.8613920984618517, 0.8878910332895052, 0.8068689353781048, 0.8300840284730053, 0.7704413225382737, 0.7417649755687444, 0.8983458373481197, 0.9033562574067555, 0.8869462037834586, 0.7677510863797831, 0.8167494986178444, 0.9252404728232123, 0.8925750322856253, 0.7762558020163133, 0.7215840151217147, 0.6963144635695426, 0.695221483926295, 0.7512790784426635, 0.7796947448350587, 0.7929989902836118, 0.6304715164441513, 0.8630148686957385, 0.8533101238158612, 0.7200742181279697, 0.7933278480037174, 0.8027935140231772, 0.7300245344445622, 0.8392670729488124, 0.8379970694292648, 0.6041015178402573, 0.6558685020688315, 0.8465459621579923, 0.7349391132105312, 0.6064056507244915, 0.8876136657284629, 0.8277507326601966, 0.7883223107709995, 0.8408293021746573, 0.8186746237262775, 0.7047895701424719, 0.889380559985439, 0.8289186549849452, 0.7969504117871811, 0.8132282098826855, 0.8332218442419613, 0.932318483539336, 0.9426072738402349, 0.9240673853597089, 0.7061213566496696, 0.9308175406058488, 0.7927704667031105, 0.8462804958747887, 0.7807715418374923, 0.6911550084867921, 0.8952075127806167, 0.8685647551571443, 0.7544577343319935, 0.4989229558307264, 0.6290099639656715, 0.8034642789970786, 0.7094300318442229, 0.6629775773337359, 0.7400493971300333, 0.6616456937896211, 0.8929917716576552, 0.8878214488801286, 0.8943019500630692, 0.6648822395094253, 0.4402772314981299, 0.7845149498507943, 0.6725963569517432, 0.7361731400270302, 0.5729267296899335, 0.7247448866057222, 0.8886774411439157, 0.844544367043534, 0.8921296233439359, 0.9238243150286577, 0.7660963930560872, 0.6791237013771835, 0.9134374342981755, 0.902796413460997, 0.8690073365100999, 0.3873374460212143, 0.4415936914221667, 0.8695405214757685, 0.8975954832034763, 0.8737919671828587, 0.6965259712847938, 0.8373132713510649, 0.8523206606997403, 0.7157437681939225, 0.7241099645717679, 0.9123316744661745, 0.8809505869315408, 0.8894213725081961, 0.6751599104535345, 0.6722494915296006, 0.7597526666919544, 0.8849297877033923, 0.8259192929004681, 0.8076180836026712, 0.7452347757735747, 0.9094276313899115, 0.8492035896825332, 0.8963751258360024, 0.8156432073438489, 0.763854309607275, 0.9029400742151995, 0.8352182120481934, 0.7136496631798748, 0.9307534658616279, 0.7583181457377287, 0.8645929044007532, 0.759080840089797, 0.6191574910012315, 0.8425112080009836, 0.9241014546616564, 0.82655446812122, 0.8141989522026079, 0.8093638555524945, 0.8602115615833525, 0.8012551631294135, 0.8138880182668121, 0.7984810995988745, 0.8981671014420369, 0.9080151975599127, 0.9012898315884902, 0.7985648362163945, 0.8213686817984107, 0.8539517629549189, 0.8153797010895347, 0.8309693162665637, 0.8612931263697242, 0.7172544680335803, 0.711497401772763, 0.9278436446992064, 0.9155509609031964, 0.9330557957918928, 0.6275333995594665, 0.936876482159773, 0.8876871677832221, 0.8570893468880084, 0.8480702760724816, 0.7742838711549506, 0.6861121437613926, 0.6887547392503244, 0.6692064492240788, 0.8178794233122255, 0.5350680134509463, 0.6029015673012477, 0.881401969672434, 0.8914440060978427, 0.8369204215214099, 0.7143336739134951, 0.8228387083529816, 0.9217818399840454, 0.8989571003525245, 0.5603366655665484, 0.8467033727085177, 0.7823233250037431, 0.9047396452452221, 0.7639108098310009, 0.8965602249479376, 0.7812983619545176, 0.8974680299438371, 0.331549869250688, 0.9149533373212466, 0.7353064749038508, 0.8738736590801535, 0.49384940122588883, 0.590616168805124, 0.8657403221335083, 0.8570997137492207, 0.8713431047584199, 0.8978022222293072, 0.8799624700373662, 0.8860877322535851, 0.8695157908399169, 0.5819925022759385, 0.7760288372958553, 0.8024854407715801, 0.48324056102663004, 0.873182540440088, 0.7519082294270742, 0.7843689083740496, 0.7615666865526336, 0.7713468287364216, 0.8194638682693811, 0.798962432387256, 0.740763676556344, 0.8943078521153041, 0.7125076794913178, 0.7575053251624293, 0.7994686247429763, 0.5790142214285698, 0.7370278512977513, 0.635298352913633, 0.8238061837348691, 0.7724485929515454, 0.6233315977583985, 0.71750336037889, 0.8355261167595109, 0.7988798526124636, 0.8223065375735391, 0.8063033812445803, 0.8415554728270862, 0.8693907555595162, 0.7858826939182024, 0.8404706916748098, 0.6677202394186884, 0.7185940001051647, 0.9090556781104191, 0.8829647100300089, 0.83098392479007, 0.5417822748015484, 0.5708118018459598, 0.8977928774729237, 0.7357333730635374, 0.8081232455187445, 0.9019384159824917, 0.9036492147679169, 0.8180073092220864, 0.7343008130879973, 0.7480002707798906, 0.8347691311831924, 0.6157044331673227, 0.7354158018011351, 0.4777400016173186, 0.587937159692434, 0.82843681522035, 0.6867769037161414, 0.88052157869665, 0.7642526888871092, 0.8169038927750575, 0.7193652069601885, 0.9007284778563129, 0.8759736798874969, 0.8075020013189993, 0.781356684992186, 0.7412766747687015, 0.7912594958482031, 0.7495586702782013, 0.8767964384136526, 0.7882741502268679, 0.553838900719679, 0.6116264652253652, 0.940902852026139, 0.8980578195921071, 0.7823070829559552, 0.883342300257783, 0.6576584669329519]

    plt.style.use('seaborn-deep')

    merged_ious = reference_model_216_mIoUs + prunned_with_distillation_retrain_without_distillation_216_mIoUs + prunned_without_distillation_216_mIoUs
    bins = np.linspace(min(merged_ious), 1, 100)

    #plt.hist([reference_model_216_mIoUs, prunned_with_distillation_retrain_without_distillation_216_mIoUs, prunned_without_distillation_216_mIoUs],
     #        bins, alpha=0.3, label=['reference_model_216_mIoUs', 'prunned_with_distillation_retrain_without_distillation_216_mIoUs', 'prunned_without_distillation_216_mIoUs'])

    plt.hist(reference_model_216_mIoUs, bins, alpha=0.5, label='reference_model_216_mIoUs')
    plt.hist(prunned_with_distillation_retrain_without_distillation_216_mIoUs, bins, alpha=0.5, label='prunned_with_distillation_retrain_without_distillation_216_mIoUs')
    plt.hist(prunned_without_distillation_216_mIoUs, bins, alpha=0.5, label='prunned_without_distillation_216_mIoUs')

    #plt.plot(reference_model_216_mIoUs, bins, 'bo')
    #plt.plot(prunned_with_distillation_retrain_without_distillation_216_mIoUs, bins, 'ro')
    #plt.plot(prunned_without_distillation_216_mIoUs, bins, 'go')
    plt.legend(loc='upper left')
    plt.title("model with 216k parameters")
    plt.show()
    """



    from visdom import Visdom
    import numpy as np

    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)
    win_loss = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        name='init',
        opts=dict(showlegend=True)
    )
    win_iou = viz.line(
        X=np.array([0]),
        Y=np.array([0]),
        opts=dict(showlegend=True)
    )
    opts = dict(showlegend=True)
    #viz.line(X=None, Y=None, win=win_loss, name='init', update='remove')


    def plot_with_visdom(viz, win_loss, epoch, value, description):
        try:
            viz.line(
                X=np.array([epoch]),
                Y=np.array([value]),
                win=win_loss,
                name=description,
                update='append'
            )
        except Exception as ex:
            print(ex)
            pass


    training_sum = 10
    validation_loss_sum = 8
    for epoch in [1,2,3,4,5]:
        training_sum = training_sum - 1
        validation_loss_sum = validation_loss_sum - 1
        plot_with_visdom(viz, win_loss, epoch, training_sum, 'training loss')
        plot_with_visdom(viz, win_loss, epoch, validation_loss_sum, 'validation loss')

    """
    try:
        win = viz.bar(
            X=[1, 2, 3, 6],
            opts=dict(
                rownames=['down_block1.conv1', 'down_block1.conv22', 'down_block1.conv31', 'down_block1.conv32'],
                title='Number of removed parameters for each layer',
                marginbottom=100,
                marginright=80
            )
        )


    except Exception as ex:
        print(ex)
        raise ex
    """



