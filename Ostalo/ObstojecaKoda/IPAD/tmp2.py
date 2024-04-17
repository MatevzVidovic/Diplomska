from tmp import count_zeroed_filters_for_model, disable_filter, get_parameter_name_and_index_from_activations_dict_key, outer_hook
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

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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


#def initialize_globals(args):
#    global LOGDIR
#    LOGDIR = 'logs/{}'.format(args.expname)
#    global logger
#    logger = Logger(os.path.join(LOGDIR, 'logs.log'))

def load_student(args, device):
    student_model = Net(input_channels=1, output_channels=4)
#    logger.write('using student model: ' + str(type(student_model)))
    student_model = student_model.to(device)

    if args.resume != '':
        print("EXISTING STUDENT DICT from: {}".format(args.resume))
        student_state_dict = torch.load(args.resume)
        student_model.load_state_dict(student_state_dict)
        # student_model.eval() # not needed if training continues

    return student_model

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


    args.resume = 'logs/TMP/models/model_without_2_filters.pkl'
    model = load_student(args, device) # ignore optimizer and scheduler
    all_zeroed_filters, all_used_filters = count_zeroed_filters_for_model(model, device)
    print('all zeroed filters: {0}'.format(len(all_zeroed_filters)))
    print('all used filters: {0}'.format(len(all_used_filters)))

    print('models loaded correctly...')
    args.bs = 1 # todo
    Path2file = "eyes_tmp"
    print('path to file: ' + str(Path2file))
    train_dataset = IrisDataset(filepath=Path2file, split='train',
                        transform=transform, **kwargs)
    print('len: ' + str(train_dataset.__len__()))

    trainloader = DataLoader(train_dataset, batch_size=args.bs,
                             shuffle=False, num_workers=args.workers, drop_last=False)

    print('datasets made... ' + str(len(trainloader)))

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    criterion = CrossEntropyLoss2d()

    print('init weights..')
    model._initialize_weights()

    for filter in all_zeroed_filters:
        print(filter)
        disable_filter(device, model, filter)  # disable this filter

        name, index = get_parameter_name_and_index_from_activations_dict_key(filter)
        model_layer = getattr(model, name)
        model_layer_weight = getattr(model_layer, 'weight')
        model_layer_weight.register_hook(outer_hook(device, index))

    train(device, model, trainloader, optimizer, criterion)


def train(device, model, trainloader, optimizer, criterion):
    for epoch in range(2):
        model.train()
        # check how many zeroed filters there are
        all_zeroed_filters, all_used_filters = count_zeroed_filters_for_model(model, device)
        print(model.conv1.weight)
        print(model.conv1.bias)
        print(model.conv1.weight.grad)
        for i, batchdata in enumerate(trainloader):
            img, labels, index, spatialWeights, maxDist = batchdata
            data = img.to(device)
            target = labels.to(device).long()
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()



if __name__ == '__main__':
    main()