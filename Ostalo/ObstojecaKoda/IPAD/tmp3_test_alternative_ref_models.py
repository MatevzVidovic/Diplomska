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

from models import model_dict
from utils import ResourceManager
from train_with_pruning import count_number_of_learnable_parameters



def load_student(args, device):
    student_model = model_dict['tmp3']
    print('using student model: ' + str(type(student_model)))
    student_model = student_model.to(device)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=135, gamma=0.1)

    return student_model, optimizer, scheduler




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

    model, _, _ = load_student(args, device) # ignore optimizer and scheduler


    #from torchsummary import summary
    #summary(model, input_size=(1, args.height, args.width))  # , batch_size=args.bs)  #  input_size=(channels, H, W)
    #print(model)

    rm = ResourceManager(model)
    rm.calculate_resources(torch.zeros((1, 1, args.height, args.width), device=device))
    learnable_parameters, all_parameters = count_number_of_learnable_parameters(model, device)
    print('resources (removed_filters: {0}): {1}/{2} flops, {3}/{4} params'
          .format(rm.n_removed_filters, rm.cur_flops, rm.original_flops, learnable_parameters, all_parameters) )






if __name__ == '__main__':
    main()