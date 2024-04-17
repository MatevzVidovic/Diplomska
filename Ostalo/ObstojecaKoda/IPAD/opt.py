from pprint import pprint
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--dataset', type=str, default='eyes/', help='name of dataset')
    # Used for test and sclera_images script...
    parser.add_argument('--model', help='model name',default='densenet')
    # Optimization: General
    parser.add_argument('--teacher', type=str, help='Path to trained teacher model', default='')
                        #default='logs/teacher_original_ritnet_lr0_001/models/dense_net_180.pkl')
    parser.add_argument('--bs', type=int, default=4)  # 8,12   4 is used for train - to make results comparable
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=201)
    parser.add_argument('--startEpoch', type=int, help='', default=0)
    parser.add_argument('--workers', type=int, help='Number of workers', default=8)
    parser.add_argument('--load', type=str, default=None, help='load checkpoint file name')  # logs/teacher_original_ritnet_lr0_001/models/dense_net_200.pkl
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--expname', type=str, default='distillation', help='extra explanation of the method')
    parser.add_argument('--useGPU', type=str, default='True', help='Set it as False if GPU is unavailable')
    parser.add_argument('--alwaysPenalize', type=str, default='true',
                        help='If true always penalize, otherwise compare student and teacher IoU')
    parser.add_argument('--resume', type=str, default='', help='Path to model (name must be the same for optimizer in folder optimizers) to resume training')
                        #default='logs/ritnet_for_pruning_lr0_001/models/densenet_180.pkl')
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help='used gpu')
    parser.add_argument('--pruningGpus', type=str, default='0', nargs='+', help='used gpus')
    parser.add_argument('--width', type=int, default=400, help='resize dataset images to this width') #400
    parser.add_argument('--height', type=int, default=640, help='resize dataset images to this height') #640
    parser.add_argument('--prune', choices=('all', 'conv', 'channels'), default='all', help="'channels' will prune only 1x1 convolutions, 'conv' will prune all others, 'all' will prune both")
    parser.add_argument('--norm', default=2, type=int, help="what norm to use for the criterion")
    parser.add_argument('--channelsUseWeightsOnly', action='store_true', help="don't use activation criterion for 1x1 convolutions")
    parser.add_argument('--no-interpolate', action='store_false', dest='interpolate', help="don't interpolate convolution weights")
    parser.add_argument('--visualize', '--visualise', action='store_true', help="visualise predictions, filters, and activations")
    parser.add_argument('--random', action='store_true', help="ignore the criterion and prune layers randomly")
    parser.add_argument('--uniform', action='store_true', help="ignore the criterion and prune layers uniformly")

    # parse
    args = parser.parse_args()
    opt = vars(args)
    pprint('parsed input parameters:')
    pprint(opt)
    return args


if __name__ == '__main__':
    opt = parse_args()
    print('opt[\'dataset\'] is ', opt.dataset)
