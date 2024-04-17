import os
import time
import torch
import queue
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from visdom import Visdom


from utils.drivers import train, test, get_dataloader, retrain_model, evaluate_on_test_set, ResourceManager
from model.MobileNetV2 import MobileNetV2, InvertedResidual
from pruner.fp_mbnetv2 import FilterPrunerMBNetV2
from pruner.fp_resnet import FilterPrunerResNet
from pruner.fp_ritnet import FilterPrunerRitnet
from pruner.fp_unet import FilterPrunerUNet
from model.ritnet import DenseNet2D_original
from model.unet import UNet

def print_gpu_memory_allocated(optional_prefix=''):
    print('==============={0}: memory allocated: {1:.3f}/{2:.3f} GB'.format(
        optional_prefix,
        torch.cuda.memory_allocated(device=torch.cuda.current_device()) / np.power(1024, 3),
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / np.power(1024, 3)))

class LeGR:
    def __init__(self, dataset, datapath, model, pruner, rank_type='l2_combined', batch_size=4, lr=1e-3, safeguard=0, global_random_rank=False, lub='', device='cuda', omega=None):
        self.device = device
        self.sample_for_ranking = 1 if rank_type in ['l1_weight', 'l2_weight', 'l2_bn', 'l1_bn', 'l2_bn_param', 'l1_combined', 'l2_combined'] else 5000
        self.safeguard = safeguard
        self.lub = lub
        self.lr = lr
        self.img_size = 32 if 'CIFAR' in args.dataset else 224
        self.batch_size = batch_size
        self.rank_type = rank_type

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.img_size, dataset, datapath, batch_size, args.no_val, args.workers)
        #_, _, self.test_loader = get_dataloader(self.img_size, dataset, datapath, batch_size, args.no_val)
        #print_gpu_memory_allocated('after getting data loaders')

        if 'CIFAR100' in dataset:
            num_classes = 100
        elif 'CIFAR10' in dataset:
            num_classes = 10
        elif 'ImageNet' in dataset:
            num_classes = 1000
        elif 'CUB200' in dataset:
            num_classes = 200
        elif 'sip' in dataset:
            num_classes = 4
        elif 'eyes' in dataset:
            num_classes = 2
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pruner = eval(pruner)(self.model, rank_type, num_classes, safeguard, random=global_random_rank, device=device, omega=omega)

        self.model.train()

    def learn_ranking_ea(self, name, model_desc, tau_hat, long_ft, target, my_model=None):
        # print(x.shape)
        #print('model summary')
        #print(self.model)
        #from torchsummary import summary
        #summary(self.model, input_size=(1,640, 400))  # , batch_size=args.bs)  #  input_size=(channels, H, W)



        #print('testing if retraning works on original model')
        #optimizer = torch.optim.Adam(self.pruner.model.parameters(), lr=self.lr)
        #retrain_model(self.device, self.model, optimizer, self.train_loader, retrain_epochs=tau_hat)

        # visdom
        # RUN python -m visdom.server
        DEFAULT_PORT = 8097
        DEFAULT_HOSTNAME = "http://localhost"
        viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

        name = name
        start_t = time.time()
        self.pruner.reset()
        self.pruner.model.eval()
        if my_model:
            self.pruner.forward(torch.zeros((1,1,640,400), device=self.device))
            #rm = ResourceManager(self.model)
            #rm.calculate_resources(torch.zeros((1, 1, 640,400), device=self.device))
            #print('ResourceManager calculated {0} flops'.format(rm.cur_flops))
            #print('ResourceManager calculated {0} parameters'.format(rm.cur_n_params))
        else:
            self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
        original_flops = self.pruner.cur_flops
        original_size = self.pruner.cur_size

        print('Before Pruning, FLOPs: {:.3f}, Size: {:.3f}M'.format(original_flops, original_size/1e6))

        mean_loss = []
        num_layers = len(self.pruner.filter_ranks)
        minimum_loss = 10
        best_perturbation = None
        POPULATIONS = 64
        SAMPLES = 16
        GENERATIONS = 10    #400
        SCALE_SIGMA = 1
        MUTATE_PERCENT = 0.1
        index_queue = queue.Queue(POPULATIONS)
        population_loss = np.zeros(0)
        population_data = []

        original_dist = self.pruner.filter_ranks.copy()
        original_dist_stat = {}
        for k in sorted(original_dist):
            a = original_dist[k].cpu().numpy()
            original_dist_stat[k] = {'mean': np.mean(a), 'std': np.std(a)}

        # Initialize Population
        for i in range(GENERATIONS):
            step_size = 1-(float(i)/(GENERATIONS*1.25))
            # Perturn distribution
            perturbation = []

            if i == POPULATIONS-1:
                for k in sorted(self.pruner.filter_ranks.keys()):
                    perturbation.append((1,0))
            elif i < POPULATIONS-1:
                for k in sorted(self.pruner.filter_ranks.keys()):
                    scale = np.exp(float(np.random.normal(0, SCALE_SIGMA)))
                    shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                    perturbation.append((scale, shift))
            else:
                mean_loss.append(np.mean(population_loss))
                sampled_idx = np.random.choice(POPULATIONS, SAMPLES)
                sampled_loss = population_loss[sampled_idx]
                winner_idx_ = np.argmin(sampled_loss)
                winner_idx = sampled_idx[winner_idx_]
                oldest_index = index_queue.get()

                # Mutate winner
                base = population_data[winner_idx]
                # Perturb distribution
                mnum = int(MUTATE_PERCENT * len(self.pruner.filter_ranks))
                mutate_candidate = np.random.choice(len(self.pruner.filter_ranks), mnum)
                for k in sorted(self.pruner.filter_ranks.keys()):
                    scale = 1
                    shift = 0
                    if k in mutate_candidate:
                        scale = np.exp(float(np.random.normal(0, SCALE_SIGMA*step_size)))
                        shift = float(np.random.normal(0, original_dist_stat[k]['std']))
                    perturbation.append((scale*base[k][0], shift+base[k][1]))

            # Given affine transformations, rank and prune
            self.pruner.pruning_with_transformations(original_dist, perturbation, target)

            # Re-measure the pruned model in terms of FLOPs and size
            self.pruner.reset()
            self.pruner.model.eval()

            if my_model:
                self.pruner.forward(torch.zeros((1, 1, 640,400), device=self.device))
            else:
                self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
            cur_flops = self.pruner.cur_flops
            cur_size = self.pruner.cur_size
            self.pruner.model = self.pruner.model.to(self.device)

            print('Density: {:.3f}% ({:.3f}M/{:.3f}M) | FLOPs: {:.3f}% ({:.3f}M/{:.3f}M)'.format(float(cur_size)/original_size*100, cur_size/1e6, original_size/1e6,
                                                                      float(cur_flops)/original_flops*100, cur_flops/1e6, original_flops/1e6))
            print('Fine tuning to recover from pruning iteration.')
            if my_model:
                optimizer = torch.optim.Adam(self.pruner.model.parameters(), lr=self.lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

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

                #print('model summary')
                #print(self.model)
                #from torchsummary import summary
                #summary(self.model, input_size=(1,640, 400))  # , batch_size=args.bs)  #  input_size=(channels, H, W)

                retrain_model(self.device, self.model, optimizer, scheduler, self.train_loader, self.val_loader, retrain_epochs=tau_hat,viz=viz, win_loss=win_loss, win_iou=win_iou)
            else:
                optimizer = optim.SGD(self.pruner.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

                if tau_hat > 0:
                    train(self.model, self.train_loader, self.val_loader, optimizer, epochs=1, steps=tau_hat, run_test=False, device=self.device)

            _, loss = test(self.model, self.val_loader, device=self.device, get_loss=True)

            if np.mean(loss) < minimum_loss:
                minimum_loss = np.mean(loss)
                best_perturbation = perturbation
                #print('saving curr best model')
                #torch.save(self.pruner.model, os.path.join('ckpt', '{}_best_model.pt'.format(name)))

            if i < POPULATIONS:
                index_queue.put(i)
                population_data.append(perturbation)
                population_loss = np.append(population_loss, [np.mean(loss)])
            else:
                index_queue.put(oldest_index)
                population_data[oldest_index] = perturbation
                population_loss[oldest_index] = np.mean(loss)

            # Restore the model back to origin
            if my_model:
                model_state_dict = torch.load(args.model)
                if my_model == 'ritnet':
                    model = DenseNet2D_original(dropout=True, prob=0.2, out_channels=4 if 'sip' in args.dataset else 2)
                else:
                    model = UNet(n_classes=4 if 'sip' in args.dataset else 2, pretrained=False)
                model = model.to(self.device)
                model.load_state_dict(model_state_dict, strict=False)  # We have attention map sums saved, but they're not in this model architecture, so we use strict=False
            else:
                model = torch.load(model_desc)
            if isinstance(model, nn.DataParallel):
                model = model.module
            model.eval()
            model = model.to(self.device)
            self.pruner.model = model
            self.model = model
            self.pruner.reset()
            self.pruner.model.eval()

            if my_model:
                self.pruner.forward(torch.zeros((1, 1, 640,400), device=self.device))
            else:
                self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
            print('Generation {}, Step: {:.2f}, Min Loss: {:.3f}'.format(i, step_size, np.min(population_loss)))

        total_t = time.time() - start_t
        print('Finished. Use {:.2f} hours. Minimum Loss: {:.3f}'.format(float(total_t) / 3600, minimum_loss))
        os.makedirs(os.path.dirname(os.path.join('log', name)), exist_ok=True)
        np.savetxt(os.path.join('./log', '{}_ea_loss.txt'.format(name)), np.array(mean_loss))
        np.savetxt(os.path.join('./log', '{}_ea_min.data'.format(name)), best_perturbation)

        # Use the best affine transformation to obtain the resulting model
        self.pruner.pruning_with_transformations(original_dist, best_perturbation, target)
        os.makedirs(os.path.dirname(os.path.join('ckpt', name)), exist_ok=True)
        torch.save(self.pruner.model, os.path.join('ckpt', '{}_bestarch_init.pt'.format(name)))

    def prune(self, name, model_name, long_ft, target=-1):
        test_acc = []
        b4ft_test_acc = []
        density = []
        flops = []

        # Get the accuracy before pruning
        acc = test(self.model, self.test_loader, device=self.device)
        test_acc.append(acc)
        b4ft_test_acc.append(acc)

        self.pruner.reset()
        self.model.eval()
        self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
        b4prune_size = self.pruner.cur_size
        b4prune_flops = self.pruner.cur_flops
        density.append(self.pruner.cur_size)
        flops.append(self.pruner.cur_flops)

        print('Before Pruning, Acc: {:.2f}%, FLOPs: {:.3f}M, Size: {:.3f}M'.format(acc, b4prune_flops/1e6, b4prune_size/1e6))

        # If there is learned affine transformation, load it.
        if self.lub != '':
            perturbation = np.loadtxt(self.lub)
        else:
            perturbation = np.array([[1., 0.] for _ in range(len(self.pruner.filter_ranks))])

        self.pruner.pruning_with_transformations(self.pruner.filter_ranks, perturbation, target)

        self.pruner.reset()
        self.model.eval()
        self.pruner.forward(torch.zeros((1,3,self.img_size,self.img_size), device=self.device))
        cur_flops = self.pruner.cur_flops
        cur_size = self.pruner.cur_size

        density.append(cur_size)
        flops.append(cur_flops)

        print('Experiment: {}'.format(name))
        print('Density: {:.3f}% ({:.3f}M/{:.3f}M) | FLOPs: {:.3f}% ({:.3f}M/{:.3f}M)'.format(cur_size/b4prune_size*100, cur_size/1e6, b4prune_size/1e6,
                                                                  cur_flops/b4prune_flops*100, cur_flops/1e6, b4prune_flops/1e6))
        print('Fine tuning to recover from pruning iteration.')
        if not os.path.exists('./ckpt'):
            os.makedirs('./ckpt')
        print('Saving untrained pruned model...')
        torch.save(self.pruner.model, os.path.join('ckpt', '{}_init.t7'.format(name)))
        acc = test(self.model, self.test_loader, device=self.device)
        b4ft_test_acc.append(acc)

        if not os.path.exists('./log'):
            os.makedirs('./log')

        print('Finished. Going to fine tune the model a bit more')
        if long_ft > 0:
                optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
                #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, long_ft)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [int(long_ft*0.3), int(long_ft*0.6), int(long_ft*0.8)], gamma=0.2)
                if args.no_val:
                    train(self.model, self.train_loader, self.test_loader, optimizer, epochs=long_ft, scheduler=scheduler, device=self.device, name=name)
                else:
                    train(self.model, self.train_loader, self.val_loader, optimizer, epochs=long_ft, scheduler=scheduler, device=self.device, name=name)
                acc = test(self.model, self.test_loader, device=self.device)
                test_acc.append(acc)
        else:
            acc = test(self.model, self.test_loader, device=self.device)
            test_acc.append(acc)

        log = np.stack([np.array(b4ft_test_acc), np.array(test_acc), np.array(density), np.array(flops)], axis=1)
        np.savetxt(os.path.join('./log', '{}_test_acc.txt'.format(name)), log)
        print('Summary')
        print('Before Pruning- Accuracy: {:.3f}, Cost: {:.3f}M'.format(test_acc[0], b4prune_flops/1e6))
        print('After Pruning- Accuracy: {:.3f}, Cost: {:.3f}M'.format(test_acc[-1], cur_flops/1e6))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='delete_me3', help='Name for the experiments, the resulting model and logs will use this')
    parser.add_argument("--datapath", type=str, default='./data', help='Path toward the dataset that is used for this experiment')
    parser.add_argument("--dataset", type=str, default='eyes', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    #parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10', help='The class name of the dataset that is used, please find available classes under the dataset folder')
    #parser.add_argument("--model", type=str, default='./ckpt/resnet56_cifar10.t7', help='The pre-trained model that pruning starts from')
    #parser.add_argument("--pruner", type=str, default='FilterPrunerResNet', help='Different network require differnt pruner implementation')

    parser.add_argument("--model", type=str, default='./ckpt/ritnet_original.pkl', help='The pre-trained model that pruning starts from')
    parser.add_argument("--pruner", type=str, default='FilterPrunerRitnet', help='Different network require differnt pruner implementation')
    parser.add_argument("--rank_type", type=str, default='l2_combined', help='The ranking criteria for filter pruning')
    parser.add_argument("--lub", type=str, default='', help='The affine transformations')
    parser.add_argument("--global_random_rank", action='store_true', default=False, help='When this is specified, none of the rank_type matters, it will randomly prune the filters')
    parser.add_argument("--tau_hat", type=int, default=25, help='The number of updates before evaluating for fitness (used in EA).')
    parser.add_argument("--long_ft", type=int, default=0, help='It specifies how many epochs to fine-tune the network once the pruning is done')
    parser.add_argument("--prune_away", type=float, default=75, help='How many percentage of constraints should be pruned away. E.g., 50 means 50% of FLOPs will be pruned away')
    parser.add_argument("--safeguard", type=float, default=0, help='A floating point number that represent at least how many percentage of the original number of channel should be preserved. E.g., 0.10 means no matter what ranking, each layer should have at least 10% of the number of original channels.')
    parser.add_argument("--batch_size", type=int, default=4, help='Batch size for training.')
    parser.add_argument("--workers", type=int, default=4, help='Number of workers')
    parser.add_argument("--min_lub", action='store_true', default=True, help='Use Evolutionary Algorithm to solve latent variable for minimizing Lipschitz upper bound')
    parser.add_argument("--uniform_pruning", action='store_true', default=False, help='Use Evolutionary Algorithm to solve latent variable for minimizing Lipschitz upper bound')
    parser.add_argument("--no_val", action='store_true', default=False, help='Use full dataset to train (use to compare with prior art in CIFAR-10)')
    parser.add_argument("--cpu", action='store_true', default=False, help='Use CPU')
    parser.add_argument("--lr", type=float, default=0.001, help='The learning rate for fine-tuning')
    parser.add_argument('--gpu', type=int, default=[1], nargs='+', help='used gpu')
    parser.add_argument('--omega', type=float, default=.5, help="omega parameter for custom ranking criterion")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    print('Pruning {}'.format(args.name))

    img_size = 32

    device = 'cpu' if args.cpu else 'cuda'
    prune_till = -1
    prune_away = args.prune_away

    my_model = None
    if 'ritnet' in args.model or 'densenet' in args.model:
        # os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        device = torch.device("cuda")
        my_model = 'ritnet'
        model_state_dict = torch.load(args.model)
        model = DenseNet2D_original(dropout=True, prob=0.2, out_channels=4 if 'sip' in args.dataset else 2)
        print('using model: ' + str(type(model)))
        model = model.to(device)
        model.load_state_dict(model_state_dict, strict=False)  # We have attention and activation map sums saved, but they're not in this model architecture, so we use strict=False
    elif 'unet' in args.model:
        # os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in args.gpu)
        device = torch.device("cuda")
        my_model = 'unet'
        model_state_dict = torch.load(args.model)
        model = UNet(n_classes=4 if 'sip' in args.dataset else 2, pretrained=False)
        print('using model: ' + str(type(model)))
        model = model.to(device)
        model.load_state_dict(model_state_dict, strict=False)  # We have activation map sums saved, but they're not in this model architecture, so we use strict=False
    else:
        model = torch.load(args.model)
        if isinstance(model, nn.DataParallel):
            model = model.module
        model = model.to(device)

    legr = LeGR(args.dataset, args.datapath, model, args.pruner, args.rank_type, args.batch_size, args.lr, safeguard=args.safeguard, global_random_rank=args.global_random_rank, lub=args.lub, device=device, omega=args.omega)

    if prune_away > 0:
        dummy_size = 32 if 'CIFAR' in args.dataset else 224
        legr.pruner.reset()
        legr.model.eval()
        if my_model:
            legr.pruner.forward(torch.zeros((1,1,640, 400), device=device))
        else:
            legr.pruner.forward(torch.zeros((1,3,dummy_size, dummy_size), device=device))
        b4prune_flops = legr.pruner.cur_flops
        prune_till = b4prune_flops * (1-(prune_away)/100.)
        #print('Pruned untill {:.3f}M'.format(prune_till/1000000.))
        if args.uniform_pruning:
            ratio = legr.pruner.get_uniform_ratio(prune_till)
            legr.pruner.safeguard = ratio
            prune_away = 99

    if args.min_lub:
        legr.learn_ranking_ea(args.name, args.model, args.tau_hat, args.long_ft, (1-(prune_away)/100.), my_model)
    else:
        legr.prune(args.name, args.model, args.long_ft, (1-(prune_away)/100.))


# python legr.py --name matic_pruneAway52_generations1_retrain50epochs_lr0_001_rank_l2_weight --dataset eyes