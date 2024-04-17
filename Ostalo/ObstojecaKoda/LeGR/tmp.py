import torch
import os
from utils.drivers import get_dataloader, evaluate_on_test_set, ResourceManager
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib


def get_conf_matrix(args, predictions, targets):
    predictions_np = predictions.data.cpu().long().numpy()
    targets_np = targets.cpu().long().numpy()
    # for batch of predictions
    # if len(np.unique(targets)) != 2:
    #    print(len(np.unique(targets)))
    assert (predictions.shape == targets.shape)
    num_classes = 4

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






def conf_matrix_to_mIoU(args, confusion_matrix, log_per_class_miou=True):
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

    #print(confusion_matrix)
    if confusion_matrix.shape != (4,4):
        print(confusion_matrix.shape)
        raise NotImplementedError()

    MIoU = np.diag(confusion_matrix) / (
            np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
            np.diag(confusion_matrix))


    return np.mean(MIoU).item(1) # only IoU for sclera (not background)



def get_predictions(output):
    bs,c,h,w = output.size()
    values, indices = output.cpu().max(1)
    indices = indices.view(bs,h,w)
    #indices[indices > 1] = 0 # only 0 and 1 TODO
    #indices = indices.float()
    #print(torch.unique(indices))
    return indices


def visualize_on_test_images(expname, filename, student_predictions):
    os.makedirs('../IPAD/eyes_visualize/visualized_masks/', exist_ok=True)
    os.makedirs('../IPAD/eyes_visualize/visualized_masks/{}'.format(expname), exist_ok=True)
    for j in range(len(filename)):
        pred_img = student_predictions[j].cpu().numpy() / 3.0
        # inp = img[j].squeeze() * 0.5 + 0.5
        # img_orig = np.clip(inp, 0, 1)
        # img_orig = np.array(img_orig)
        # label = label_tensor[j].view(args.height, args.width)
        # label = np.array(label)

        # combine = np.hstack([img_orig, pred_img, label])
        clist = [(0., [0, 0, 0]), (1. / 3., [0, 1, 0]), (2. / 3., [1, 0, 0]), (3. / 3., [0, 0, 1])]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("name", clist)

        #combine = np.hstack([img_orig, pred_img, label])
        plt.imsave('../IPAD/eyes_visualize/visualized_masks/{}/{}.jpg'.format(expname, filename[j]), pred_img, cmap=cmap)





def main():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in [1])

    #model = torch.load("ckpt/matic_test_real_eyes2_gradIsNone_retrain2epochs_pruneAway60_bestarch_init.pt")
    #model = torch.load("ckpt/matic_test3_small_prune_away_70_bestarch_init.pt")
    #model = torch.load("ckpt/matic_test3_small_prune_away_65_bestarch_init.pt")
    #model = torch.load("ckpt/matic_test3_small_prune_away_63_bestarch_init.pt")
    #model = torch.load("ckpt/matic_test3_small_prune_away_61_bestarch_init.pt")
    #model = torch.load("ckpt/matic_test3_small_prune_away_60_bestarch_init.pt")
    #model = torch.load("ckpt/matic_pruneAway50_generations1_retrain50epochs_lr0_001_rank_l2_weight_bestarch_init.pt")


    #model = torch.load("ckpt/matic_pruneAway52_generations10_retrain50epochs_lr0_001_rank_l2_weight_best_model.pt")
    model = torch.load("ckpt/matic_pruneAway74_generations10_retrain50epochs_lr0_001_rank_l2_weight.pt")
    #if isinstance(model, nn.DataParallel):
    #    model = model.module
    model = model.to(device)

    # print('model summary')
    #print(model)
    #from torchsummary import summary
    #summary(model, input_size=(1, 640, 400))  # , batch_size=args.bs)  #  input_size=(channels, H, W)

    # TO EVALUATE ON TEST
    testloader = get_dataloader(None, 'eyes', None, 4, None, get_only_test=True)
    evaluate_on_test_set(device, model, testloader)


    # TO VISUALIZE
    #testloader = get_dataloader(None, 'eyes_visualize', None, 4, None, get_only_test=True)

    #with torch.no_grad():
    #    for i, batchdata in tqdm(enumerate(testloader), total=len(testloader)):
    #        img, label_tensor, filename, x, y = batchdata
    #        data = img.to(device)
    #        output = model(data)
            #target = label_tensor.to(device).long()
    #        predictions = get_predictions(output)
    #        visualize_on_test_images('LeGR', filename, predictions)




    """
    # print('list of test mious: ')
    # print(list_of_test_mIoUs)
    # plot histogram of test mIoUs
    axes = plt.gca()
    axes.set_ylim([0, 18])
    axes.set_xlim([0, 1])
    plt.hist(list_of_test_mIoUs, bins=100)
    plt.show()
    """

    # show resources
    rm = ResourceManager(model)
    rm.calculate_resources(torch.zeros((1, 1, 640, 400), device=device))
    print('resources: {0} flops, {1} params'
          .format(rm.cur_flops, rm.cur_n_params))



if __name__ == '__main__':
    main()
