import torch
import os
from utils.drivers import get_dataloader, evaluate_on_test_set, ResourceManager, retrain_model
import numpy as np
from tqdm import tqdm
from visdom import Visdom


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







def main():
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(str(x) for x in [0])

    model_name = 'matic_pruneAway74_generations10_retrain50epochs_lr0_001_rank_l2_weight'
    model = torch.load("ckpt/{0}_bestarch_init.pt".format(model_name))
    model = model.to(device)


    # visdom
    # RUN python -m visdom.server
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = "http://localhost"
    viz = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)
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

    # show resources
    rm = ResourceManager(model)
    rm.calculate_resources(torch.zeros((1, 1, 640, 400), device=device))
    print('resources: {0} flops, {1} params'
          .format(rm.cur_flops, rm.cur_n_params))

    train_loader, val_loader, test_loader = get_dataloader(None, 'eyes', None, 4, None)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    model = retrain_model(device, model, optimizer, scheduler, train_loader, val_loader, retrain_epochs=200, viz=viz, win_loss=win_loss, win_iou=win_iou)

    #if epoch % 5 == 0:
    torch.save(model.state_dict(), 'ckpt/{}.pkl'.format(model_name))
    torch.save(model, os.path.join('ckpt', '{}.pt'.format(model_name)))


if __name__ == '__main__':
    main()
