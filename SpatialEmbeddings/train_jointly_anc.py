"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from train_config import get_config
from criterions.multi_cls_my_loss import SpatialEmbLoss
from models import get_weakly_model
from tools.utils import AverageMeter, Cluster, Logger, Visualizer
from train_utils import get_device
from train_utils import get_device, get_optimizer, print_lr, get_sheduler, load, save_checkpoint
from tools.utils import VisualizeTool, get_cluster_result, get_cluster, display
from os import path as osp
from train_utils import model_accelerate, update_lr, poly_adjust_lr, get_lr
from criterions import get_criterion
from datasets import get_cityscapes_dataloader
from torch import nn
from train_utils import freeze_batch_norm, check_batch_norm_freeze
from datasets.CityscapesDataset import class_ids_to_index
from criterions import get_segloss
import numpy as np

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def instances_refine(instances, bbox):
    """
    refine the instance mask using bbox mask

    instances.append({'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': class_ids[cls]})

    mode
    score of res will be > weakly threshold

    :param instances:
    :return:
    """
    assert bbox.dim() == 3
    instance_map = torch.zeros(bbox.shape).to(bbox.device)

    for i, instance in enumerate(instances):
        mask = instance['mask'].float().to(bbox.device)  # hxw
        score = instance['score']  # float
        cls = class_ids_to_index[instance['cls']]  # float
        mask = mask * bbox[cls]
        instances[i]['mask'] = mask
        instance_map[cls][mask > 0] = score

    return instances, instance_map.argmax(dim=0).float()


def train_epoch(args, model, optimizer, criterion, data_loader, visualizetool, is_weakly_dataset=False):
    # define meters
    loss_meter = AverageMeter()

    # args.cluster = 'for_each'
    cluster_func = get_cluster(args)
    device = get_device(args)

    # put model into training mode
    model.train()
    if args.freeze > 0:
        freeze_batch_norm(model)
        check_batch_norm_freeze(model)

    else:
        model.train()

    # model output seg:
    if isinstance(model, nn.DataParallel):
        model.module.seg = True
    else:
        model.seg = True

    # Lovasz + CE:
    seg_criterion = get_segloss(args)

    data_loader = tqdm(data_loader)
    for i, sample in enumerate(data_loader):

        im = sample['image'].to(device)
        instances = sample['instance'].to(device)
        class_labels = sample['label'].to(device)
        bbox = sample['bbox'].to(device)

        if is_weakly_dataset:
            raise NotImplementedError

        else:
            model.train()
            optimizer.zero_grad()
            output, seg = model(im, bbox=bbox)
            loss = criterion(output, instances, class_labels) + seg_criterion(seg, class_labels.long())
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            loss_meter.update(loss)

            optimizer.zero_grad()
            output, seg = model(im, bbox=None)
            loss = criterion(output, instances, class_labels) + seg_criterion(seg, class_labels.long())
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            loss_meter.update(loss)

            # show the first result of batch
            if args.display and i % args.display_iter == 0:
                instance_map, predictions, instances_cls = \
                    get_cluster_result(args, cluster_func, output[0].detach(), instances[0])

                cls = visualizetool.visualize(image=im[0], label=class_labels[0], instance=instances[0],
                                              prediction=output[0].detach(), instance_pred=instance_map)

                data_loader.write(f'{args.save} | loss:{loss:.8f} | lr:{get_lr(optimizer):.8f} | cls:{cls}')
            else:
                data_loader.write(f'{args.save} | loss:{loss:.8f} | lr:{get_lr(optimizer):.8f}')

    return loss_meter.avg


from tools.iou_eval import _fast_hist, label_accuracy_score, label_accuracy_score_use_hist

# using bbox as input
def val_with_bbox(args, model, criterion, data_loader, visualizetool):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    cluster_func = get_cluster(args)
    device = get_device(args)

    # put model into eval mode
    model.eval()
    hist = np.zeros((args.n_class + 1, args.n_class + 1))

    # Lovasz + CE:
    seg_criterion = get_segloss(args)

    with torch.no_grad():
        data_loader = tqdm(data_loader)
        for i, sample in enumerate(data_loader):
            im = sample['image'].to(device)
            instances = sample['instance'].to(device)
            class_labels = sample['label'].to(device)
            bbox = sample['bbox'].to(device)

            output, seg = model(im, bbox)
            N, _, h, w = seg.shape
            pred = seg.cpu().numpy().transpose(0, 2, 3, 1).reshape(-1, args.n_class + 1).argmax(axis=1).reshape(N, h, w)
            hist += _fast_hist(class_labels.cpu().numpy().flatten(), pred.flatten(), args.n_class + 1)
            loss = criterion(prediction=output, instances=instances,
                             labels=class_labels, iou_meter=iou_meter) + seg_criterion(seg, class_labels.long())
            loss = loss.mean()

            if args.display and i % args.display_iter == 0:
                instance_map, predictions, instances_cls = \
                    get_cluster_result(args, cluster_func, output[0].detach(), instances[0])

                cls = visualizetool.visualize(image=im[0], label=class_labels[0], instance=instances[0],
                                              prediction=output[0].detach(), instance_pred=instance_map)
                data_loader.write(f'loss:{loss:.8f} | cls:{cls}')
            else:
                data_loader.write(f'loss:{loss:.8f}')
            loss = loss.detach().item()
            loss_meter.update(loss)
    acc, acc_cls, mean_iu, fwavacc, iu = label_accuracy_score_use_hist(hist)
    print('acc:{}\nacc_cls:{}\nmean_iu:{}\nfwavacc:{}\niu:{}'.format(acc, acc_cls, mean_iu, fwavacc, iu))
    return loss_meter.avg, iou_meter.avg, mean_iu


def train():
    args = get_config()
    if 'bbox' not in args.aug:
        args.aug += '_bbox'
    assert args.data_choose_size is not None
    assert args.data_choose_size > 0

    logger = Logger(('train', 'val', 'iou', 'miou'), 'loss')
    visualizetool = VisualizeTool(args)

    model = get_weakly_model(args)
    # init or not
    model.init_output(args.n_sigma)

    model = model_accelerate(args, model)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
        # model = model.to(get_device(args))

    optimizer = get_optimizer(args, model)

    start_epoch, best_iou, model, optimizer, logger = load(args, model, optimizer=optimizer, logger=logger)

    args.criterion = 'weakly'
    weakly_criterion = get_criterion(args)

    args.criterion = 'bce'
    criterion = get_criterion(args)

    # update lr
    print_lr(optimizer)
    if args.lr_update is not None:
        update_lr(optimizer, args.lr_update)
    print_lr(optimizer)

    for epoch in range(start_epoch, args.epochs):
        poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                       itr=epoch, max_itr=args.epochs, power=0.9)
        args.dataset_repeat = 3
        train_data_loader = get_cityscapes_dataloader(args, train=True)

        val_data_loader = get_cityscapes_dataloader(args, train=False)

        print('Starting epoch {}'.format(epoch))
        # scheduler.step(epoch)
        print_lr(optimizer)

        # strong training
        train_loss = train_epoch(args, model, optimizer, criterion,
                                 train_data_loader, visualizetool, is_weakly_dataset=False)

        val_loss, val_iou, miou = val_with_bbox(args, model, criterion, val_data_loader, visualizetool)

        print('===> train loss: {:.2f}'.format(train_loss))
        print('===> val loss: {:.4f}, val iou: {:.4f}, miou: {:.4f}'.format(val_loss, val_iou, miou))

        logger.add('train', train_loss)
        logger.add('val', val_loss)
        logger.add('iou', val_iou)
        logger.add('miou', miou)

        is_best = miou > best_iou
        best_iou = max(miou, best_iou)

        state = {
            'epoch': epoch,
            'best_iou': best_iou,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(args, state, is_best, miou)

        try:
            logger.plot(save=True, save_dir=args.save)
        except Exception:
            print('*** ERROR: save plot fail.')

        torch.cuda.empty_cache()
        del train_data_loader
        del val_data_loader


if __name__ == '__main__':
    train()
