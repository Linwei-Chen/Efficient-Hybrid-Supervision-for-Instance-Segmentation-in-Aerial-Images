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
from models import get_model
from tools.utils import AverageMeter, Cluster, Logger, Visualizer
from train_utils import get_device
from train_utils import get_device, get_optimizer, print_lr, get_sheduler, load, save_checkpoint
from tools.utils import VisualizeTool, get_cluster_result, get_cluster, display
from tools.utils_for_density import VisualizeToolForDensity
from os import path as osp
from train_utils import model_accelerate, update_lr, poly_adjust_lr, get_lr
from criterions import get_criterion
from datasets import get_sbd_instance_dataloader, get_voc_instance_dataloader
from torch import nn
from train_utils import freeze_batch_norm, check_batch_norm_freeze

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train_epoch(args, model, optimizer, criterion, data_loader, visualizetool):
    # define meters
    loss_meter = AverageMeter()
    cluster_func = get_cluster(args)
    device = get_device(args)

    model.train()

    # put model into training mode
    if args.freeze > 0:
        # print(f'===> Model freeze.')
        # model.eval()
        freeze_batch_norm(model)
        check_batch_norm_freeze(model)
    else:
        model.train()
    # model.eval()

    data_loader = tqdm(data_loader)
    for i, sample in enumerate(data_loader):
        info = poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                              itr=args.iter + i, max_itr=args.max_iter, power=0.9, mute=True)
        data_loader.write(info)

        im = sample['image'].to(device)
        instances = sample['instance'].to(device)
        class_labels = sample['label'].to(device)

        output = model(im)

        loss = criterion(output, instances, class_labels)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.detach().item()

        # print(f'loss:{loss}')
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


from eval_voc import eval_during_traing


def val(args, model, criterion, data_loader, visualizetool):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()
    cluster_func = get_cluster(args)
    device = get_device(args)

    # put model into eval mode
    model.eval()

    with torch.no_grad():
        data_loader = tqdm(data_loader)
        for i, sample in enumerate(data_loader):
            im = sample['image'].to(device)
            instances = sample['instance'].to(device)
            class_labels = sample['label'].to(device)

            output = model(im)
            loss = criterion(prediction=output, instances=instances, labels=class_labels, iou_meter=iou_meter)
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

    return loss_meter.avg, iou_meter.avg


def train():
    args = get_config()
    logger = Logger(('train', 'mAP50', 'mAP75'), 'loss')
    # use density check
    if 'density' in args.model_name:
        assert 'density' in args.cluster
        assert args.criterion == 'density'
        visualizetool = VisualizeToolForDensity(args)
    elif 'multioffset' in args.model_name:
        assert 'multioffset' in args.cluster
        assert 'multioffset' in args.criterion
        visualizetool = VisualizeTool(args)
    else:
        visualizetool = VisualizeTool(args)

    assert args.test_batch_size == 1
    model = get_model(args)
    # init or not
    model.init_output(args.n_sigma)

    # model = model_accelerate(args, model)

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model).to(get_device(args))

    # model = nn.DataParallel(model)
    # model = model.to(get_device(args))
    optimizer = get_optimizer(args, model)

    start_epoch, best_iou, model, optimizer, logger = load(args, model, optimizer=optimizer, logger=logger)

    criterion = get_criterion(args)

    print_lr(optimizer)
    if args.lr_update is not None:
        update_lr(optimizer, args.lr_update)
    print_lr(optimizer)

    # if args.epoch_update is not None:
    #     start_epoch = args.epoch_update

    # scheduler = get_sheduler(args, optimizer)
    for epoch in range(start_epoch, args.epochs):
        voc_train_data_loader = get_voc_instance_dataloader(args, train=True)
        sbd_train_data_loader = get_sbd_instance_dataloader(args)
        val_data_loader = get_voc_instance_dataloader(args, train=False)
        args.max_iter = (len(sbd_train_data_loader) + len(voc_train_data_loader)) * args.epochs
        args.iter = (len(sbd_train_data_loader) + len(voc_train_data_loader)) * epoch

        print('Starting epoch {}'.format(epoch))
        # scheduler.step(epoch)
        print_lr(optimizer)

        train_loss = 0
        train_loss += train_epoch(args, model, optimizer, criterion, sbd_train_data_loader, visualizetool)
        args.iter += (len(sbd_train_data_loader))
        train_loss += train_epoch(args, model, optimizer, criterion, voc_train_data_loader, visualizetool)
        map_50, map_75 = eval_during_traing(args, model, criterion, val_data_loader, visualizetool, counter=None)

        print('===> train loss: {:.2f}'.format(train_loss))
        print('===> mAP50: {:.4f}, mAP75: {:.4f}'.format(map_50, map_75))

        logger.add('train', train_loss)
        logger.add('mAP50', map_50)
        logger.add('mAP75', map_75)

        is_best = map_50 > best_iou
        best_iou = max(map_50, best_iou)

        state = {
            'epoch': epoch,
            'best_iou': best_iou,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'logger_data': logger.data
        }
        save_checkpoint(args, state, is_best, map_50)

        try:
            logger.plot(save=True, save_dir=args.save)
        except Exception:
            print('*** ERROR: save plot fail.')
        torch.cuda.empty_cache()
        del voc_train_data_loader, sbd_train_data_loader
        del val_data_loader


if __name__ == '__main__':
    train()
