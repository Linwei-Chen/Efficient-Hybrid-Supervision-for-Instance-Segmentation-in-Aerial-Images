"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time
from os import path as osp
import sys

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
sys.path.append('../')

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
from os import path as osp
from train_utils import model_accelerate, update_lr, poly_adjust_lr, get_lr
from criterions import get_criterion
from datasets import get_cityscapes_dataloader
from torch import nn
from train_utils import freeze_batch_norm, check_batch_norm_freeze
from criterions import get_segloss
import numpy as np
from models import get_weakly_model
from train_utils import copy_weakly_model
from weakly_smask_bbox.utils import get_bbox_instance_label_mask_refine_batch, get_bbox_instance_label_mask_prefine_batch, get_instance_label_mask_overlap255_batch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def set_seg_true(model):
    if isinstance(model, nn.DataParallel):
        model.module.seg = True
    else:
        model.seg = True


def train_epoch(args, model, optimizer, criterion, data_loader, visualizetool, is_weakly_dataset=False):
    # define meters
    loss_meter = AverageMeter()
    cluster_func = get_cluster(args)
    device = get_device(args)

    # get Lovasz + CE
    seg_criterion = get_segloss(args)

    # put model into training mode
    if args.freeze > 0:
        model.train()
        freeze_batch_norm(model)
        check_batch_norm_freeze(model)
    else:
        model.train()

    # anc model
    if is_weakly_dataset:
        anc_model = copy_weakly_model(args, model)
        set_seg_true(anc_model)
    else:
        anc_model = None

    set_seg_true(model)

    # make model output seg:

    data_loader = tqdm(data_loader)
    for i, sample in enumerate(data_loader):
        im = sample['image'].to(device)
        instances = sample['instance'].to(device)
        class_labels = sample['label'].to(device)
        bbox = sample['bbox'].to(device)

        if is_weakly_dataset:
            anc_model.eval()
            optimizer.zero_grad()
            with torch.no_grad():
                output_with_bbox, seg_with_bbox = anc_model(im, bbox=bbox)
                bbox[:, 0] = 1
                seg_with_bbox = seg_with_bbox * bbox
                probs, mask = seg_with_bbox.softmax(dim=1).max(dim=1, keepdim=True)
                # instances, mask = get_bbox_instance_label_mask_prefine_batch(instances, class_labels, mask)
                ##########
                instances, inst_mask = get_instance_label_mask_overlap255_batch(instances, class_labels, mask)
                mask[probs < 0.5] = 255
                ##########

            output, seg = model(im)
            iloss = criterion(output, instances, inst_mask)
            sloss = seg_criterion(input=seg, target=mask.long())
            loss = iloss + 2 * sloss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            iloss = iloss.mean().detach().item()
            sloss = sloss.mean().detach().item()
            loss = loss.detach().item()
            loss_meter.update(loss)

            # show the first result of batch
            if args.display and i % args.display_iter == 0:
                instance_map, predictions, instances_cls = \
                    get_cluster_result(args, cluster_func, output[-1].detach(), instances[-1])
                cls = visualizetool.visualize(image=im[-1], label=class_labels[-1],
                                              instance=instances[-1],
                                              prediction=output[-1].detach(), instance_pred=instance_map)

                data_loader.write(
                    f'{args.save} | iloss:{iloss:.8f} | sloss:{sloss:.8f} | lr:{get_lr(optimizer):.8f} | cls:{cls}')
            else:
                data_loader.write(f'{args.save} | iloss:{iloss:.8f} | sloss:{sloss:.8f} | lr:{get_lr(optimizer):.8f}')

        else:
            #######
            instances, inst_mask = get_instance_label_mask_overlap255_batch(instances, class_labels)
            #######
            if 'finetune' not in args.weakly_train:
                optimizer.zero_grad()
                output, seg = model(im, bbox=bbox)
                loss = criterion(output, instances, inst_mask) + 2 * seg_criterion(seg, class_labels.long())
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                loss = loss.detach().item()
                loss_meter.update(loss)

            optimizer.zero_grad()
            output, seg = model(im, bbox=None)
            iloss = criterion(output, instances, inst_mask)
            sloss = seg_criterion(input=seg, target=class_labels.long())
            loss = iloss + 2 * sloss
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            iloss = iloss.mean().detach().item()
            sloss = sloss.mean().detach().item()
            loss = loss.detach().item()
            loss_meter.update(loss)

            # show the first result of batch
            if args.display and i % args.display_iter == 0:
                instance_map, predictions, instances_cls = \
                    get_cluster_result(args, cluster_func, output[0].detach(), instances[0])

                cls = visualizetool.visualize(image=im[0], label=class_labels[0], instance=instances[0],
                                              prediction=output[0].detach(), instance_pred=instance_map)

                data_loader.write(
                    f'{args.save} | iloss:{iloss:.8f} | sloss:{sloss:.8f} | lr:{get_lr(optimizer):.8f} | cls:{cls}')
            else:
                data_loader.write(f'{args.save} | iloss:{iloss:.8f} | sloss:{sloss:.8f} | lr:{get_lr(optimizer):.8f}')

    return loss_meter.avg


from train_jointly import val_with_seg


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

    # model = model_accelerate(args, model)
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model).to(get_device(args))

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
        poly_adjust_lr(optimizer, lr=args.lr if args.lr_update is None else args.lr_update,
                       itr=epoch, max_itr=args.epochs, power=0.9)

        args.data_choose_size = abs(args.data_choose_size)
        args.dataset_repeat = 3
        train_data_loader = get_cityscapes_dataloader(args, train=True)

        args.data_choose_size = -args.data_choose_size
        args.dataset_repeat = 1
        weakly_train_data_loader = get_cityscapes_dataloader(args, train=True)
        val_data_loader = get_cityscapes_dataloader(args, train=False)

        print('Starting epoch {}'.format(epoch))
        # scheduler.step(epoch)
        print_lr(optimizer)

        if 'strong' in args.weakly_train:
            print('===> training on strong dataset...')
            train_loss = train_epoch(args, model, optimizer, criterion, train_data_loader, visualizetool)
            val_loss, val_iou, miou = val_with_seg(args, model, criterion, val_data_loader, visualizetool)

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
        if 'weakly' in args.weakly_train:
            print('===> training on weakly dataset...')
            # weakly training
            train_loss = train_epoch(args, model, optimizer, criterion,
                                     weakly_train_data_loader, visualizetool, is_weakly_dataset=True)

            val_loss, val_iou, miou = val_with_seg(args, model, criterion, val_data_loader, visualizetool)

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
        del train_data_loader, weakly_train_data_loader
        del val_data_loader


if __name__ == '__main__':
    train()
