"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from train_config import get_config
import torch
from train_utils import get_device
from datasets import get_voc_instance_dataloader, get_sbd_instance_dataloader
from models import get_model, get_weakly_model
from tools.utils import Cluster, Visualizer
from torchvision import transforms
import os
import sys
import os.path as osp
from tools.utils_for_density import VisualizeToolForDensity
from voc_eval_tool.eval_voc_inst_seg import eval_instance_segmentation_voc
import numpy as np

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
sys.path.append(osp.join(sys.path[0], '../../../'))

from tools.utils import VisualizeTool, get_cluster, get_cluster_result
from models import get_model
from train_utils import load, model_accelerate
import pickle
from torch import nn
from weakly_smask_bbox.utils import get_bbox_info_from_mask

# torch.backends.cudnn.benchmark = True

gt_cache = './voc_gt_cache.pkl'


def get_and_cache_gt(args):
    try:
        with open(gt_cache, 'rb') as f:
            gt_masks_all, gt_labels_all = pickle.load(f)
    except Exception:
        # batch size set to 1
        assert args.test_batch_size == 1
        device = get_device(args)
        data_loader = get_voc_instance_dataloader(args, train=False)

        gt_masks_all, gt_labels_all = [], []
        data_loader = tqdm(enumerate(data_loader))
        for i, sample in data_loader:
            im = sample['image'].to(device)
            instance = sample['instance'].to(device).squeeze()  # h x w
            class_label = sample['label'].to(device).squeeze()  # h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]
            instance_to_cls = {int(i): int(class_label[instance == i].unique()) for i in instance_ids}

            gt_masks, gt_labels = [], []
            for inst in instance_ids:
                inst = int(inst)
                gt_masks.append((instance == inst).cpu().numpy())

                # start with zero, do not contarin bkgd
                gt_labels.append(instance_to_cls[inst] - 1)

            gt_masks_all.append(np.array(gt_masks))
            gt_labels_all.append(np.array(gt_labels))
        with open(gt_cache, 'wb') as f:
            pickle.dump((gt_masks_all, gt_labels_all), f)

    return gt_masks_all, gt_labels_all


def get_gt_masks_and_labels(sample):
    im = sample['image']
    instance = sample['instance'].squeeze()  # h x w
    class_label = sample['label'].squeeze()  # h x w

    instance_ids = instance.unique()
    instance_ids = instance_ids[instance_ids != 0]
    instance_to_cls = {int(i): int(class_label[instance == i].unique()) for i in instance_ids}

    gt_masks, gt_labels = [], []
    for inst in instance_ids:
        inst = int(inst)
        gt_masks.append((instance == inst).cpu().numpy())

        # start with zero, do not contarin bkgd
        gt_labels.append(instance_to_cls[inst] - 1)
    return np.array(gt_masks), np.array(gt_labels)


def from_seg_pred_to_one_hot(seg_pred):
    score, seg_max = seg_pred.max(dim=1, keepdim=True)
    return label_to_one_hot(seg_max, n_class=seg_pred.size(1))


def label_to_one_hot(targets: torch.Tensor, n_class: int):
    """
    get one-hot tensor from targets, ignore the 255 label
    :param targets: long tensor[bs, 1, h, w]
    :param nlabels: int
    :return: float tensor [bs, nlabel, h, w]
    """
    # batch_size, _, h, w = targets.size()
    # res = torch.zeros([batch_size, nlabels, h, w])
    targets = targets.squeeze(dim=1)
    # print(targets.shape)
    zeros = torch.zeros(targets.shape).long().to(targets.device)

    # del 255.
    targets_ignore = targets >= n_class
    # print(targets_ignore)
    targets = torch.where(targets < n_class, targets, zeros)

    one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=n_class)
    one_hot[targets_ignore] = 0
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


def eval_voc(args):
    from tools.multi_scale_eval import multi_scale_eval
    device = get_device(args)
    cluster_func = get_cluster(args)
    visualizetool = VisualizeTool(args)
    densityvisualizetool = VisualizeToolForDensity(args)

    print(f'*** test mode : {args.test_mode}')
    data_loader = get_voc_instance_dataloader(args, train=False)

    if args.test_mode in ('anc', 'weakly'):
        model = get_weakly_model(args)
    elif args.test_mode in ('normal', 'test', 'multi_scale', 'flip', 'multi_scale_flip'):
        model = get_model(args)
    elif args.test_mode in ('seg', 'anc_seg'):
        model = get_weakly_model(args)
        model.seg = True
    else:
        raise NotImplementedError

    model = model_accelerate(args, model)
    _, _, model, _, _ = load(args, model)

    model.eval()
    data_loader = tqdm(enumerate(data_loader))
    # gt_masks_all, gt_labels_all = get_and_cache_gt(args)
    gt_masks_all, gt_labels_all = [], []
    pred_masks_all, pred_labels_all, pred_scores_all = [], [], []
    counter = None
    with torch.no_grad():
        for i, sample in data_loader:
            if counter is not None and i >= counter:
                break
            im = sample['image'].to(device)
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)
            gt_masks, gt_labels = get_gt_masks_and_labels(sample)
            gt_masks_all.append(gt_masks)
            gt_labels_all.append(gt_labels)

            # check
            batch_size = im.size(0)
            assert batch_size == 1

            if args.test_mode == 'anc':
                bbox = sample['bbox'].to(device)
                output = model(im, bbox)
            elif args.test_mode in ('normal', 'test', 'weakly'):
                output = model(im)
            elif 'seg' in args.test_mode:
                if 'anc' in args.test_mode:
                    bbox = sample['bbox'].to(device)
                    output, seg = model(im, bbox)
                else:
                    output, seg = model(im)
                score, seg_max = seg.max(dim=1, keepdim=True)
                seg_one_hot = from_seg_pred_to_one_hot(seg)
                # seg_one_hot = seg > 0.4
                # output[:, 4:] = output[:, 4:] * (seg[:, 0:1] - 1).abs()
                output[:, 4:] = output[:, 4:] * seg_one_hot[:, 1:].float()
                pred_label = seg_max.squeeze(dim=0)
            elif args.test_mode in ('multi_scale', 'flip', 'multi_scale_flip'):
                output = multi_scale_eval(im, model, flip=True)
                pass
            else:
                raise NotImplementedError

            assert args.cluster != 'gt'

            instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func, output[0], instance[0])
            if args.display > 0:
                if 'density' in args.cluster:
                    densityvisualizetool.visualize(image=im[0], label=class_label[0],
                                                   instance=instance[0],
                                                   prediction=output[0].detach(), instance_pred=instance_map)
                else:
                    visualizetool.visualize(image=im[0], label=class_label[0],
                                            instance=instance[0],
                                            prediction=output[0].detach(), instance_pred=instance_map)
            pred_masks, pred_labels, pred_scores = [], [], []
            for p in predictions:
                # {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': start_with_0})
                pred_masks.append(p['mask'].cpu().numpy())
                pred_labels.append(p['cls'])
                pred_scores.append(p['score'])

            pred_masks_all.append(np.array(pred_masks))  # 0, 255
            pred_labels_all.append(np.array(pred_labels))
            pred_scores_all.append(np.array(pred_scores))

            pass
            # break
        if counter is not None:
            gt_masks_all, gt_labels_all = gt_masks_all[:counter], gt_labels_all[:counter]
        res_5 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.5, use_07_metric=False)
        for i, ap in enumerate(res_5['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.5', res_5['map'])

        res_75 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.75, use_07_metric=False)
        for i, ap in enumerate(res_75['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.75', res_75['map'])
    del visualizetool, data_loader
    torch.cuda.empty_cache()
    return res_5['map'], res_75['map']


def eval_during_traing(args, model, criterion, data_loader, visualizetool, counter=None):
    # define meters
    cluster_func = get_cluster(args)
    device = get_device(args)

    # put model into eval mode
    model.eval()
    data_loader = tqdm(enumerate(data_loader))
    # gt_masks_all, gt_labels_all = get_and_cache_gt(args)
    gt_masks_all, gt_labels_all = [], []
    pred_masks_all, pred_labels_all, pred_scores_all = [], [], []
    with torch.no_grad():
        for i, sample in data_loader:
            if counter is not None and i >= counter:
                break
            im = sample['image'].to(device)
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)
            gt_masks, gt_labels = get_gt_masks_and_labels(sample)
            gt_masks_all.append(gt_masks)
            gt_labels_all.append(gt_labels)

            output = model(im)
            assert args.cluster != 'gt'
            instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func, output[0], instance[0])
            if args.display > 0:
                visualizetool.visualize(image=im[0], label=class_label[0],
                                        instance=instance[0],
                                        prediction=output[0].detach(), instance_pred=instance_map)
            pred_masks, pred_labels, pred_scores = [], [], []
            for p in predictions:
                # {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': start_with_0})
                pred_masks.append(p['mask'].cpu().numpy())
                pred_labels.append(p['cls'])
                pred_scores.append(p['score'])

            pred_masks_all.append(np.array(pred_masks))  # 0, 255
            pred_labels_all.append(np.array(pred_labels))
            pred_scores_all.append(np.array(pred_scores))

            pass
            # break
        if counter is not None:
            gt_masks_all, gt_labels_all = gt_masks_all[:counter], gt_labels_all[:counter]
        res_5 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.5, use_07_metric=False)
        for i, ap in enumerate(res_5['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.5', res_5['map'])

        res_75 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.75, use_07_metric=False)
        for i, ap in enumerate(res_75['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.75', res_75['map'])
    del visualizetool, data_loader
    torch.cuda.empty_cache()
    return res_5['map'], res_75['map']


def eval_with_bbox_and_seg(args, model, seg_model, data_loader, visualizetool,
                           counter=None, infer_with_GTBBox=False):
    # define meters
    cluster_func = get_cluster(args)
    device = get_device(args)

    # put model into eval mode
    model.eval()
    seg_model.eval()
    if not isinstance(seg_model, nn.DataParallel):
        seg_model.refine_learning()
    else:
        seg_model.module.refine_learning()
    data_loader = tqdm(enumerate(data_loader))
    # gt_masks_all, gt_labels_all = get_and_cache_gt(args)
    gt_masks_all, gt_labels_all = [], []
    pred_masks_all, pred_labels_all, pred_scores_all = [], [], []
    with torch.no_grad():
        for i, sample in data_loader:
            if counter is not None and i >= counter:
                break
            im = sample['image'].to(device)
            B, _, H, W = im.shape
            assert B == 1  # batch size should be one
            print(f'imput size:({H}, {W})')
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)
            bbox = sample['bbox'].to(device)

            gt_masks, gt_labels = get_gt_masks_and_labels(sample)
            gt_masks_all.append(gt_masks)
            gt_labels_all.append(gt_labels)

            seg_with_bbox = seg_model(im, bbox)
            probs, mask = seg_with_bbox.max(dim=1, keepdim=True)
            if args.seg_preprocessing == 'ignore':
                mask[probs < 0.5] = args.n_class + 1  # ignore
            # one-hot -> 0 ~ args.n_class+1
            mask_one_hot = label_to_one_hot(targets=mask, n_class=args.n_class + 2)
            if args.seg_preprocessing in ('ignore', 'onehot'):
                output = model(im, bbox, mask_one_hot)
            else:
                output = model(im, bbox, seg_with_bbox)
            assert args.cluster != 'gt'
            if not infer_with_GTBBox:
                instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func,
                                                                              output[0], instance[0])
            else:
                instance_map = torch.zeros(H, W).long()
                bboxes = get_bbox_info_from_mask(instance[0], class_label[0])
                mask_count = 1
                predictions = []
                for bbox in bboxes:
                    """
                    {
                        'cls': instance_to_cls[inst_id],
                        'area': area,
                        'bbox': bbox,
                        'instance_id': inst_id
                    }
                    """
                    cls = bbox['cls']
                    bbox_mask = torch.zeros(H, W).to(output.device)
                    u, l, b, r = bbox['bbox']
                    u, l, b, r = int(u), int(l), int(b), int(r)
                    bbox_mask[u:b + 1, l:r + 1] = 1
                    cls_bbox_mask = torch.zeros(args.n_class, H, W).to(output.device)
                    cls_bbox_mask[cls - 1, u:b + 1, l:r + 1] = 1
                    output_squeeze = output.squeeze()
                    temp_prediction = torch.cat([output_squeeze[:2] * bbox_mask, output_squeeze[2:4],
                                                 output_squeeze[4:] * cls_bbox_mask], dim=0)
                    instance_map_one, result, instances_cls = get_cluster_result(args, cluster_func,
                                                                                 temp_prediction, instance[0])
                    # combine all instance map
                    # choose biggest mask
                    max_pixel_count = -1
                    biggest_mask = None
                    for i in result:
                        # {'mask': mask_0_255, 'score': seed_score, 'cls': cls }
                        # i['cls'] start with 0, cls start with 1 in VOC, for Cityscapes, need to be converted
                        if (args.n_class == 20 and i['cls'] == cls - 1) \
                                or (args.n_class == 8 and class2index[i['cls']] == cls - 1):
                            temp_mask = i['mask'] > 0
                            pixel_count = temp_mask.sum()
                            # print(f'pixel_count:{pixel_count}')
                            if pixel_count > max_pixel_count:
                                max_pixel_count = pixel_count
                                biggest_mask = i
                            pass
                    if biggest_mask is not None:
                        instance_map[biggest_mask['mask'] > 0] = mask_count
                        mask_count += 1
                        predictions.append(biggest_mask)
                    pass
                pass
            if args.display > 0:
                visualizetool.visualize(image=im[0], label=class_label[0],
                                        instance=instance[0],
                                        prediction=output[0].detach(), instance_pred=instance_map)
            pred_masks, pred_labels, pred_scores = [], [], []
            for p in predictions:
                # {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': start_with_0})
                pred_masks.append(p['mask'].cpu().numpy())
                if args.n_class == 8:
                    # convert class_id to index
                    pred_labels.append(class2index[p['cls']])
                else:
                    pred_labels.append(p['cls'])
                pred_scores.append(p['score'])

            pred_masks_all.append(np.array(pred_masks))  # 0, 255
            pred_labels_all.append(np.array(pred_labels))
            pred_scores_all.append(np.array(pred_scores))

            pass
            # break
        if counter is not None:
            gt_masks_all, gt_labels_all = gt_masks_all[:counter], gt_labels_all[:counter]
        res_5 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.5, use_07_metric=False)
        for i, ap in enumerate(res_5['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.5', res_5['map'])

        res_75 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.75, use_07_metric=False)
        for i, ap in enumerate(res_75['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.75', res_75['map'])
    data_loader.close()
    del visualizetool, data_loader
    torch.cuda.empty_cache()
    return res_5['map'], res_75['map']


def eval_with_bbox(args, model, data_loader, visualizetool, counter=None, infer_with_GTBBox=False):
    # define meters
    cluster_func = get_cluster(args)
    device = get_device(args)

    # put model into eval mode
    model.eval()
    data_loader = tqdm(enumerate(data_loader))
    # gt_masks_all, gt_labels_all = get_and_cache_gt(args)
    gt_masks_all, gt_labels_all = [], []
    pred_masks_all, pred_labels_all, pred_scores_all = [], [], []
    with torch.no_grad():
        for i, sample in data_loader:
            if counter is not None and i >= counter:
                break
            im = sample['image'].to(device)
            B, _, H, W = im.shape
            assert B == 1  # batch size should be one
            print(f'imput size:({H}, {W})')
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)
            bbox = sample['bbox'].to(device)

            gt_masks, gt_labels = get_gt_masks_and_labels(sample)
            gt_masks_all.append(gt_masks)
            gt_labels_all.append(gt_labels)

            output = model(im, bbox)
            assert args.cluster != 'gt'
            if not infer_with_GTBBox:
                instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func,
                                                                              output[0], instance[0])
            else:
                instance_map = torch.zeros(H, W).long()
                bboxes = get_bbox_info_from_mask(instance[0], class_label[0])
                mask_count = 1
                predictions = []
                for bbox in bboxes:
                    """
                    {
                        'cls': instance_to_cls[inst_id],
                        'area': area,
                        'bbox': bbox,
                        'instance_id': inst_id
                    }
                    """
                    cls = bbox['cls']
                    bbox_mask = torch.zeros(H, W).to(output.device)
                    u, l, b, r = bbox['bbox']
                    u, l, b, r = int(u), int(l), int(b), int(r)
                    bbox_mask[u:b + 1, l:r + 1] = 1
                    cls_bbox_mask = torch.zeros(args.n_class, H, W).to(output.device)
                    cls_bbox_mask[cls - 1, u:b + 1, l:r + 1] = 1
                    output_squeeze = output.squeeze()
                    temp_prediction = torch.cat([output_squeeze[:2] * bbox_mask, output_squeeze[2:4],
                                                 output_squeeze[4:] * cls_bbox_mask], dim=0)
                    instance_map_one, result, instances_cls = get_cluster_result(args, cluster_func,
                                                                                 temp_prediction, instance[0])
                    # combine all instance map
                    # choose biggest mask
                    max_pixel_count = -1
                    biggest_mask = None
                    for i in result:
                        # {'mask': mask_0_255, 'score': seed_score, 'cls': cls }
                        # i['cls'] start with 0, cls start with 1 in VOC, for Cityscapes, need to be converted
                        if (args.n_class == 20 and i['cls'] == cls - 1) \
                                or (args.n_class == 8 and class2index[i['cls']] == cls - 1):
                            temp_mask = i['mask'] > 0
                            pixel_count = temp_mask.sum()
                            # print(f'pixel_count:{pixel_count}')
                            if pixel_count > max_pixel_count:
                                max_pixel_count = pixel_count
                                biggest_mask = i
                            pass
                    if biggest_mask is not None:
                        instance_map[biggest_mask['mask'] > 0] = mask_count
                        mask_count += 1
                        predictions.append(biggest_mask)
                    pass
                pass
            if args.display > 0:
                visualizetool.visualize(image=im[0], label=class_label[0],
                                        instance=instance[0],
                                        prediction=output[0].detach(), instance_pred=instance_map)
            pred_masks, pred_labels, pred_scores = [], [], []
            for p in predictions:
                # {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': start_with_0})
                pred_masks.append(p['mask'].cpu().numpy())
                if args.n_class == 8:
                    # convert class_id to index
                    pred_labels.append(class2index[p['cls']])
                else:
                    pred_labels.append(p['cls'])
                pred_scores.append(p['score'])

            pred_masks_all.append(np.array(pred_masks))  # 0, 255
            pred_labels_all.append(np.array(pred_labels))
            pred_scores_all.append(np.array(pred_scores))

            pass
            # break
        if counter is not None:
            gt_masks_all, gt_labels_all = gt_masks_all[:counter], gt_labels_all[:counter]
        res_5 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.5, use_07_metric=False)
        for i, ap in enumerate(res_5['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.5', res_5['map'])

        res_75 = eval_instance_segmentation_voc(
            pred_masks_all, pred_labels_all, pred_scores_all,
            gt_masks_all, gt_labels_all,
            iou_thresh=0.75, use_07_metric=False)
        for i, ap in enumerate(res_75['ap']):
            print(f'{i}:{ap}')
        print(f'mAP@0.75', res_75['map'])
    data_loader.close()
    del visualizetool, data_loader
    torch.cuda.empty_cache()
    return res_5['map'], res_75['map']


class2index = {24: 0, 25: 1, 26: 2, 27: 3, 28: 4, 31: 5, 32: 6, 33: 7}


def _test():
    args = get_config()
    gt_masks_all, gt_labels_all = get_and_cache_gt(args)
    pred_scores_all = [np.array([1.] * len(i)) for i in gt_labels_all]
    res = eval_instance_segmentation_voc(
        gt_masks_all, gt_labels_all, pred_scores_all,
        gt_masks_all, gt_labels_all,
        iou_thresh=0.5, use_07_metric=False)
    print(res)
    pass


if __name__ == '__main__':
    args = get_config()
    _test()
    # get_and_cache_gt(args)
