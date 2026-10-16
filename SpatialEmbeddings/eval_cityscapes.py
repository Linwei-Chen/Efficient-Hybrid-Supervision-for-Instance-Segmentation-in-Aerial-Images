"""
For testing model on Cityscapes automatically using official metrics
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from train_config import get_config
import torch
from train_utils import get_device
from datasets import get_cityscapes_dataloader, get_test_dataloader
from models import get_model, get_weakly_model
from tools.utils import Cluster, Visualizer
from torchvision import transforms
import os
import sys
import os.path as osp
from tools.utils_for_density import VisualizeToolForDensity

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
sys.path.append(osp.join(sys.path[0], '../../../'))

from tools.utils import VisualizeTool, get_cluster, get_cluster_result
from models import get_model
from train_utils import load, model_accelerate
import numpy as np
from voc_eval_tool.eval_voc_inst_seg import eval_instance_segmentation_voc

torch.backends.cudnn.benchmark = True


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


def test(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = get_device(args)
    cluster_func = get_cluster(args)
    visualizetool = VisualizeTool(args)
    densityvisualizetool = VisualizeToolForDensity(args)

    print(f'*** test mode : {args.test_mode}')
    if args.test_mode == 'test':
        data_loader = get_test_dataloader(args)
    else:
        data_loader = get_cityscapes_dataloader(args, train=False)

    if args.test_mode in ('anc', 'weakly'):
        model = get_weakly_model(args)
    elif args.test_mode in ('normal', 'test'):
        model = get_model(args)
    elif args.test_mode in ('seg', 'anc_seg', 'recurrent'):
        model = get_weakly_model(args)
        model.seg = True
    else:
        raise NotImplementedError

    model = model_accelerate(args, model)
    _, _, model, _, _ = load(args, model)

    model.eval()
    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for sample in data_loader:

            im = sample['image'].to(device)
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)

            # check
            batch_size = im.size(0)
            if args.test_save > 0:
                assert batch_size == 1

            if args.test_mode == 'anc':
                bbox = sample['bbox'].to(device)
                output = model(im, bbox)
            elif args.test_mode == 'recurrent':
                output, seg = model(im)
                # seg_one_hot = from_seg_pred_to_one_hot(seg)
                # seg_one_hot[:, 1:,][seg[:, 1:] > 0.25] = 1.0
                # output, seg = model(im, seg_one_hot)
                output, seg = model(im, seg)
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
            else:
                raise NotImplementedError

            assert args.cluster != 'gt'

            instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func, output[0], instance[0])
            if 'density' in args.cluster:
                densityvisualizetool.visualize(image=im[0], label=class_label[0],
                                               instance=instance[0],
                                               prediction=output[0].detach(), instance_pred=instance_map)
            else:
                visualizetool.visualize(image=im[0], label=class_label[0],
                                        instance=instance[0],
                                        prediction=output[0].detach(), instance_pred=instance_map)

            data_loader.write(args.save)
            if args.test_save > 0:

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

                txt_file = os.path.join(args.eval_save, base + '.txt')
                with open(txt_file, 'w') as f:

                    # loop over instances
                    for id, pred in enumerate(predictions):
                        im_name = base + '_{:02d}.png'.format(id)
                        im = torchvision.transforms.ToPILImage()(
                            pred['mask'].unsqueeze(0))

                        # write image
                        im.save(os.path.join(args.eval_save, im_name))

                        # write to file
                        cl = pred['cls']
                        score = pred['score']
                        f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
    del visualizetool, data_loader
    torch.cuda.empty_cache()


class_ids_to_index = {24: 1, 25: 2, 26: 3, 27: 4, 28: 5, 31: 6, 32: 7, 33: 8}


def fast_test(args, counter=None):
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
            gt_labels.append(instance_to_cls[inst])
        return np.array(gt_masks), np.array(gt_labels)

    torch.multiprocessing.set_sharing_strategy('file_system')
    device = get_device(args)
    cluster_func = get_cluster(args)
    visualizetool = VisualizeTool(args)
    densityvisualizetool = VisualizeToolForDensity(args)

    print(f'*** test mode : {args.test_mode}')
    if args.test_mode == 'test':
        data_loader = get_test_dataloader(args)
    else:
        data_loader = get_cityscapes_dataloader(args, train=False)

    if args.test_mode in ('anc', 'weakly'):
        model = get_weakly_model(args)
    elif args.test_mode in ('normal', 'test'):
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
    pred_masks_all, pred_labels_all, pred_scores_all = [], [], []
    gt_masks_all, gt_labels_all = [], []
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
            if args.test_save > 0:
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
            else:
                raise NotImplementedError

            assert args.cluster != 'gt'

            instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func, output[0],
                                                                          instance[0])
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
                # pred_labels.append(p['cls'])
                pred_labels.append(class_ids_to_index[p['cls']])
                pred_scores.append(p['score'])

            pred_masks_all.append(np.array(pred_masks))  # 0, 255
            pred_labels_all.append(np.array(pred_labels))
            pred_scores_all.append(np.array(pred_scores))

        if counter is not None:
            gt_masks_all, gt_labels_all = gt_masks_all[:counter], gt_labels_all[:counter]

        AP = 0.
        for iou in range(50, 100, 5):
            res = eval_instance_segmentation_voc(
                pred_masks_all, pred_labels_all, pred_scores_all,
                gt_masks_all, gt_labels_all,
                iou_thresh=float(iou) / float(100), use_07_metric=False)
            for i, ap in enumerate(res['ap']):
                print(f'{i}:{ap}')
            print(f'mAP@{float(iou) / float(100):.2f}', res['map'])
            AP += res['map']
        print(f'AP:{AP / float(11)}')


def upper_bound_with_pred_seg(args):
    def get_prediction_liks_dict(instance, label):
        instance = instance.squeeze()
        label = label.squeeze()
        h, w = instance.size(-2), instance.size(-1)
        instance_ids = instance.unique()
        instance_ids = instance_ids[instance_ids != 0]
        instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}
        for i in instance_ids:
            i = int(i)
            temp = torch.zeros(h, w).byte()
            temp
        pass

    device = get_device(args)
    cluster_func = get_cluster(args)
    visualizetool = VisualizeTool(args)
    densityvisualizetool = VisualizeToolForDensity(args)
    #########
    assert args.test_mode == 'seg'
    #########

    print(f'*** test mode : {args.test_mode}')
    if args.test_mode == 'test':
        data_loader = get_test_dataloader(args)
    else:
        data_loader = get_cityscapes_dataloader(args, train=False)

    if args.test_mode in ('anc', 'weakly'):
        model = get_weakly_model(args)
    elif args.test_mode in ('normal', 'test'):
        model = get_model(args)
    elif args.test_mode in ('seg', 'anc_seg'):
        model = get_weakly_model(args)
        model.seg = True
    else:
        raise NotImplementedError

    model = model_accelerate(args, model)
    _, _, model, _, _ = load(args, model)

    model.eval()
    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for sample in data_loader:

            im = sample['image'].to(device)
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)

            # check
            batch_size = im.size(0)
            if args.test_save > 0:
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
                ##########
                inter_seg = class_label[seg_max == class_label].squeeze()
                inter_inst = instance[seg_max == class_label].squeeze()
                #########

            else:
                raise NotImplementedError

            assert args.cluster != 'gt'

            instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func, output[0], instance[0])
            if 'density' in args.cluster:
                densityvisualizetool.visualize(image=im[0], label=class_label[0],
                                               instance=instance[0],
                                               prediction=output[0].detach(), instance_pred=instance_map)
            else:
                visualizetool.visualize(image=im[0], label=class_label[0],
                                        instance=instance[0],
                                        prediction=output[0].detach(), instance_pred=instance_map)

            data_loader.write(args.save)
            if args.test_save > 0:

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

                txt_file = os.path.join(args.eval_save, base + '.txt')
                with open(txt_file, 'w') as f:

                    # loop over instances
                    for id, pred in enumerate(predictions):
                        im_name = base + '_{:02d}.png'.format(id)
                        im = torchvision.transforms.ToPILImage()(
                            pred['mask'].unsqueeze(0))

                        # write image
                        im.save(os.path.join(args.eval_save, im_name))

                        # write to file
                        cl = pred['cls']
                        score = pred['score']
                        f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
    del visualizetool, data_loader
    torch.cuda.empty_cache()


def upper_bound_with_bbox_smask(args):
    from weakly_smask_bbox.utils import get_bbox_instance_label_mask_refine_batch
    def get_dict_like_prediction(instance, label):
        instance = instance.squeeze()
        label = label.squeeze()
        instance_ids = instance.unique()
        instance_ids = instance_ids[instance_ids != 0]
        instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}
        pass

    device = get_device(args)
    visualizetool = VisualizeTool(args)
    densityvisualizetool = VisualizeToolForDensity(args)

    print(f'*** test mode : {args.test_mode}')
    if args.test_mode == 'test':
        data_loader = get_test_dataloader(args)
    else:
        data_loader = get_cityscapes_dataloader(args, train=False)

    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for sample in data_loader:

            im = sample['image'].to(device)
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)
            instance, class_label = get_bbox_instance_label_mask_refine_batch(instance, class_label)

            assert args.cluster != 'gt'

            data_loader.write(args.save)
            if args.test_save > 0:

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

                txt_file = os.path.join(args.eval_save, base + '.txt')
                with open(txt_file, 'w') as f:

                    # loop over instances
                    for id, pred in enumerate(predictions):
                        im_name = base + '_{:02d}.png'.format(id)
                        im = torchvision.transforms.ToPILImage()(
                            pred['mask'].unsqueeze(0))

                        # write image
                        im.save(os.path.join(args.eval_save, im_name))

                        # write to file
                        cl = pred['cls']
                        score = pred['score']
                        f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
    del visualizetool, data_loader
    torch.cuda.empty_cache()


def auto_test():
    pass


def clear_txt():
    files = ['./temp.txt', './res.txt']
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
            res = '\n'
            for line in lines:
                for word in ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle',
                             'ap:', 'ap50:', 'AP', 'average', '####', '----'):
                    if word in line:
                        res += line
                        break
            with open(file, 'w') as f:
                f.write(res)


def parse_txt():
    with open('./res.txt', 'r') as f:
        f = f.readlines()
        ap, ap50 = 0, 0
        for i, line in enumerate(f[::-1]):

            ap_pos = line.find('ap:')
            if ap_pos >= 0:
                ap = float(line[ap_pos + 3:ap_pos + 9])
                print(f'ap:{ap}')

            ap50_pos = line.find('ap50:')
            if ap50_pos >= 0:
                ap50 = float(line[ap50_pos + 5:ap50_pos + 11])
                print(f'ap50:{ap50}')

            if ap > 0 and ap50 > 0:
                return ap, ap50


if __name__ == '__main__':
    from auto_test import config

    args = config()
    aug = args.aug
    print(args.list)

    for std, save in zip(args.std, args.list):
        # std or not
        if std > 0 and 'std' not in aug:
            args.aug = aug + '_std'
        else:
            args.aug = aug

        if '.pth' not in save:
            args.resume_path = osp.join(save, 'checkpoint.pth')
            args.save = save
        else:
            args.resume_path = save
            args.save = osp.split(save)[0]
        print(args)
        if args.test_data_set == 'cityscapes':
            assert args.n_class in (8, 9)
            if osp.exists(args.resume_path):
                fast_test(args, counter=None)
            else:
                print(f'*** {args.resume_path} not exist! continue to next one.')
                continue
