"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tools.utils_for_density import DensityCluster
from tools.utils_for_multi_offset import ClusterForMultiOffset
from pycocotools import mask as coco_mask
# from multiprocessing import Process, Pool
import torch
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


# clustering
def get_cluster(args, name=None):
    if name is None:
        name = args.cluster

    if name == 'for_each':
        cluster = Cluster(args)
        return cluster.cluster_for_each
    elif name == 'for_each_iter':
        cluster = Cluster(args)
        return cluster.cluster_for_each_iter
    elif name == 'max':
        cluster = Cluster(args)
        return cluster.cluster
    elif name == 'gt':
        cluster = Cluster(args)
        return cluster.cluster_with_gt

    elif name == 'density_for_each':
        densitycluster = DensityCluster(args)
        return densitycluster.cluster_for_each
    elif name == 'density_max':
        densitycluster = DensityCluster(args)
        return densitycluster.cluster
    elif name == 'density_gt':
        densitycluster = DensityCluster(args)
        return densitycluster.cluster_with_gt

    elif name == 'multioffset_for_each':
        multioffsetcluster = ClusterForMultiOffset(args)
        return multioffsetcluster.cluster_for_each
    elif name == 'multioffset_for_each_iter':
        multioffsetcluster = ClusterForMultiOffset(args)
        return multioffsetcluster.cluster_for_each_iter
    else:
        raise NotImplementedError


def get_cluster_result(args, cluster_func, prediction, instance=None):
    with torch.no_grad():
        if args.cluster == 'gt' or args.cluster == 'density_gt':
            instance_map = cluster_func(prediction, instance)
            return instance_map, None, None
        else:
            instance_map, predictions, instances_cls = cluster_func(prediction)
            return instance_map, predictions, instances_cls


class AverageMeter(object):

    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.reset()
        self.lock = threading.Lock()

    def reset(self):
        self.sum = [0] * self.num_classes
        self.count = [0] * self.num_classes
        self.avg_per_class = [0] * self.num_classes
        self.avg = 0

    def update(self, val, cl=0):
        with self.lock:
            self.sum[cl] += val
            self.count[cl] += 1
            self.avg_per_class = [
                x / y if x > 0 else 0 for x, y in zip(self.sum, self.count)]
            self.avg = sum(self.avg_per_class) / len(self.avg_per_class)


def display(args):
    if args.display:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")


# Visualizer
# visualizer = Visualizer()

class VisualizeTool:
    def __init__(self, args):
        self.args = args
        display(args)
        self.visualizer = Visualizer(keys=('image', 'xy', 'sigma', 'seed', 'label', 'instance'))

    def visualize(self,
                  image: torch.Tensor,
                  label: torch.Tensor,
                  instance: torch.Tensor,
                  prediction: torch.Tensor,
                  instance_pred: torch.Tensor):
        """
        :param args:
        :param image: 3, h, w
        :param prediction:
        :return:
        """
        assert image.dim() == 3
        assert label.dim() == 3
        assert instance.dim() == 3
        assert prediction.dim() == 3
        assert instance_pred.dim() == 2

        with torch.no_grad():
            xy = prediction[:2]
            xy[0] = (xy[0] - xy[0].min()) / (xy[0].max() - xy[0].min())
            xy[1] = (xy[1] - xy[1].min()) / (xy[1].max() - xy[1].min())
            xy = xy.mean(dim=0, keepdim=True)

            sigma = prediction[2:2 + self.args.n_sigma]
            sigma[0] = (sigma[0] - sigma[0].min()) / (sigma[0].max() - sigma[0].min())
            sigma[1] = (sigma[1] - sigma[1].min()) / (sigma[1].max() - sigma[1].min())
            sigma = sigma.mean(dim=0, keepdim=True)

            seed_map = prediction[2 + self.args.n_sigma:]
            n_class = seed_map.size(0)
            prob, cls = seed_map.max(dim=0)
            if n_class in (8, 15, 20):
                cls = cls + 1
                cls = cls.float()
                cls[prob < 0.5] = 0.
                prob[prob < 0.5] = 0.

            elif n_class == 9:
                prob[cls == 0] = 0
                cls = cls.float()
            else:
                raise NotImplementedError

            # print('===> cls:', cls.unique().data)

        image = image.cpu()
        label = label.cpu()
        instance = instance.cpu()
        prob = prob.cpu()
        cls = cls.cpu()
        instance_pred = instance_pred.cpu()

        # show img
        self.visualizer.display(image.cpu() * 0.25 + 0.5, 'image')

        # show xy
        self.visualizer.display(xy.cpu(), 'xy')

        # show sigma
        self.visualizer.display(sigma.cpu(), 'sigma')

        # show label
        self.visualizer.display([cls.cpu(), prob.cpu(), label.cpu()], 'label')

        # show instance
        self.visualizer.display([instance_pred, instance], 'instance')

        return cls.unique().data


class Visualizer:

    def __init__(self, keys):
        plt.ion()
        self.wins = {k: None for k in keys}

    def display(self, image, key):

        n_images = len(image) if isinstance(image, (list, tuple)) else 1

        if self.wins[key] is None:
            self.wins[key] = plt.subplots(ncols=n_images)

        fig, ax = self.wins[key]
        n_axes = len(ax) if isinstance(ax, collections.Iterable) else 1

        assert n_images == n_axes

        if n_images == 1:
            ax.cla()
            ax.set_axis_off()
            ax.imshow(self.prepare_img(image))
        else:
            for i in range(n_images):
                ax[i].cla()
                ax[i].set_axis_off()
                ax[i].imshow(self.prepare_img(image[i]))

        plt.draw()
        self.mypause(0.001)

    @staticmethod
    def prepare_img(image):
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            image.squeeze_()
            image = image.numpy()

        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[0] in {1, 3}:
                image = image.transpose(1, 2, 0)
            return image

    @staticmethod
    def mypause(interval):
        backend = plt.rcParams['backend']
        if backend in matplotlib.rcsetup.interactive_bk:
            figManager = matplotlib._pylab_helpers.Gcf.get_active()
            if figManager is not None:
                canvas = figManager.canvas
                if canvas.figure.stale:
                    canvas.draw()
                canvas.start_event_loop(interval)
                return


class ClusterMP:
    """
    cluster for each
    """

    def __init__(self, spatial_emb, sigma, seed_map_all, threshold, proposal_thrs, max_dets):
        self.spatial_emb = spatial_emb
        self.sigma = sigma
        self.seed_map_all = seed_map_all
        self.n_class = seed_map_all.size(0)
        self.n_sigma = sigma.size(0)
        self.threshold = threshold
        self.proposal_thrs = proposal_thrs
        self.max_dets = 128
        # self.max_dets = max_dets

    def __call__(self, cls):
        spatial_emb, sigma, seed_map_all = self.spatial_emb, self.sigma, self.seed_map_all
        height, width = spatial_emb.size(1), spatial_emb.size(2)
        instance_map = torch.zeros(height, width).long()
        instances = []
        instance_cls = {}

        '''
        import multiprocessing
        cores = len(worklist)
        pool = multiprocessing.Pool(processes=cores)
        pool.map(os.system, worklist)
        '''

        # instance counter
        count = 1
        seed_map = seed_map_all[cls]
        mask = seed_map > 0.5
        if mask.sum() > self.proposal_thrs:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)

            # seed_map_masked = seed_map[mask].view(1, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            # if 1, mark as unclustered
            # unclustered = torch.ones(mask.sum()).byte().to(seed_map.device)
            unclustered = torch.ones(mask.sum()).long().to(seed_map.device)
            # if zero, mark as unclustered
            # instance_map_masked = torch.zeros(mask.sum()).byte().to(seed_map.device)
            instance_map_masked = torch.zeros(mask.sum()).long().to(seed_map.device)

            while (unclustered.sum() > self.proposal_thrs):
                # find the max score seed and its location
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < self.threshold:
                    break
                # x, y of center as shape: [2,1]
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, dim=0, keepdim=True))

                # proposal is a group of pixels which near to seed which regarded as center
                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > self.proposal_thrs:
                    # if over half of pixels in proposal have not been regard as a instance, see it as a instacne
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        # mark pixels belong to proposal as 'count'
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        # instance_mask = torch.zeros(height, width).long()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        # instance_mask[mask.squeeze().cpu()] = proposal.cpu().long()
                        if self.n_class in (20, 21):
                            # voc
                            # {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': start_with_0})
                            instances.append(
                                {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': cls})
                            instance_cls[count] = cls + 1
                        elif self.n_class in (15, 16):
                            # iSAID
                            mask_np = np.array(instance_mask).copy(order='F')
                            mask_encode = coco_mask.encode(mask_np)
                            mask_encode['counts'] = str(mask_encode['counts'], encoding='utf-8')
                            instances.append(
                                {'mask': mask_encode, 'score': seed_score, 'cls': cls + 1})
                            instance_cls[count] = cls + 1
                            pass
                        elif self.n_class in (8, 9):
                            # will mapping with a class_ids
                            instances.append(
                                {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': class_ids[cls]})
                            instance_cls[count] = class_ids[cls]
                        count += 1
                        ### 20200217
                        if count > self.max_dets:
                            break
                unclustered[proposal] = 0

                instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map, instances, instance_cls


class Cluster:
    def __init__(self, args):
        bkgd = (args.n_class in (9, 16, 21))
        n_sigma = args.n_sigma
        threshold = args.threshold
        n_class = args.n_class
        ### 20200217
        self.proposal_thrs = 128
        self.max_dets = 1024
        ###
        print(f'===> Clustering config: bkgd:{bkgd} | n_sigma:{n_sigma} | threshold:{threshold}')
        if args.n_class in (8, 9):
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        elif args.n_class in (20, 21):
            xm = torch.linspace(0, 1, 512).view(1, 1, -1).expand(1, 512, 512)
            ym = torch.linspace(0, 1, 512).view(1, -1, 1).expand(1, 512, 512)
        elif args.n_class in (15, 16):
            xm = torch.linspace(0, 10, 8000).view(1, 1, -1).expand(1, 8000, 8000)
            ym = torch.linspace(0, 10, 8000).view(1, -1, 1).expand(1, 8000, 8000)
            self.proposal_thrs = 16
        else:
            raise NotImplementedError
        xym = torch.cat((xm, ym), 0)
        print(f'xym:{xym.shape}')
        self.bkgd = bkgd
        self.xym = xym
        self.n_sigma = n_sigma
        self.n_class = n_class
        self.threshold = threshold

    def cluster_with_gt(self, prediction, instance):
        """
        using gt to get center, and cal the clustering of prediction.

        :param prediction:
        :param instance:
        :param n_sigma:
        :return:
        """
        assert prediction.dim() == 3
        assert instance.dim() == 3

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
        if prediction.is_cuda:
            xym_s = xym_s.cuda()

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).byte().to(prediction.device)

        # instances have no bkgd
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, height, width)

            center = spatial_emb[mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

            s = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1).mean(1).view(self.n_sigma, 1, 1)
            s = torch.exp(s * 10)  # n_sigma x 1 x 1

            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))

            proposal = (dist > 0.5)
            instance_map[proposal] = id

        return instance_map

    def cluster(self, prediction):
        """
        for each seed map:
        1. find the max seed score and the location
        2. get the clustering radius by the sigma of max seed:
            s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, dim=0, keepdim=True))
        3. get all pixels in clustering scope, regard them as a proposal
        4. a proposal must meet two conditions to be a instance:
            a. more than half of pixels have not been marked as instance yet
            b. the number of pixels > self.proposal_thrs


        :param prediction:
        :param n_sigma:
        :param threshold:
        :return:
            instance_map [h, w]
            instances, a dict of 'mask', 'score', 'cls'
            instance_cls, a dict of instance_id:cls_id

        """
        assert prediction.dim() == 3
        torch.cuda.empty_cache()
        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        if prediction.is_cuda:
            xym_s = xym_s.cuda()

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + self.n_sigma:])  # cls x h x w

        n_class = seed_map.size(0)

        # 1xhxw
        seed_map_max, seed_map_argmax = seed_map.max(dim=0, keepdim=True)

        if self.bkgd:
            seed_map_max[seed_map_argmax == 0] = 0

        instance_map = torch.zeros(height, width).byte()
        instances = []
        instance_cls = {}

        # instance counter
        count = 1
        mask = seed_map_max > 0.5

        if mask.sum() > self.proposal_thrs:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)

            seed_map_masked = seed_map_max[mask].view(1, -1)
            seed_map_argmax_masked = seed_map_argmax[mask].view(1, -1)

            # if 1, mark as unclustered
            unclustered = torch.ones(mask.sum()).byte().to(prediction.device)
            # if zero, mark as unclustered
            instance_map_masked = torch.zeros(mask.sum()).byte().to(prediction.device)

            while (unclustered.sum() > self.proposal_thrs):
                # find the max score seed and its location
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                cls = int(seed_map_argmax_masked[0, seed])
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < self.threshold:
                    break
                # x, y of center as shape: [2,1]
                center = spatial_emb_masked[:, seed:seed + 1]
                unclustered[seed] = 0
                s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                          center, 2) * s, dim=0, keepdim=True))

                # proposal is a group of pixels which near to seed which regarded as center
                proposal = (dist > 0.5).squeeze()

                if proposal.sum() > self.proposal_thrs:
                    # if over half of pixels in proposal have not been regard as a instance, see it as a instacne
                    if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                        # mark pixels belong to proposal as 'count'
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        if n_class > 9:
                            # voc
                            instances.append(
                                {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': cls})
                            instance_cls[count] = cls + 1
                        else:
                            if self.bkgd:
                                instances.append(
                                    {'mask': instance_mask.squeeze() * 255, 'score': seed_score,
                                     'cls': class_ids[cls - 1]})
                                instance_cls[count] = class_ids[cls - 1]
                            else:
                                instances.append(
                                    {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': class_ids[cls]})
                                instance_cls[count] = class_ids[cls]
                        count += 1
                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map, instances, instance_cls

    def cluster_for_each(self, prediction):
        """
        do not include bkgd
        :param spatial_emb:
        :param sigma:
        :param seed_map:
        :param cls:
        :param threshold:
        :return: cls_wo_bkgd
        """
        assert prediction.dim() == 3

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        # 20200229 modified
        if prediction.is_cuda:
            xym_s = xym_s.cuda()
        # prediction = prediction.cpu()
        # xym_s = xym_s.cpu()

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        seed_map_all = torch.sigmoid(prediction[2 + self.n_sigma:])  # cls x h x w

        # 1xhxw
        n_sigma = sigma.size(0)
        n_class, height, width = seed_map_all.shape
        assert n_class in (8, 9, 15, 16, 20, 21)
        if n_class == 9:
            seed_map_all = seed_map_all[1:]
            n_class = 8

        # instance_map = torch.zeros(height, width).byte()
        instance_map = torch.zeros(height, width).long()
        instances = []
        instance_cls = {}

        pool = Pool(processes=3)
        cluster_mp = ClusterMP(spatial_emb, sigma, seed_map_all, self.threshold, self.proposal_thrs, self.max_dets)
        returnvals = pool.map(cluster_mp, [i for i in range(n_class)])
        pool.close()
        for cls_res in returnvals:
            cls_instance_map, cls_instances, cls_instance_cls = cls_res
            instance_map[cls_instance_map > 0] = cls_instance_map[cls_instance_map > 0]
            instance_map[instance_map > 0] += cls_instance_map.max()
            for i in cls_instances:
                instances.append(i)
            for i in cls_instance_cls:
                instance_cls[i] = cls_instance_cls[i]

        return instance_map, instances, instance_cls

    def cluster_for_each_iter(self, prediction, iter=10):
        """
        do not include bkgd
        :param spatial_emb:
        :param sigma:
        :param seed_map:
        :param cls:
        :param threshold:
        :return: cls_wo_bkgd
        """
        assert prediction.dim() == 3

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]
        if prediction.is_cuda:
            xym_s = xym_s.cuda()

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        seed_map_all = torch.sigmoid(prediction[2 + self.n_sigma:])  # cls x h x w

        # 1xhxw
        n_sigma = sigma.size(0)
        n_class, height, width = seed_map_all.shape
        assert n_class in (8, 9, 20, 21)
        if n_class == 9:
            seed_map_all = seed_map_all[1:]
            n_class = 8

        instance_map = torch.zeros(height, width).byte()
        instances = []
        instance_cls = {}

        '''
        import multiprocessing
        cores = len(worklist)
        pool = multiprocessing.Pool(processes=cores)
        pool.map(os.system, worklist)
        '''

        # instance counter
        count = 1
        for cls in range(n_class):
            torch.cuda.empty_cache()
            seed_map = seed_map_all[cls]
            mask = seed_map > 0.5
            if mask.sum() > self.proposal_thrs:
                spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
                sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)

                # seed_map_masked = seed_map[mask].view(1, -1)
                seed_map_masked = seed_map[mask].view(1, -1)

                # if 1, mark as unclustered
                unclustered = torch.ones(mask.sum()).byte().to(seed_map.device)
                # if zero, mark as unclustered
                instance_map_masked = torch.zeros(mask.sum()).byte().to(seed_map.device)

                while (unclustered.sum() > self.proposal_thrs):
                    # find the max score seed and its location
                    seed = (seed_map_masked * unclustered.float()).argmax().item()
                    seed_score = (seed_map_masked * unclustered.float()).max().item()
                    if seed_score < self.threshold:
                        break
                    # x, y of center as shape: [2,1]
                    center = spatial_emb_masked[:, seed:seed + 1]
                    unclustered[seed] = 0
                    s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                    dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                              center, 2) * s, dim=0, keepdim=True))
                    if (dist > 0.5).sum() > self.proposal_thrs:
                        for i in range(iter):
                            center = spatial_emb_masked[(dist > 0.5).expand_as(spatial_emb_masked)] \
                                .view(2, -1).mean(1, keepdim=True)
                            s = torch.exp(sigma_masked[(dist > 0.5).expand_as(sigma_masked)]
                                          .view(2, -1).mean(1, keepdim=True) * 10)
                            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                                      center, 2) * s, dim=0, keepdim=True))
                            pass

                    # proposal is a group of pixels which near to seed which regarded as center
                    proposal = (dist > 0.5).squeeze()

                    if proposal.sum() > self.proposal_thrs:
                        # if over half of pixels in proposal have not been regard as a instance, see it as a instacne
                        if unclustered[proposal].sum().float() / proposal.sum().float() > 0.5:
                            # mark pixels belong to proposal as 'count'
                            instance_map_masked[proposal.squeeze()] = count
                            instance_mask = torch.zeros(height, width).byte()
                            instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                            if n_class > 9:
                                # voc
                                instances.append(
                                    {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': cls})
                                instance_cls[count] = cls + 1
                            else:
                                instances.append(
                                    {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': class_ids[cls]})
                                instance_cls[count] = class_ids[cls]
                            count += 1
                    unclustered[proposal] = 0

                instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map, instances, instance_cls


class_names = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
class_ids = (24, 25, 26, 27, 28, 31, 32, 33)


class Logger:

    def __init__(self, keys, title=""):

        self.data = {k: [] for k in keys}
        self.title = title
        self.win = None

        print('created logger with keys:  {}'.format(keys))

    def plot(self, save=False, save_dir=""):
        size_list = [len(self.data[k]) for k in self.data]
        min_size = min(size_list)
        for k in self.data:
            self.data[k] = self.data[k][-min_size:]

        if self.win is None:
            self.win = plt.subplots()
        fig, ax = self.win
        ax.cla()

        keys = []
        for key in self.data:
            keys.append(key)
            data = self.data[key]
            ax.plot(range(len(data)), data, marker='.')

        ax.legend(keys, loc='upper right')
        ax.set_title(self.title)

        plt.draw()
        Visualizer.mypause(0.001)

        if save:
            # save figure
            fig.savefig(os.path.join(save_dir, self.title + '.png'))

            # save data as csv
            df = pd.DataFrame.from_dict(self.data)
            df.to_csv(os.path.join(save_dir, self.title + '.csv'))

    def add(self, key, value):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)


if __name__ == '__main__':
    from train_config import get_config

    args = get_config()
    func = get_cluster(args)
    print(func)
