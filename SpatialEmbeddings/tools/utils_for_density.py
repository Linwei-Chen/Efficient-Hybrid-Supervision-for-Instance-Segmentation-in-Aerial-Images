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

import torch


class DensityCluster:

    def __init__(self, args):
        bkgd = args.n_class == 9
        n_sigma = args.n_sigma
        threshold = args.threshold
        n_class = args.n_class
        print(f'===> Clustering config: bkgd:{bkgd} | n_sigma:{n_sigma} | threshold:{threshold}')
        if args.n_class in (8, 9):
            xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        elif args.n_class in (20, 21):
            xm = torch.linspace(0, 1, 512).view(1, 1, -1).expand(1, 512, 512)
            ym = torch.linspace(0, 1, 512).view(1, -1, 1).expand(1, 512, 512)
        else:
            raise NotImplementedError
        xym = torch.cat((xm, ym), 0)
        print(f'xym:{xym.shape}')
        xym = torch.cat((xm, ym), 0)
        self.bkgd = bkgd
        self.xym = xym
        self.n_sigma = n_sigma
        self.threshold = threshold
        self.n_class = args.n_class

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
        density_map = prediction[2 + self.n_sigma:4 + self.n_sigma]  # 2 x h x w
        xm = (density_map[0].softmax(dim=-1) * float(width) / 1024.).cumsum(dim=-1)  # w->x
        ym = (density_map[1].softmax(dim=-2) * float(height) / 1024.).cumsum(dim=-2)  # h->y
        xym_s = torch.cat([xm.unsqueeze(dim=0), ym.unsqueeze(dim=0)], 0)

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

            proposal = (dist >= 0.5)
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
            b. the number of pixels > 128


        :param prediction:
        :param n_sigma:
        :param threshold:
        :return:
            instance_map [h, w]
            instances, a dict of 'mask', 'score', 'cls'
            instance_cls, a dict of instance_id:cls_id

        """
        assert prediction.dim() == 3

        height, width = prediction.size(1), prediction.size(2)
        density_map = prediction[2 + self.n_sigma:4 + self.n_sigma]  # 2 x h x w
        xm = (density_map[0].softmax(dim=-1) * float(width) / 1024.).cumsum(dim=-1)  # w->x
        ym = (density_map[1].softmax(dim=-2) * float(height) / 1024.).cumsum(dim=-2)  # h->y
        xym_s = torch.cat([xm.unsqueeze(dim=0), ym.unsqueeze(dim=0)], 0)

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + self.n_sigma:])  # cls x h x w

        n_class = seed_map.size(0)

        # 1xhxw
        seed_map_max, seed_map_argmax = seed_map.max(dim=0, keepdim=True)

        if self.bkgd:
            seed_map_max[seed_map_argmax == 0] = 0

        # instance_map = torch.zeros(height, width).long()
        instance_map = torch.zeros(height, width).byte()
        instances = []
        instance_cls = {}

        # instance counter
        count = 1
        mask = seed_map_max >= 0.5

        if mask.sum() > 128:
            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
            sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)

            seed_map_masked = seed_map_max[mask].view(1, -1)
            seed_map_argmax_masked = seed_map_argmax[mask].view(1, -1)

            # if 1, mark as unclustered
            unclustered = torch.ones(mask.sum()).byte().to(prediction.device)
            # if zero, mark as unclustered
            instance_map_masked = torch.zeros(mask.sum()).byte().to(prediction.device)

            while (unclustered.sum() > 128):
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
                proposal = (dist >= 0.5).squeeze()

                if proposal.sum() > 128:
                    # if over half of pixels in proposal have not been regard as a instance, see it as a instacne
                    if unclustered[proposal].sum().float() / proposal.sum().float() >= 0.5:
                        # mark pixels belong to proposal as 'count'
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()

                        if n_class > 9:
                            # voc
                            instances.append(
                                {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': cls})
                            instance_cls[count] = class_ids[cls]
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
        :return:
        """
        assert prediction.dim() == 3

        height, width = prediction.size(1), prediction.size(2)
        density_map = prediction[2 + self.n_sigma:4 + self.n_sigma]  # 2 x h x w
        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
        if (self.n_class + 6) == prediction.size(-3):
            xm = (density_map[0].softmax(dim=-1) * float(width) / 1024.).cumsum(dim=-1)  # w->x
            # ym = xym_s[1].to(prediction.device)  # 2 x h x w
            ym = (density_map[1].softmax(dim=-2) * float(height) / 1024.).cumsum(dim=-2)  # h->y
            seed_map_all = prediction[4 + self.n_sigma:4 + self.n_sigma + self.n_class].sigmoid()
        elif (self.n_class + 5) == prediction.size(-3):
            xm = (density_map[0].softmax(dim=-1) * float(width) / 1024.).cumsum(dim=-1)  # w->x
            ym = xym_s[1].to(prediction.device)  # 2 x h x w
            seed_map_all = prediction[3 + self.n_sigma:3 + self.n_sigma + self.n_class].sigmoid()
        else:
            raise NotImplementedError
        xym_s = torch.cat([xm.unsqueeze(dim=0), ym.unsqueeze(dim=0)], 0)

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w

        # 1xhxw
        n_sigma = sigma.size(0)
        n_class, height, width = seed_map_all.shape
        assert n_class in (8, 9, 20)
        if n_class == 9:
            seed_map_all = seed_map_all[1:]
            n_class = 8

        instance_map = torch.zeros(height, width).byte()
        # instance_map = torch.zeros(height, width).long()
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
            seed_map = seed_map_all[cls]
            mask = seed_map >= 0.5
            if mask.sum() > 128:
                spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
                sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)

                # seed_map_masked = seed_map[mask].view(1, -1)
                seed_map_masked = seed_map[mask].view(1, -1)

                # if 1, mark as unclustered
                unclustered = torch.ones(mask.sum()).byte().to(seed_map.device)
                # if zero, mark as unclustered
                instance_map_masked = torch.zeros(mask.sum()).byte().to(seed_map.device)

                while (unclustered.sum() > 128):
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
                    proposal = (dist >= 0.5).squeeze()

                    if proposal.sum() > 128:
                        # if over half of pixels in proposal have not been regard as a instance, see it as a instacne
                        if unclustered[proposal].sum().float() / proposal.sum().float() >= 0.5:
                            # mark pixels belong to proposal as 'count'
                            instance_map_masked[proposal.squeeze()] = count
                            instance_mask = torch.zeros(height, width).byte()
                            instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                            if n_class > 9:
                                # voc
                                instances.append(
                                    {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': cls})
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


def display(args):
    if args.display:
        plt.ion()
    else:
        plt.ioff()
        plt.switch_backend("agg")


# Visualizer
# visualizer = Visualizer()
class VisualizeToolForDensity:
    def __init__(self, args):
        from tools.utils import Visualizer, display
        self.args = args
        display(args)
        self.visualizer = Visualizer(keys=('image', 'xy', 'sigma', 'seed', 'label', 'instance', 'density'))

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
            ########
            if 'densityx' in self.args.model_name:
                density_map = prediction[2 + self.args.n_sigma:3 + self.args.n_sigma]
                seed_map = prediction[3 + self.args.n_sigma:]
            else:
                density_map = prediction[2 + self.args.n_sigma:4 + self.args.n_sigma]
                seed_map = prediction[4 + self.args.n_sigma:]

            n_class = seed_map.size(0)
            prob, cls = seed_map.max(dim=0)
            if n_class == 8:
                cls = cls + 1
                cls = cls.float()
                cls[prob < 0.5] = 0.
                prob[prob < 0.5] = 0.

            elif n_class == 9:
                prob[cls == 0] = 0
                cls = cls.float()
            elif n_class == 20:
                cls = cls + 1
                cls = cls.float()
                cls[prob < 0.5] = 0.
                prob[prob < 0.5] = 0.
            else:
                raise NotImplementedError

            # print('===> cls:', cls.unique().data)

        image = image.cpu()
        label = label.cpu()
        instance = instance.cpu()
        density_map = density_map.cpu()
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
        # show instance
        if 'densityx' in self.args.model_name:
            self.visualizer.display(density_map[0].softmax(dim=-1), 'density')
        else:
            self.visualizer.display([density_map[0].softmax(dim=-1),
                                     density_map[1].softmax(dim=-2)], 'density')

        return cls.unique().data
