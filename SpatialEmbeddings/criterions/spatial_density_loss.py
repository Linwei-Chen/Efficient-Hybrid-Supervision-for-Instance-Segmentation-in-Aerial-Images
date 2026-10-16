"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import math

import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from criterions.lovasz_losses import lovasz_hinge
import random


class SpatialDensityEmbLoss(nn.Module):

    def __init__(self, args):

        to_center = 'to_center' in args.criterion
        n_sigma = args.n_sigma
        foreground_weight = args.foreground_weight
        w_inst = args.loss_w['w_inst']
        w_var = args.loss_w['w_var']
        w_seed = args.loss_w['w_seed']
        n_class = args.n_class

        super().__init__()

        print('===> Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
            to_center, n_sigma, foreground_weight))
        print(f'===> w_inst:{w_inst} | w_var:{w_var} | w_seed:{w_seed}')

        self.n_class = n_class
        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight
        self.w_inst = w_inst
        self.w_var = w_var
        self.w_seed = w_seed

        # coordinate map

        xm = torch.linspace(0, 2, 2048).view(
            1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(
            1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)
        self.register_buffer("xym", xym)

    def forward(self, prediction, instances, labels, iou_meter=None):
        torch.cuda.empty_cache()

        assert prediction.dim() == 4
        assert instances.dim() == 4
        assert labels.dim() == 4

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0

        for b in range(0, batch_size):
            torch.cuda.empty_cache()
            density_map = prediction[b, 2 + self.n_sigma:4 + self.n_sigma]  # 2 x h x w
            h, w = density_map.size(-2), density_map.size(-1)
            # density_map[0] = density_map[0].softmax(dim=-1) * float(w) / 1024.
            # density_map[1] = density_map[1].softmax(dim=-2) * float(h) / 1024.
            if (self.n_class + 6) == prediction.size(-3):
                xm = (density_map[0].softmax(dim=-1) * float(w) / 1024.).cumsum(dim=-1)  # w->x
                ym = xym_s[1].to(prediction.device)  # 2 x h x w
                # ym = (density_map[1].softmax(dim=-2) * float(h) / 1024.).cumsum(dim=-2)  # h->y
                seed_map = prediction[b, 4 + self.n_sigma:4 + self.n_sigma + self.n_class].sigmoid()
            elif (self.n_class + 5) == prediction.size(-3):
                xm = (density_map[0].softmax(dim=-1) * float(w) / 1024.).cumsum(dim=-1)  # w->x
                ym = xym_s[1].to(prediction.device)  # 2 x h x w
                seed_map = prediction[b, 3 + self.n_sigma:3 + self.n_sigma + self.n_class].sigmoid()
            else:
                raise NotImplementedError
            xym_s = torch.cat([xm.unsqueeze(dim=0), ym.unsqueeze(dim=0)], 0)
            spatial_emb = prediction[b, 0:2].tanh() + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2 + self.n_sigma]  # n_sigma x h x w
            # 1x cls(no bkgd) x h x w

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b]  # 1 x h x w
            label = labels[b]  # 1 x h x w

            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}
            # print(instance_to_cls)
            # instance_ids = instance_ids[instance_ids != 0]

            # regress bg to zero, cls_id 0~7, label 0~8
            for cls_id in range(self.n_class):
                bg_mask = ((label != (cls_id + 1)) & (label != 255))

                if bg_mask.sum() > 0:
                    cls_seed_map = seed_map[cls_id].unsqueeze(dim=0)
                    seed_loss += torch.sum(
                        torch.pow(cls_seed_map[bg_mask] - 0, 2))

            for id in instance_ids:

                in_mask = instance.eq(id)  # 1 x h x w
                # calculate center of attraction
                if self.to_center:
                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)  # n_sigma x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + \
                           torch.mean(torch.pow(sigma_in - s.detach(), 2))

                s = torch.exp(s * 10)

                # calculate gaussian
                dist = torch.exp(-1 * torch.sum(
                    torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))

                # apply lovasz-hinge loss. dist > 0.5
                inst_mask = in_mask.clone()
                inst_mask[label == 255] = 255
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, inst_mask, ignore=255)

                # seed loss
                instance_cls = int(instance_to_cls[int(id)]) - 1
                cls_seed_map = seed_map[instance_cls].unsqueeze(dim=0)
                seed_loss += self.foreground_weight[instance_cls] * torch.sum(
                    torch.pow(cls_seed_map[in_mask] - dist[in_mask].detach(), 2))

                # calculate instance iou
                if iou_meter is not None:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += self.w_inst * instance_loss + self.w_var * var_loss + self.w_seed * seed_loss

        loss = loss / batch_size
        return loss + prediction.sum() * 0


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou


if __name__ == '__main__':
    x = torch.rand(2, 9, 8, 8)
    y = torch.randint(low=0, high=2, size=(2, 1, 8, 8))
