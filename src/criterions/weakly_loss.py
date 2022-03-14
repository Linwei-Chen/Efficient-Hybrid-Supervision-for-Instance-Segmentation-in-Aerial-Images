import math

import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
from criterions.lovasz_losses import lovasz_hinge
import random


class WeaklySpatialEmbLoss(nn.Module):

    def __init__(self, n_class=8, to_center=False, n_sigma=2,
                 foreground_weight=(10, 10, 10, 40, 80, 100, 60, 20), w_inst=1, w_var=10, w_seed=1):
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

    def forward(self, prediction, instances_batch, bbox, iou_meter=None):
        """

        :param prediction: [xy, sigma, cls]
        :param instances_batch: list of [{'mask':, 'cls' }, ...]
        :param bbox: semantic bbox
        :param iou_meter:
        :return:
        """
        torch.cuda.empty_cache()

        assert prediction.dim() == 4
        assert bbox.dim() == 4

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # 2 x h x w

        loss = 0

        for b in range(0, batch_size):
            torch.cuda.empty_cache()
            spatial_emb = prediction[b, 0:2].tanh() + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2 + self.n_sigma]  # n_sigma x h x w
            seed_map = prediction[b, 2 + self.n_sigma:2 + self.n_sigma + self.n_class].sigmoid()
            # 1x cls(no bkgd) x h x w

            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instances = instances_batch[b]  # a dict of list of {'mask', 'score', 'cls'}
            bbox_b = bbox[b]

            for cls_index in range(1, 9):
                # bkgd using bbox
                bg_mask = bbox_b[cls_index] == 0
                if bg_mask.sum() > 0:
                    seed_loss += torch.sum(
                        torch.pow(seed_map[cls_index - 1][bg_mask] - 0, 2))

            # regress bg to zero, cls_id 0~7, label 0~8
            for instance in instances:
                # from 1~8 to 0~7
                try:
                    # (24, 25, 26, 27, 28, 31, 32, 33)
                    cls_index = class_ids_to_index[instance['cls']] - 1
                except Exception:
                    # 1~8
                    cls_index = instance['cls'] - 1

                seed_map_cls = seed_map[cls_index]

                # instance using pred
                mask_np = instance['mask']
                mask = torch.from_numpy(mask_np).unsqueeze(dim=0).float()
                in_mask = mask > 0  # 1 x h x w
                in_mask = in_mask.to(xym_s.device)

                # or it will show NAN
                if in_mask.sum() < 128:
                    continue

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
                # (between -\infty and +\infty)
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, in_mask)

                # seed loss
                seed_loss += self.foreground_weight[cls_index] * torch.sum(
                    torch.pow(seed_map_cls.unsqueeze(dim=0)[in_mask] - dist[in_mask].detach(), 2))

                # calculate instance iou
                if iou_meter is not None:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1
                # print(f'instance_loss:{instance_loss}')

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            # print(f'instance_loss:{instance_loss.item()}')
            # print(f'var_loss:{var_loss.item()}')
            # print(f'seed_loss:{seed_loss.item()}')
            # print(f'instance_loss:{instance_loss}')
            # print(f'var_loss:{var_loss}')
            # print(f'seed_loss:{seed_loss}')

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
