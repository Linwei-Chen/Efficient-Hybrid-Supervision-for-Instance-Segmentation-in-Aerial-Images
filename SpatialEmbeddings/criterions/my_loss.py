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


class SpatialEmbLoss(nn.Module):

    def __init__(self, args):
        super().__init__()
        to_center = 'to_center' in args.criterion
        n_sigma = args.n_sigma
        foreground_weight = args.foreground_weight
        w_inst = args.loss_w['w_inst']
        w_var = args.loss_w['w_var']
        w_seed = args.loss_w['w_seed']
        n_class = args.n_class

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

        # coordinate map, yx_map actually
        if args.n_class in (8, 9):
            xm = torch.linspace(0, 2, 2048).view(
                1, 1, -1).expand(1, 1024, 2048)
            ym = torch.linspace(0, 1, 1024).view(
                1, -1, 1).expand(1, 1024, 2048)
        elif args.n_class in (20, 21):
            xm = torch.linspace(0, 1, 512).view(
                1, 1, -1).expand(1, 512, 512)
            ym = torch.linspace(0, 1, 512).view(
                1, -1, 1).expand(1, 512, 512)
        elif args.n_class in (15, 16):
            xm = torch.linspace(0, 1, 800).view(
                1, 1, -1).expand(1, 800, 800)
            ym = torch.linspace(0, 1, 800).view(
                1, -1, 1).expand(1, 800, 800)
        else:
            raise NotImplementedError
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
            spatial_emb = prediction[b, 0:2].tanh() + xym_s  # 2 x h x w
            sigma = prediction[b, 2:2 + self.n_sigma]  # n_sigma x h x w
            seed_map = prediction[b, 2 + self.n_sigma:2 + self.n_sigma + self.n_class].sigmoid()
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
            ### 20200218
            # set max instances for training
            if len(instance_ids)>100:
                random.shuffle(instance_ids)
                instance_ids = instance_ids[:100]
                instance_to_cls = {int(k):instance_to_cls[int(k)] for k in instance_ids}
            ###
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


from criterions.lovasz_losses import lovasz_softmax, lovasz_hinge


class LovaszBCE(nn.Module):
    def __init__(self, per_image=True, ignore_label=255):
        super().__init__()
        self.ignore = ignore_label
        self.per_image = per_image

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Binary Lovasz hinge loss
          logits: [B, cls, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, 1, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """

        assert input.dim() == 4
        batch_size = input.size(0)
        n_class = input.size(1)
        input = input.tanh()
        target_one_hot = label_to_one_hot(target, n_class, with_255=True)
        loss = 0
        for b in range(batch_size):
            # when:
            # 1. mask = 1, and pre_logit > 1, loss will be zero
            # 2. mask = 0, and pre_logit < -1, loss will be zero
            loss += lovasz_hinge(logits=input[b], labels=target_one_hot[b],
                                 per_image=self.per_image, ignore=self.ignore)
        return loss / batch_size


class BCE2D:
    def __init__(self, n_calss=21, ignore_label=255):
        self.n_class = n_calss
        self.BCELoss = nn.BCELoss(reduction='none')

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        target_one_hot = label_to_one_hot(target, n_class=self.n_class)
        pt = input.sigmoid()
        BCE = self.BCELoss(pt, target_one_hot)
        BCE[target.expand(BCE.shape) >= self.n_class] = 0
        # print(BCE)
        return BCE.mean()


class LovaszBCEWithBCE(nn.Module):
    def __init__(self, n_class, ignore_label=255, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.l = LovaszBCE(ignore_label=ignore_label)
        self.bce = BCE2D(n_class, ignore_label=ignore_label)

    def forward(self, input, target):
        """
        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """
        lovaszloss = self.alpha * self.l(input, target)
        bceloss = self.bce(input, target)
        return lovaszloss + bceloss


class LovaszSoftMax(nn.Module):
    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return lovasz_softmax(input.softmax(dim=1), target.squeeze(dim=1), ignore=self.ignore_label)


class CrossEntropyLoss2D(nn.Module):
    def __init__(self, n_calss=21, ignore_index=255, reduction='mean', weight=None):
        super().__init__()
        self.n_class = n_calss
        self.CELoss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        if input.dim() > 2:
            input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
            target = target.transpose(1, 2).transpose(2, 3).reshape(-1, 1)

        CE = self.CELoss(input, target.view(-1))
        return CE


class LovaszSoftMaxWithCE(nn.Module):
    def __init__(self, n_class, ignore_label=255, alpha=1):
        super().__init__()
        self.alpha = alpha
        self.l = LovaszSoftMax(ignore_label=ignore_label)
        self.ce = CrossEntropyLoss2D(n_class, ignore_index=ignore_label)

    def forward(self, input, target):
        """
        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """
        lovaszloss = self.alpha * self.l(input, target)
        celoss = self.ce(input, target)
        return lovaszloss + celoss


def label_to_one_hot(targets: torch.Tensor, n_class: int, with_255: bool = False):
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

    one_hot = torch.nn.functional.one_hot(targets, num_classes=n_class)
    if with_255:
        one_hot[targets_ignore] = 0
    else:
        one_hot[targets_ignore] = 255
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


class FocalLoss(nn.Module):

    def __init__(self,
                 mode='ce', n_class=21, mean=True,
                 gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        # self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.mode = mode
        self.n_class = n_class
        self.mean = mean

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        """

        :param input: [bs, c, h, w],
        :param target: [bs, 1, h, w]
        :return: tensor
        """

        if self.mode == 'bce':
            target_one_hot = label_to_one_hot(target, n_class=self.n_class)
            pt = input.sigmoid()
            BCE = nn.BCELoss(reduction='none')(pt, target_one_hot)
            BCE[target.expand(BCE.shape) >= self.n_class] = 0
            # print(BCE)
            loss = torch.abs(target_one_hot - pt.detach()) ** self.gamma * BCE

        elif self.mode == 'ce':
            if input.dim() > 2:
                input = input.transpose(1, 2).transpose(2, 3).reshape(-1, self.n_class)
                target = target.transpose(1, 2).transpose(2, 3).reshape(-1, 1)

            pt = input.softmax(dim=1)
            pt = pt.detach()
            CE = nn.CrossEntropyLoss(reduction='none', ignore_index=255)(input, target.view(-1))
            target = torch.where(target < self.n_class, target, torch.zeros(target.shape, device=target.device).long())

            pt = pt.gather(dim=1, index=target).view(-1)
            # print(f'pt:{pt}')

            # print(f'CE:{CE.shape}\nCE:{CE}')
            loss = (1 - pt) ** self.gamma * CE
        else:
            raise Exception(f'*** focal loss mode:{self.mode} wrong!')

        if self.mean:
            return loss.mean()
        else:
            return loss.sum()


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
    print(LovaszBCEWithBCE(n_class=9)(x, y))
