import torch
from datasets.CityscapesDataset import class_ids_to_index
import numpy as np
from tools.utils import class_ids
from torch import nn
import math
import cv2


def get_bbox_info_from_mask(instance: torch.Tensor, label: torch.Tensor):
    """
    notice that bottom and right is extreme value, + 1 when use it as bbox
    :param instance:
    :param label:
    :return:
    """
    assert instance.dim() == 3 or instance.dim() == 2
    assert label.dim() == 3 or label.dim() == 2

    if instance.dim() == 3:
        instance = instance.squeeze(dim=0)

    if label.dim() == 3:
        label = label.squeeze(dim=0)

    instance_ids = instance.unique()
    instance_ids = instance_ids[instance_ids != 0]

    instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}
    # print(instance_to_cls)
    instance_np = instance.cpu().numpy()
    res = []
    for inst_id in instance_to_cls:
        index = np.argwhere(instance_np == inst_id)
        ul = index.min(axis=0)
        br = index.max(axis=0)
        bbox = list(ul) + list(br)
        hw = br - ul
        area = int(hw[1] + 1) * int(hw[0] + 1)
        res.append({
            'cls': instance_to_cls[inst_id],
            'area': area,
            'bbox': bbox,
            'instance_id': inst_id
        })
    res.sort(key=lambda x: x['area'], reverse=True)
    return res


def get_obbox_info_from_mask(instance: torch.Tensor, label: torch.Tensor):
    """
    notice that bottom and right is extreme value, + 1 when use it as bbox
    :param instance:
    :param label:
    :return:
    """
    assert instance.dim() == 3 or instance.dim() == 2
    assert label.dim() == 3 or label.dim() == 2

    if instance.dim() == 3:
        instance = instance.squeeze(dim=0)

    if label.dim() == 3:
        label = label.squeeze(dim=0)

    instance_ids = instance.unique()
    instance_ids = instance_ids[instance_ids != 0]

    instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}
    # print(instance_to_cls)
    instance_np = instance.cpu().numpy()
    res = []
    for inst_id in instance_to_cls:
        index = np.argwhere(instance_np == inst_id)
        ul = index.min(axis=0)
        br = index.max(axis=0)
        bbox = list(ul) + list(br)
        hw = br - ul
        area = int(hw[1] + 1) * int(hw[0] + 1)
        cnt = np.array([[j, i] for i, j in index])
        rect = cv2.minAreaRect(cnt)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        obbox = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
        obbox = np.int0(obbox)
        # print(obbox)
        # cv2.fillPoly(img, [box], 255)
        res.append({
            'cls': instance_to_cls[inst_id],
            'area': area,
            'bbox': bbox,
            'instance_id': inst_id,
            'obbox': obbox
        })
    res.sort(key=lambda x: x['area'], reverse=True)
    return res


def get_bbox_instance_np(instance: torch.Tensor, label: torch.Tensor):
    """
    get instance bbox mask from instance mask and label
    :param instance:  a mask of instance
    :param label: a mask of label
    :return:
    """
    # instances_dict
    info = get_bbox_info_from_mask(instance, label)
    h, w = instance.size(-2), instance.size(-1)
    bbox_instances = []
    for inst in info:
        instance_mask = torch.zeros(h, w).to(instance.device)
        u, l, b, r = inst['bbox']
        instance_mask[u:b + 1, l:r + 1] = 1
        temp = {'mask': instance_mask.cpu().numpy(), 'cls': inst['cls']}
        bbox_instances.append(temp)
    # print(bbox_instances)
    return bbox_instances


def get_bbox_instance_label_mask(instance: torch.Tensor, label: torch.Tensor):
    """
    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return: tensor
    """
    info = get_bbox_info_from_mask(instance, label)
    h, w = instance.size(-2), instance.size(-1)
    bbox_instance_mask = torch.zeros(h, w).to(instance.device)
    bbox_label_mask = torch.zeros(h, w).to(instance.device)
    for inst in info:
        u, l, b, r = inst['bbox']
        bbox_instance_mask[u:b + 1, l:r + 1] = inst['instance_id']
        bbox_label_mask[u:b + 1, l:r + 1] = inst['cls']
    return bbox_instance_mask, bbox_label_mask


def get_bbox_instance_label_mask_batch(instances: torch.Tensor, labels: torch.Tensor):
    """

    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return:
    """
    batch_size = instances.size(0)
    bbox_instance_masks, bbox_label_masks = [], []
    for i in range(batch_size):
        bbox_instance_mask, bbox_label_mask = get_bbox_instance_label_mask(instances[i], labels[i])
        bbox_instance_masks.append(bbox_instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        bbox_label_masks.append(bbox_label_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    bbox_instance_masks = torch.cat(bbox_instance_masks, dim=0)
    bbox_label_masks = torch.cat(bbox_label_masks, dim=0)
    return bbox_instance_masks, bbox_label_masks


def get_obbox_instance_label_mask(instance: torch.Tensor, label: torch.Tensor):
    res = get_obbox_info_from_mask(instance, label)
    h, w = instance.size(-2), instance.size(-1)
    obbox_instance_mask = np.zeros((h, w))
    obbox_label_mask = np.zeros((h, w))
    for i in res:
        obbox_instance_mask = cv2.fillPoly(obbox_instance_mask, [i['obbox']], i['instance_id'])
        obbox_label_mask = cv2.fillPoly(obbox_label_mask, [i['obbox']], i['cls'])
    obbox_instance_mask = torch.from_numpy(obbox_instance_mask).to(instance.device)
    obbox_label_mask = torch.from_numpy(obbox_label_mask).to(instance.device)
    return obbox_instance_mask, obbox_label_mask


def get_obbox_instance_label_mask_batch(instances: torch.Tensor, labels: torch.Tensor):
    batch_size = instances.size(0)
    instance_masks = []
    label_masks = []
    for i in range(batch_size):
        instance_mask, label_mask = get_obbox_instance_label_mask(instances[i], labels[i])
        instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        label_masks.append(label_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    instance_masks = torch.cat(instance_masks, dim=0)
    label_masks = torch.cat(label_masks, dim=0)
    return instance_masks, label_masks.float()


def get_bbox_instance_label_mask_refine(instance: torch.Tensor, label: torch.Tensor):
    """
    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return: tensor
    """
    info = get_bbox_info_from_mask(instance, label)
    h, w = instance.size(-2), instance.size(-1)
    bbox_instance_mask = torch.zeros(h, w).to(instance.device)
    # bbox_label_mask = torch.zeros(h, w).to(instance.device)
    for inst in info:
        u, l, b, r = inst['bbox']
        bbox_instance_mask[u:b + 1, l:r + 1][label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = inst[
            'instance_id']
        # bbox_label_mask[u:b + 1, l:r + 1][label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = inst['cls']
    return bbox_instance_mask, label


def get_bbox_instance_label_mask_refine_batch(instances: torch.Tensor, labels: torch.Tensor):
    """

    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return:
    """
    batch_size = instances.size(0)
    instance_masks = []
    for i in range(batch_size):
        instance_mask, label = get_bbox_instance_label_mask_refine(instances[i], labels[i])
        instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    instance_masks = torch.cat(instance_masks, dim=0)
    return instance_masks, labels.float()


class BboxSmaskClusterRefine:
    def __init__(self, args):
        xm = torch.linspace(0, 2, 2048).view(
            1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(
            1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)
        self.xym = xym
        self.n_sigma = args.n_sigma
        self.area_choose = args.area_choose
        self.to_center = 'to_center' in args.criterion
        self.mode = args.imask_refine
        print(f'===> BboxSmaskClusterRefine setting\n'
              f'mode:{self.mode} | n_sigma:{self.n_sigma} | area_choose:{self.area_choose} | to_center:{self.to_center}')

    def get_one_with_gt_mask(self, instance: torch.Tensor, label: torch.Tensor, prediction: torch.Tensor):
        """
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return: tensor
        """
        assert prediction.dim() == 3
        prediction = prediction.detach()

        info = get_bbox_info_from_mask(instance, label)
        h, w = instance.size(-2), instance.size(-1)
        bbox_instance_mask = torch.zeros(h, w).to(instance.device)
        # for anx
        bbox_label_mask = torch.zeros(h, w).to(instance.device)

        xym_s = self.xym[:, 0:h, 0:w].contiguous().to(instance.device)
        spatial_emb = prediction[0:2].tanh() + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        seedmap = prediction[2 + self.n_sigma:]  # cls(w/o bkgd) x h x w

        inst_dict = {inst['instance_id']: inst for inst in info}
        for inst in info:
            u, l, b, r = inst['bbox']
            bbox_instance_erea = bbox_instance_mask[u:b + 1, l:r + 1]
            bbox_label_area = bbox_label_mask[u:b + 1, l:r + 1]
            gtlabel_erea = label.squeeze(dim=0)[u:b + 1, l:r + 1]

            overlap_insts = bbox_instance_erea[bbox_label_area == inst['cls']].unique()
            overlap_insts = overlap_insts[overlap_insts > 0]
            if len(overlap_insts) > 0:
                if self.mode == 'normal':
                    inst_dist = self.get_dist_of_inst(inst, label, xym_s, spatial_emb, sigma)
                elif self.mode == 'exoverlap':
                    inst_dist = self.get_dist_of_inst_exoverlap(inst,
                                                                [inst_dict[int(overlap_inst_id)] for overlap_inst_id in
                                                                 overlap_insts], label, xym_s, spatial_emb, sigma)
                elif self.mode == 'seedmapmax':
                    inst_dist = self.get_dist_of_inst_seedmapmax(inst,
                                                                 [inst_dict[int(overlap_inst_id)] for overlap_inst_id in
                                                                  overlap_insts], label, xym_s,
                                                                 spatial_emb, sigma, seedmap)
                else:
                    raise NotImplementedError

                for overlap_inst_id in overlap_insts:
                    overlap_inst_id = int(overlap_inst_id)

                    overlap_mask = torch.zeros(h, w).to(instance.device)
                    overlap_mask_erea = overlap_mask[u:b + 1, l:r + 1]
                    overlap_mask_erea[bbox_instance_erea == overlap_inst_id] = inst['instance_id']

                    oinst = inst_dict[overlap_inst_id]
                    if self.mode == 'normal':
                        oinst_dist = self.get_dist_of_inst(oinst, label, xym_s, spatial_emb, sigma)
                    elif self.mode == 'exoverlap':
                        oinst_dist = self.get_dist_of_inst_exoverlap(oinst, [inst], label, xym_s, spatial_emb, sigma)
                    elif self.mode == 'seedmapmax':
                        oinst_dist = self.get_dist_of_inst_seedmapmax(oinst, [inst], label, xym_s,
                                                                      spatial_emb, sigma, seedmap)
                    else:
                        raise NotImplementedError

                    overlap_inst_dist = inst_dist[overlap_mask > 0].float()
                    # print('overlap_inst_dist', overlap_inst_dist.unique())

                    overlap_oinst_dist = oinst_dist[overlap_mask > 0].float()
                    # print('overlap_oinst_dist', overlap_oinst_dist.unique())
                    # print('inst vs oinst', (overlap_oinst_dist < overlap_inst_dist).unique())

                    # if inst_dist is larger, which means closer, update overlap eara to inst
                    overlap_mask[overlap_mask > 0] = torch.where(overlap_inst_dist >= overlap_oinst_dist,
                                                                 overlap_mask[overlap_mask > 0],
                                                                 bbox_instance_mask[overlap_mask > 0])
                    bbox_instance_mask[overlap_mask > 0] = overlap_mask[overlap_mask > 0]
                    torch.cuda.empty_cache()

            update_mask = torch.zeros(h, w).to(instance.device)
            update_mask[u:b + 1, l:r + 1][gtlabel_erea == inst['cls']] = 1
            update_mask[u:b + 1, l:r + 1][bbox_instance_erea > 0] = 0
            # bbox_instance_erea[bbox_instance_erea == 0][gtlabel_erea[bbox_instance_erea == 0] == inst['cls']] = inst[
            #     'instance_id']
            bbox_instance_mask[update_mask > 0] = inst['instance_id']
            # print(bbox_instance_erea.unique())
            # bbox_instance_erea[gtlabel_erea == inst['cls']] = inst['instance_id']
            bbox_label_area[gtlabel_erea == inst['cls']] = inst['cls']
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        return bbox_instance_mask

    def get_batch_with_gt_mask(self, instances: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        """
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return:
        """
        predictions = predictions.detach()

        batch_size = instances.size(0)
        instance_masks = []
        for i in range(batch_size):
            instance_mask = self.get_one_with_gt_mask(instances[i], labels[i], predictions[i])
            instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        instance_masks = torch.cat(instance_masks, dim=0)
        return instance_masks

    def get_dist_of_inst(self, inst, label, xym_s, spatial_emb, sigma):
        u, l, b, r = inst['bbox']
        h, w = label.size(-2), label.size(-1)
        in_mask = torch.zeros(h, w).to(label.device).byte()
        assert self.area_choose >= 0.
        if self.area_choose > 1.:
            in_mask[u:b + 1, l:r + 1][label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = 1
        else:
            h_shrink = (1. - (b - u) * math.sqrt(self.area_choose)) / 2.
            w_shrink = (1. - (r - l) * math.sqrt(self.area_choose)) / 2.
            u_s = int(u + h_shrink)
            b_s = int(b - h_shrink)
            l_s = int(l + w_shrink)
            r_s = int(r - w_shrink)
            in_mask[u_s:b_s + 1, l_s:r_s + 1][label.squeeze(dim=0)[u_s:b_s + 1, l_s:r_s + 1] == inst['cls']] = 1
            pass
        # print('in_mask', in_mask.unique())

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

        s = sigma_in.mean(1).view(self.n_sigma, 1, 1)  # n_sigma x 1 x 1

        s = torch.exp(s * 10)

        # calculate gaussian
        dist = torch.exp(-1 * torch.sum(
            torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))
        torch.cuda.empty_cache()
        return dist.squeeze()

    def get_dist_of_inst_exoverlap(self, inst, oinst_list, label, xym_s, spatial_emb, sigma):
        u, l, b, r = inst['bbox']
        h, w = label.size(-2), label.size(-1)
        in_mask = torch.zeros(h, w).to(label.device).byte()
        in_mask[u:b + 1, l:r + 1][label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = 1
        in_mask_clone = in_mask.clone()
        count1 = in_mask.sum()
        for oinst in oinst_list:
            u_o, l_o, b_o, r_o = oinst['bbox']
            in_mask[u_o:b_o + 1, l_o:r_o + 1] = 0
        count2 = in_mask.sum()
        fill_rate = float(count2) / float(count1)
        if fill_rate < 0.5:
            in_mask = in_mask_clone
        # print(f'{fill_rate}')

        # calculate center of attraction
        if self.to_center:
            xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
            center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1
        else:
            center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

        # if count2 == 0:
        #     print(f'in_mask:{in_mask}')
        #     print(f'spatial:{spatial_emb[in_mask.expand_as(spatial_emb)]}')
        #     print('center:', center)

        # calculate sigma
        sigma_in = sigma[in_mask.expand_as(
            sigma)].view(self.n_sigma, -1)

        s = sigma_in.mean(1).view(self.n_sigma, 1, 1)  # n_sigma x 1 x 1

        s = torch.exp(s * 10)

        # calculate gaussian
        dist = torch.exp(-1 * torch.sum(
            torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))
        torch.cuda.empty_cache()
        return dist.squeeze()

    def get_dist_of_inst_seedmapmax(self, inst, oinst_list, label, xym_s, spatial_emb, sigma, seedmap):
        u, l, b, r = inst['bbox']
        h, w = label.size(-2), label.size(-1)
        in_mask = torch.zeros(h, w).to(label.device).byte()
        assert self.area_choose > 0.
        if self.area_choose >= 1.:
            in_mask[u:b + 1, l:r + 1][label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = 1
        else:
            h_shrink = (1. - (b - u) * math.sqrt(self.area_choose)) / 2.
            w_shrink = (1. - (r - l) * math.sqrt(self.area_choose)) / 2.
            u_s = int(u + h_shrink)
            b_s = int(b - h_shrink)
            l_s = int(l + w_shrink)
            r_s = int(r - w_shrink)
            in_mask[u_s:b_s + 1, l_s:r_s + 1][label.squeeze(dim=0)[u_s:b_s + 1, l_s:r_s + 1] == inst['cls']] = 1
            pass
        in_mask_clone = in_mask.clone()
        count1 = in_mask.sum()
        for oinst in oinst_list:
            u_o, l_o, b_o, r_o = oinst['bbox']
            in_mask[u_o:b_o + 1, l_o:r_o + 1] = 0
        count2 = in_mask.sum()
        fill_rate = float(count2) / float(count1 + 1)
        if fill_rate < 0.5:
            in_mask = in_mask_clone
        # print(f'{fill_rate}')

        # calculate center of attraction
        seedmap_cls = seedmap[inst['cls'] - 1][in_mask].view(-1)
        seed_top_ids = seedmap_cls.argsort()
        seeds = seed_top_ids[-int(0.1 * len(seedmap_cls)):]

        center = spatial_emb[in_mask.expand_as(spatial_emb)].view(2, -1)[:, seeds].mean(1).view(2, 1, 1)  # 2 x 1 x 1
        s = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)[:, seeds].mean(1).view(self.n_sigma, 1,
                                                                                          1)  # n_sigma x 1 x 1
        s = torch.exp(s * 10)

        # calculate gaussian
        dist = torch.exp(-1 * torch.sum(
            torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))
        torch.cuda.empty_cache()
        return dist.squeeze()

    def get_one_with_pmask(self, instance: torch.Tensor, label: torch.Tensor,
                           p_label: torch.Tensor, prediction: torch.Tensor):
        """
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return: tensor
        """
        assert prediction.dim() == 3
        prediction = prediction.detach()

        info = get_bbox_info_from_mask(instance, label)
        label = None
        h, w = instance.size(-2), instance.size(-1)
        bbox_instance_mask = torch.zeros(h, w).to(instance.device)
        # for anx
        bbox_label_mask = torch.zeros(h, w).to(instance.device)

        xym_s = self.xym[:, 0:h, 0:w].contiguous().to(instance.device)
        spatial_emb = prediction[0:2].tanh() + xym_s  # 2 x h x w
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w

        inst_dict = {inst['instance_id']: inst for inst in info}
        for inst in info:
            u, l, b, r = inst['bbox']
            bbox_instance_erea = bbox_instance_mask[u:b + 1, l:r + 1]
            bbox_label_area = bbox_label_mask[u:b + 1, l:r + 1]
            plabel_erea = p_label.squeeze(dim=0)[u:b + 1, l:r + 1]

            overlap_insts = bbox_instance_erea[bbox_label_area == inst['cls']].unique()
            overlap_insts = overlap_insts[overlap_insts > 0]
            if len(overlap_insts) > 0:
                inst_dist = self.get_dist_of_inst(inst, p_label, xym_s, spatial_emb, sigma)
                for overlap_inst_id in overlap_insts:
                    overlap_inst_id = int(overlap_inst_id)

                    overlap_mask = torch.zeros(h, w).to(instance.device)
                    overlap_mask_erea = overlap_mask[u:b + 1, l:r + 1]
                    overlap_mask_erea[bbox_instance_erea == overlap_inst_id] = inst['instance_id']

                    oinst = inst_dict[overlap_inst_id]
                    oinst_dist = self.get_dist_of_inst(oinst, p_label, xym_s, spatial_emb, sigma)

                    overlap_inst_dist = inst_dist[overlap_mask > 0].float()
                    # print('overlap_inst_dist', overlap_inst_dist.unique())
                    overlap_oinst_dist = oinst_dist[overlap_mask > 0].float()
                    # print('overlap_oinst_dist', overlap_oinst_dist.unique())
                    # if inst_dist is larger, which means closer, update overlap eara to inst
                    overlap_mask[overlap_mask > 0] = torch.where(overlap_inst_dist >= overlap_oinst_dist,
                                                                 overlap_mask[overlap_mask > 0],
                                                                 bbox_instance_mask[overlap_mask > 0])
                    bbox_instance_mask[overlap_mask > 0] = overlap_mask[overlap_mask > 0]
                    torch.cuda.empty_cache()

            update_mask = torch.zeros(h, w).to(instance.device)
            update_mask[u:b + 1, l:r + 1][plabel_erea == inst['cls']] = 1
            update_mask[u:b + 1, l:r + 1][bbox_instance_erea > 0] = 0
            # bbox_instance_erea[bbox_instance_erea == 0][gtlabel_erea[bbox_instance_erea == 0] == inst['cls']] = inst[
            #     'instance_id']
            bbox_instance_mask[update_mask > 0] = inst['instance_id']
            # print(bbox_instance_erea.unique())
            # bbox_instance_erea[gtlabel_erea == inst['cls']] = inst['instance_id']
            bbox_label_area[plabel_erea == inst['cls']] = inst['cls']
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        return bbox_instance_mask

    def get_batch_with_pmask(self, instances: torch.Tensor, labels: torch.Tensor,
                             p_labels: torch.Tensor, predictions: torch.Tensor):
        """
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return:
        """
        batch_size = instances.size(0)
        predictions = predictions.detach()
        instance_masks = []
        for i in range(batch_size):
            instance_mask = self.get_one_with_pmask(instances[i], labels[i], p_labels[i], predictions[i])
            instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        instance_masks = torch.cat(instance_masks, dim=0)
        return instance_masks


def get_bbox_instance_label_mask_prefine(instance: torch.Tensor, label: torch.Tensor, p_label: torch.Tensor):
    """
    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return: tensor
    """
    info = get_bbox_info_from_mask(instance, label)
    h, w = instance.size(-2), instance.size(-1)
    bbox_instance_mask = torch.zeros(h, w).to(instance.device)
    # bbox_plabel_mask = torch.zeros(h, w).to(instance.device)
    for inst in info:
        u, l, b, r = inst['bbox']
        bbox_instance_mask[u:b + 1, l:r + 1][p_label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = inst[
            'instance_id']
        # bbox_plabel_mask[u:b + 1, l:r + 1][p_label.squeeze(dim=0)[u:b + 1, l:r + 1] == inst['cls']] = inst['cls']
    return bbox_instance_mask, label


def get_bbox_instance_label_mask_prefine_batch(instances: torch.Tensor, labels: torch.Tensor, p_labels: torch.Tensor):
    """

    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return:
    """
    batch_size = instances.size(0)
    instance_masks = []
    for i in range(batch_size):
        instance_mask, label = get_bbox_instance_label_mask_prefine(instances[i], labels[i], p_labels[i])
        instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    instance_masks = torch.cat(instance_masks, dim=0)
    return instance_masks, p_labels.float()


def get_instance_label_mask_overlap255_batch(instances: torch.Tensor, labels: torch.Tensor,
                                             p_labels: torch.Tensor = None, inbox255=True):
    def get_instance_label_mask_overlap255(instance: torch.Tensor, label: torch.Tensor, p_label: torch.Tensor = None,
                                           inbox255=True):
        """
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return: tensor
        """
        info = get_bbox_info_from_mask(instance, label)
        h, w = instance.size(-2), instance.size(-1)
        bbox_instance_mask = torch.zeros(h, w).to(instance.device)
        # pseudo mean use predicted label
        if p_label is None:
            p_label = label.squeeze()
            pseudo = False
        else:
            p_label = p_label.squeeze()
            pseudo = True
        for inst in info:
            u, l, b, r = inst['bbox']
            instance_area = bbox_instance_mask[u:b + 1, l:r + 1]
            p_label_area = p_label[u:b + 1, l:r + 1]
            p_label_area[(p_label_area == inst['cls']) & (instance_area > 0)] = 255
            # p_label[u:b + 1, l:r + 1] = p_label_area
            if pseudo and inbox255:
                p_label_area[p_label_area == 0] = 255
            instance_area[p_label_area == inst['cls']] = inst['instance_id']
        bbox_instance_mask[p_label == 255] = 0
        return bbox_instance_mask, p_label

    """
    :param instances: gt
    :param labels: gt
    :param p_labels: predictions
    :return:
    """
    batch_size = instances.size(0)
    instance_masks = []
    label_masks = []
    for i in range(batch_size):
        if p_labels is None:
            instance_mask, label_mask = get_instance_label_mask_overlap255(instances[i], labels[i])
        else:
            instance_mask, label_mask = get_instance_label_mask_overlap255(instances[i], labels[i], p_labels[i],
                                                                           inbox255)
        instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        label_masks.append(label_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    instance_masks = torch.cat(instance_masks, dim=0)
    label_masks = torch.cat(label_masks, dim=0)
    return instance_masks, label_masks.float()


def get_instance_label_mask_obbx_overlap255_batch(instances: torch.Tensor, labels: torch.Tensor,
                                                  p_labels: torch.Tensor = None, inbox255=True):
    def get_instance_label_mask_obbx_overlap255(instance: torch.Tensor, label: torch.Tensor,
                                                p_label: torch.Tensor = None, inbox255=True):
        """
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return: tensor
        """
        info = get_obbox_info_from_mask(instance, label)
        h, w = instance.size(-2), instance.size(-1)
        bbox_instance_mask = torch.zeros(h, w).to(instance.device)
        # pseudo mean use predicted label
        if p_label is None:
            p_label = label.squeeze()
            pseudo = False
        else:
            p_label = p_label.squeeze()
            pseudo = True

        for inst in info:
            obbox_area_mask = np.zeros((h, w))
            obbox_area_mask = cv2.fillPoly(obbox_area_mask, [inst['obbox']], 1)
            obbox_area_mask = torch.from_numpy(obbox_area_mask).to(instance.device)
            instance_area = bbox_instance_mask[obbox_area_mask > 0]
            p_label_area = p_label[obbox_area_mask > 0]
            print(instance_area.sum(), p_label_area.sum())
            p_label_area[(p_label_area == inst['cls']) & (instance_area > 0)] = 255
            # p_label[u:b + 1, l:r + 1] = p_label_area
            if pseudo and inbox255:
                p_label_area[p_label_area == 0] = 255
            instance_area[p_label_area == inst['cls']] = inst['instance_id']
        bbox_instance_mask[p_label == 255] = 0
        return bbox_instance_mask, p_label

    """
    :param instances: gt
    :param labels: gt
    :param p_labels: predictions
    :return:
    """
    batch_size = instances.size(0)
    instance_masks = []
    label_masks = []
    for i in range(batch_size):
        if p_labels is None:
            instance_mask, label_mask = get_instance_label_mask_obbx_overlap255(instances[i], labels[i])
        else:
            instance_mask, label_mask = get_instance_label_mask_obbx_overlap255(instances[i], labels[i], p_labels[i],
                                                                                inbox255)
        instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        label_masks.append(label_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    instance_masks = torch.cat(instance_masks, dim=0)
    label_masks = torch.cat(label_masks, dim=0)
    return instance_masks, label_masks.float()


def get_instance_label_with_abbox_overlap255_batch(instances: torch.Tensor, labels: torch.Tensor,
                                                   p_labels: torch.Tensor = None):
    def get_instance_label_with_abbox_overlap255(instance: torch.Tensor, label: torch.Tensor,
                                                 p_label: torch.Tensor = None):
        """
        agnostic bbox
        :param instance: single instance mask, can be dim=2 or 3
        :param label: single label mask, can be dim=2 or 3
        :return: tensor
        """
        info = get_bbox_info_from_mask(instance, label)
        h, w = instance.size(-2), instance.size(-1)
        bbox_instance_mask = torch.zeros(h, w).to(instance.device)
        # pseudo mean use predicted label
        if p_label is None:
            p_label = label.squeeze()
            pseudo = False
        else:
            p_label = p_label.squeeze()
            pseudo = True
        for inst in info:
            inst['cls'] = None  # agnostic bbox
            u, l, b, r = inst['bbox']
            instance_area = bbox_instance_mask[u:b + 1, l:r + 1]
            p_label_area = p_label[u:b + 1, l:r + 1]
            p_label_area[instance_area > 0] = 255
            if pseudo:
                p_label_area[p_label_area == 0] = 255
            instance_area[p_label_area > 0] = inst['instance_id']
        bbox_instance_mask[p_label == 255] = 0
        return bbox_instance_mask, p_label

    """
    :param instances: gt
    :param labels: gt
    :param p_labels: predictions
    :return:
    """
    batch_size = instances.size(0)
    instance_masks = []
    label_masks = []
    for i in range(batch_size):
        if p_labels is None:
            instance_mask, label_mask = get_instance_label_with_abbox_overlap255(instances[i], labels[i])
        else:
            instance_mask, label_mask = get_instance_label_with_abbox_overlap255(instances[i], labels[i], p_labels[i])
        instance_masks.append(instance_mask.unsqueeze(dim=0).unsqueeze(dim=0))
        label_masks.append(label_mask.unsqueeze(dim=0).unsqueeze(dim=0))
    instance_masks = torch.cat(instance_masks, dim=0)
    label_masks = torch.cat(label_masks, dim=0)
    return instance_masks, label_masks.float()


def get_FCOS_like_center_map(instance: torch.Tensor, label: torch.Tensor):
    """

    :param instance:
    :param label:
    :return:
    """
    info = get_bbox_info_from_mask(instance, label)
    h, w = instance.size(-2), instance.size(-1)
    center_mask = torch.zeros(h, w).to(instance.device)
    for inst in info:
        u, l, b, r = inst['bbox']
        temp_ub = torch.zeros(2, b - u + 1, r - l + 1).to(instance.device)
        temp_lr = torch.zeros(2, b - u + 1, r - l + 1).to(instance.device)
        u2b = range(b - u + 1)
        b2u = u2b[::-1]
        l2r = range(r - l + 1)
        r2l = l2r[::-1]
        temp_ub[0] = torch.tensor(u2b).unsqueeze(dim=1).expand_as(temp_ub[0])
        temp_ub[1] = torch.tensor(b2u).unsqueeze(dim=1).expand_as(temp_ub[1])
        temp_lr[0] = torch.tensor(l2r).unsqueeze(dim=0).expand_as(temp_lr[0])
        temp_lr[1] = torch.tensor(r2l).unsqueeze(dim=0).expand_as(temp_lr[1])

        center_temp = torch.sqrt((temp_lr.min(dim=0)[0] / temp_lr.max(dim=0)[0])
                                 * (temp_ub.min(dim=0)[0] / temp_ub.max(dim=0)[0]))

        center_mask[u:b + 1, l:r + 1] = center_temp.unsqueeze(dim=0)
    return center_mask


def get_FCOS_like_center_map_batch(instances: torch.Tensor, labels: torch.Tensor):
    """

    :param instance: single instance mask, can be dim=2 or 3
    :param label: single label mask, can be dim=2 or 3
    :return:
    """
    batch_size = instances.size(0)
    center_maps = []
    for i in range(batch_size):
        center_map = get_FCOS_like_center_map(instances[i], labels[i])
        center_maps.append(center_map.unsqueeze(dim=0).unsqueeze(dim=0))
    center_maps = torch.cat(center_maps, dim=0)
    return center_maps


class ClusterForWeakly:
    def __init__(self, bkgd=False, n_sigma=2, threshold=0.5, refine=True, seed_mode='max'):
        print(f'===> Clustering config: bkgd:{bkgd} | n_sigma:{n_sigma} | threshold:{threshold}')
        xm = torch.linspace(0, 2, 2048).view(1, 1, -1).expand(1, 1024, 2048)
        ym = torch.linspace(0, 1, 1024).view(1, -1, 1).expand(1, 1024, 2048)
        xym = torch.cat((xm, ym), 0)
        self.bkgd = bkgd
        self.xym = xym
        self.n_sigma = n_sigma
        self.threshold = threshold
        self.refine = refine
        self.seed_mode = seed_mode

    def cluster_with_gtbbox(self, prediction, bbox_instance):
        """

        :param prediction: [xy[2], sigma[2], classs[8]]
        :param bbox_instance: [{'mask':[h,w], 'cls':int}]
        :return:{'mask':np, 'cls':int}
        """
        assert prediction.dim() == 3

        height, width = prediction.size(1), prediction.size(2)
        bbox_instance.sort(key=lambda x: x['mask'].sum(), reverse=True)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
        if prediction.is_cuda:
            xym_s = xym_s.cuda()

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        # print(spatial_emb[spatial_emb < 0])
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        seed_map_all = torch.sigmoid(prediction[2 + self.n_sigma:])  # cls x h x w

        # instance counter
        instances = []
        instance_cls = {}
        instance_map = torch.zeros(height, width).byte().to(prediction.device)

        # instance counter
        count = 1
        for bbox_inst in bbox_instance:
            cls = bbox_inst['cls'] - 1
            bbox_mask = bbox_inst['mask']
            seed_map = seed_map_all[cls] * bbox_mask
            mask = seed_map > 0.5
            if self.refine:
                if mask.sum() > 128:
                    spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)  # 2xn
                    sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)  # n_simgaxn
                    seed_map_masked = seed_map[mask].view(1, -1)  # 1xn
                    instance_map_masked = torch.zeros(mask.sum()).byte().to(seed_map.device)  # 1xn

                    # find the max score seed and its location

                    # 1.
                    seed_score = seed_map_masked.max().item()
                    if self.seed_mode == 'max':
                        seed = seed_map_masked.argmax().item()
                        # x, y of center as shape: [2,1]
                        center = spatial_emb_masked[:, seed:seed + 1]

                    # 2.
                    elif self.seed_mode == 'mean':
                        center = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
                        center = center.mean(dim=1).view(2, 1)  # 2 x 1 x 1
                    else:
                        raise NotImplementedError

                    s = torch.exp(sigma_masked[:, seed:seed + 1] * 10)
                    dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb_masked -
                                                              center, 2) * s, dim=0, keepdim=True))

                    # proposal is a group of pixels which near to seed which regarded as center
                    proposal = (dist > 0.5).squeeze()

                    if proposal.sum() > 128:
                        # mark pixels belong to proposal as 'count'
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu()
                        instances.append(
                            {'mask': instance_mask.squeeze().cpu().numpy(), 'score': seed_score, 'cls': class_ids[cls]})
                        instance_cls[count] = class_ids[cls]
                        count += 1

                        instance_map[mask] = instance_map_masked
            else:
                instance_map[mask] = count
                count += 1
                instance_mask = torch.zeros(height, width)
                instance_mask[mask] = 1
                # seed_score = seed_map[mask].view(1, -1).max().item()
                instances.append(
                    {'mask': instance_mask.squeeze().cpu().numpy(), 'cls': class_ids[cls]})
                instance_cls[count] = class_ids[cls]
        return instance_map, instances, instance_cls


def sbbox_to_abbox(sbbox: torch.Tensor):
    batch_size = sbbox.size(0)
    for b in range(batch_size):
        bkgd = sbbox[b, 0]
        fgd = sbbox[b, 1:]
        fgd[bkgd.expand_as(fgd) == 0] = 1
    return sbbox


def get_grabcut_mask(sample):
    import cv2
    img = sample['image']
    mask = np.zeros(img.shape[-2:], dtype=np.uint8)
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    pass


def test():
    from train_config import get_config
    from tools.utils_for_weakly import get_bbox_instance
    from datasets import get_cityscapes_dataloader, get_voc_instance_dataloader
    from torchvision import transforms
    args = get_config()
    args.aug += 'std_bbox'
    data_loader = get_cityscapes_dataloader(args, train=False)
    # data_loader = get_voc_instance_dataloader(args, train=False)
    for i, sample in enumerate(data_loader):
        image = sample['image']
        instance = sample['instance']
        label = sample['label']
        bbox = sample['bbox']
        # print(sample['bbox_info'])
        print(sample['im_name'])

        b, h, w = image.size(0), image.size(-2), image.size(-1)
        bbox_dict = get_bbox_info_from_mask(instance[0], label[0])
        for i, inst in enumerate(bbox_dict):
            bbox_mask = torch.zeros(h, w).to(instance.device)
            instance_mask = torch.zeros(h, w).to(instance.device)
            u, l, b, r = inst['bbox']
            bbox_mask[u:b + 1, l:r + 1] = 1
            instance_mask[instance[0, 0] == inst['instance_id']] = 1
            # transforms.ToPILImage()(bbox_mask).save(f'./bbox_mask_{i}.png')
            # transforms.ToPILImage()(instance_mask).save(f'./instance_mask_{i}.png')
        bbox_instance, bbox_label = get_obbox_instance_label_mask(instance[0], label[0])
        transforms.ToPILImage()(bbox_instance.float() * 25).show()  # .save('./instance_mask_all.png')
        transforms.ToPILImage()(bbox_label.float() * 25).show()
        transforms.ToPILImage()(image[0].float() * 25).show()

        # test bbox dict
        # bbox_instance = get_bbox_instance_np(instance[0], label[0])
        # bboxes = torch.zeros(1, image.size(-2), image.size(-1))
        # for i in bbox_instance:
        #     box_temp = torch.from_numpy(i['mask']).unsqueeze(dim=0)
        #     bboxes[box_temp > 0] = i['cls']

        # test bbox instance
        # b, h, w = image.size(0), image.size(-2), image.size(-1)
        # predictions = torch.rand(size=(b, 4, h, w))
        # predictions[:, 0] = instance[0]
        # bbox_instance_masks, bbox_label_masks = get_bbox_instance_label_mask_batch(instance, label)
        # instance_masks_255, label_masks_255 = get_instance_label_mask_overlap255_batch(instance, label)
        # instance_masks_255, label_masks_255 = get_instance_label_with_abbox_overlap255_batch(instance, label, None)
        # bbox_instance_masks, bbox_label_masks = get_bbox_instance_label_mask_refine_batch(instance, label)

        # transforms.ToPILImage()(image[0]).show()
        # transforms.ToPILImage()(bboxes).show()

        # transforms.ToPILImage()(instance[0]*10).show()
        # transforms.ToPILImage()(label[0]*50).show()

        # test bbox mask
        # transforms.ToPILImage()(bbox_instance_masks[0] * 10).show()
        # transforms.ToPILImage()(bbox_label_masks[0] * 50).show()

        # transforms.ToPILImage()(instance_masks_255[0] * 20).show()
        # transforms.ToPILImage()((label_masks_255[0] == 255) * 255).show()
        # print('label', label_masks_255.unique())

        # abbox = sbbox_to_abbox(bbox)
        # c = abbox.size(1)
        # for i in range(c):
        # transforms.ToPILImage()((abbox[0, i] == 1) * 255).show()
        # pass

        # to see the overlap with GT
        # transforms.ToPILImage()((bbox_instance_masks[0] == instance[0].float()) * 255).show()
        # transforms.ToPILImage()((bbox_label_masks[0] == label[0].float()) * 255).show()

        #
        # args.criterion = 'bce_to_center'
        # ClusterRefiner = BboxSmaskClusterRefine(args)
        # cluster_instances = ClusterRefiner.get_batch(instance, label, predictions)
        # transforms.ToPILImage()(cluster_instances[0] * 10).show()

        # test center map
        # center_maps = get_FCOS_like_center_map_batch(instance, label)
        # transforms.ToPILImage()(center_maps[0]).show()

        # transforms.ToPILImage()(bbox_instance[0]*20).show()
        # bbox_temp = bbox[0].argmax(dim=0).float()
        # transforms.ToPILImage()(bbox_temp*50).show()

        # print('inst', instance.unique())
        # print('label', label.unique())
        # print(bbox_temp.unique())

        print('Next')
        # print(sample)
        pass


def cluster_refine_test():
    from train_config import get_config
    from models import get_model, get_weakly_model
    from tools.utils_for_weakly import get_bbox_instance
    from datasets import get_cityscapes_dataloader
    from torchvision import transforms
    args = get_config()
    state_dict = torch.load('/Users/chenlinwei/code/SpatialEmbeddings/weaklyerfnet/checkpoint.pth', map_location='cpu')
    model = get_weakly_model(args, name='erfnet3branch')
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    args.crop_size = 400
    args.batch_size = 1
    args.aug = 'object_center_bbox'
    args.area_choose = 0.1
    ClusterRefiner_1 = BboxSmaskClusterRefine(args)
    args.area_choose = 0.5
    ClusterRefiner_5 = BboxSmaskClusterRefine(args)
    args.area_choose = 1.0
    ClusterRefiner_10 = BboxSmaskClusterRefine(args)
    args.imask_refine = 'exoverlap'
    ClusterRefiner_exoverlap = BboxSmaskClusterRefine(args)
    args.imask_refine = 'seedmapmax'
    args.area_choose = 0.2
    ClusterRefiner_seedmapmax = BboxSmaskClusterRefine(args)
    data_loader = get_cityscapes_dataloader(args, train=True)
    for i, sample in enumerate(data_loader):
        image = sample['image']
        instance = sample['instance']
        label = sample['label']
        bbox = sample['bbox']

        bbox_instance_masks, bbox_label_masks = get_bbox_instance_label_mask_refine_batch(instance, label)
        predictions = model(image, bbox)
        # args.criterion = 'bce'
        cluster_instances_1 = ClusterRefiner_1.get_batch_with_gt_mask(instance, label, predictions)
        cluster_instances_5 = ClusterRefiner_5.get_batch_with_gt_mask(instance, label, predictions)
        cluster_instances_10 = ClusterRefiner_10.get_batch_with_gt_mask(instance, label, predictions)
        cluster_instances_exoverlap = ClusterRefiner_exoverlap.get_batch_with_gt_mask(instance, label, predictions)
        cluster_instances_seedmapmax = ClusterRefiner_seedmapmax.get_batch_with_gt_mask(instance, label, predictions)
        # cluster_instances = ClusterRefiner.get_batch_with_pmask(instance, label, label, predictions)

        # transforms.ToPILImage()(cluster_instances_1[0] * 10).show()
        # transforms.ToPILImage()(cluster_instances_5[0] * 10).show()
        # transforms.ToPILImage()(cluster_instances_10[0] * 10).show()
        # transforms.ToPILImage()(cluster_instances_exoverlap[0] * 10).show()
        # transforms.ToPILImage()(bbox_instance_masks[0] * 10).show()
        # transforms.ToPILImage()(image[0]).show()
        # transforms.ToPILImage()(instance[0]*10).show()
        # transforms.ToPILImage()((instance[0].float() == cluster_instances_1[0].float()).float()).show()
        # transforms.ToPILImage()((instance[0].float() == cluster_instances_exoverlap[0].float()).float()).show()

        # right pixel
        print('0.1', (instance[0].float() == cluster_instances_1[0].float()).sum())
        print('0.5', (instance[0].float() == cluster_instances_5[0].float()).sum())
        print('1.0', (instance[0].float() == cluster_instances_10[0].float()).sum())
        print('exo', (instance[0].float() == cluster_instances_exoverlap[0].float()).sum())
        print('seedmapmax', (instance[0].float() == cluster_instances_seedmapmax[0].float()).sum())
        print('bbox', (instance[0].float() == bbox_instance_masks[0].float()).sum())
        pass


if __name__ == '__main__':
    pass
    # cluster_refine_test()
    test()
