import torch
from datasets.CityscapesDataset import class_ids_to_index
import numpy as np


def instances_refine(instances, bbox):
    """
    refine the instance mask using bbox mask

    instances.append({'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': class_ids[cls]})

    mode
    score of res will be > weakly threshold

    :param instances:
    :return:
    """
    assert bbox.dim() == 3
    instance_map = torch.zeros(bbox.shape[1:]).to(bbox.device)
    res = []

    for i, instance in enumerate(instances):
        mask = instance['mask'].float().to(bbox.device)  # hxw
        score = instance['score']  # float
        cls = class_ids_to_index[instance['cls']]  # float
        mask = mask * bbox[cls]
        instance_map[mask > 0] = i + 1
        if (mask > 0).sum() > 128:
            res.append({'mask': mask, 'score': score, 'cls_index': cls, 'cls': instance['cls']})

    return res, instance_map.unsqueeze(dim=0)


def get_bbox_instance(instance: torch.Tensor, label: torch.Tensor):
    """
    get instance bbox mask from instance mask and label
    :param instance:  a mask of instance
    :param label: a mask of label
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

    print('debug', {int(i): label[instance == i].unique() for i in instance_ids})
    instance_to_cls = {int(i): int(label[instance == i].unique()) for i in instance_ids}
    h, w = instance.shape
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

    # instances_dict
    bbox_instances = []
    for inst in res:
        instance_mask = torch.zeros(w, h).to(instance.device)
        u, l, b, r = inst['bbox']
        instance_mask[u:b + 1, l:r + 1] = 1
        temp = {'mask': instance_mask, 'cls': inst['cls']}
        bbox_instances.append(temp)
    # print(bbox_instances)
    return bbox_instances


def from_seg_pred_to_one_hot(seg_pred):
    seg_max, score = seg_pred.max(dim=1, keepdim=True)
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
    targets = torch.where(targets.float() < n_class, targets.float(), zeros.float())

    one_hot = torch.nn.functional.one_hot(targets.long(), num_classes=n_class)
    one_hot[targets_ignore] = 0
    # print(one_hot[targets_ignore])
    one_hot = one_hot.transpose(3, 2)
    one_hot = one_hot.transpose(2, 1)
    # print(one_hot.size())
    return one_hot.float()


from tools.utils import class_ids


class ClusterForWeakly:
    """
    deprecated， not useful
    """
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
            mask = seed_map >= 0.5
            if self.refine:
                if mask.sum() > 128:
                    spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)  # 2xn
                    sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)  # n_simgaxn
                    peek_map_masked = seed_map[mask].view(1, -1)  # 1xn
                    instance_map_masked = torch.zeros(mask.sum()).byte().to(seed_map.device)  # 1xn

                    # find the max score seed and its location

                    # 1.
                    seed_score = peek_map_masked.max().item()
                    if self.seed_mode == 'max':
                        seed = peek_map_masked.argmax().item()
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

                    # bbox -> mask (>thres) -> proposal (dist>0.5)
                    # proposal is a group of pixels which near to seed which regarded as center
                    proposal = (dist >= 0.5).squeeze()

                    if proposal.sum() > 128:
                        # mark pixels belong to proposal as 'count'
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).byte()
                        instance_mask[mask.squeeze().cpu()] = proposal.cpu().byte()
                        instances.append({
                            'mask': instance_mask.squeeze().cpu().numpy(),
                            'score': seed_score,
                            'cls': class_ids[cls],
                            'area': (instance_mask > 0).sum()
                        })
                        instance_cls[count] = class_ids[cls]
                        count += 1

                        # instance_map[mask][instance_map_masked > 0] = instance_map_masked[instance_map_masked > 0]
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

    def cluster_with_seg_for_each(self, prediction):
        """

        :param prediction: [xy[2], sigma[2], peek_map, classs[8]]
        :param bbox_instance: [{'mask':[h,w], 'cls':int}]
        :return:{'mask':np, 'cls':int}
        """
        assert prediction.dim() == 3

        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w
        if prediction.is_cuda:
            xym_s = xym_s.cuda()

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        # print(spatial_emb[spatial_emb < 0])
        sigma = prediction[2:2 + self.n_sigma]  # n_sigma x h x w
        peek_map = prediction[2 + self.n_sigma].unsqeeze(dim=0).sigmoid()

        seed_map_all = torch.sigmoid(prediction[2 + self.n_sigma + 1:])  # cls(with background) x h x w
        # seed_map_all = from_seg_pred_to_one_hot(seed_map_all)  # one-hot

        n_class, height, width = seed_map_all.shape
        assert n_class == 8 or n_class == 9
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
            seed_map = seed_map_all[cls]
            mask = seed_map >= 0.5
            if mask.sum() > 128:
                spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(2, -1)
                sigma_masked = sigma[mask.expand_as(sigma)].view(self.n_sigma, -1)
                peek_map_masked = peek_map[mask].view(1, -1)

                # if 1, mark as unclustered
                unclustered = torch.ones(mask.sum()).byte().to(seed_map.device)
                # if zero, mark as unclustered
                instance_map_masked = torch.zeros(mask.sum()).byte().to(seed_map.device)

                while (unclustered.sum() > 128):
                    # find the max score seed and its location
                    seed = (peek_map_masked * unclustered.float()).argmax().item()
                    seed_score = (peek_map_masked * unclustered.float()).max().item()
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
                            instances.append(
                                {'mask': instance_mask.squeeze() * 255, 'score': seed_score, 'cls': class_ids[cls]})
                            instance_cls[count] = class_ids[cls]
                            count += 1
                    unclustered[proposal] = 0

                instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()
        return instance_map, instances, instance_cls
