"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import glob
import os
import random

import numpy as np
import pandas as pd
from PIL import Image
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import DataLoader
from datasets.transforms import get_transform
from torchvision import transforms
import torch
import math
from torchvision.transforms import functional as F
from collections import namedtuple
import json
from datasets.transforms import GetBBoxOneHotTensor, GetOrientedBBoxOneHotTensor

# from datasets.transforms import StrideAlign, RandomHorizontalFlip

class_ids_to_index = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15
}

gray_color_map = [0, 7, 14, 21, 29, 36, 44, 51, 58, 66, 76, 81, 89, 96, 103, 126]


def get_list_by_original_image(root_dir, type, choose_size, split=(0,), bias_to_minor=False):
    """
    ===> label:1 | size:3990 | fake_size:200
    ===> label:2 | size:1217 | fake_size:200
    ===> label:3 | size:597 | fake_size:200
    ===> label:4 | size:1125 | fake_size:200
    ===> label:5 | size:480 | fake_size:200
    ===> label:6 | size:1011 | fake_size:200
    ===> label:7 | size:2288 | fake_size:200
    ===> label:8 | size:6518 | fake_size:200
    ===> label:9 | size:10501 | fake_size:200
    ===> label:10 | size:194 | fake_size:200
    ===> label:11 | size:1327 | fake_size:200
    ===> label:12 | size:775 | fake_size:200
    ===> label:13 | size:1053 | fake_size:200
    ===> label:14 | size:3318 | fake_size:200
    ===> label:15 | size:2922 | fake_size:200
    :param root_dir:
    :param type:
    :param choose_size:
    :param bias_to_minor:
    :return: sample_list: json_file['images'], sample_ann_list: [[all anns for image], ...]
    """
    json_path = os.path.join(root_dir, f'iSAID_patches/{type}/instancesonly_filtered_{type}.json')
    json_file = json.load(open(json_path, 'r'))
    sample_list = json_file['images']
    ann_list = json_file['annotations']
    size = len(sample_list)
    cls_dict = {}
    with open(os.path.join(root_dir, f'iSAID_patches/{type}/classes_info.txt'), 'r') as f:
        f = f.readlines()
        f = [s.split() for s in f]
        for item in f:
            cls_dict[int(item[0])] = [int(i) for i in item[1:]]
    if type == 'train' and choose_size is not None:
        if choose_size < 0:
            choose_rest = True
            choose_size = abs(choose_size)
        else:
            choose_rest = False
        mask = []
        for i, sample in enumerate(sample_list):
            original_image_last_number = int(sample['file_name'][1:5]) % 10
            if original_image_last_number in split:
                # to include empty image
                # mask.append(i)
                if sample['id'] not in cls_dict:
                    continue
                mask.append(i)
        if choose_rest:
            temp = []
            mask = set(mask)
            for i in range(size):
                if i not in mask:
                    temp.append(i)
            mask = temp
    else:
        mask = list(range(size))

    mask.sort()
    sample_list = [sample_list[i] for i in mask]
    print(f'*** Before filtering empty image:{len(sample_list)}')
    if type == 'train':
        sample_list = [s for s in sample_list if s['id'] in cls_dict]
        print(f'*** After filtering empty image:{len(sample_list)}')
    sample_id_list = [s['id'] for s in sample_list]
    sample_ann_list = {s_id: [] for s_id in sample_id_list}
    for a in ann_list:
        image_id = a['image_id']
        if image_id in sample_ann_list:
            sample_ann_list[image_id].append(a)
    sample_ann_list = [sample_ann_list[s_id] for s_id in sample_id_list]
    for s, sa in zip(sample_list, sample_ann_list):
        s['annotations'] = sa
    """
    json_file['images']['annotations'] = [json_file['annotations']]
    """
    return sample_list


def get_list_by_cropped_images(root_dir, type, choose_size, bias_to_minor=True):
    """
    ===> label:1 | size:3990 | fake_size:200
    ===> label:2 | size:1217 | fake_size:200
    ===> label:3 | size:597 | fake_size:200
    ===> label:4 | size:1125 | fake_size:200
    ===> label:5 | size:480 | fake_size:200
    ===> label:6 | size:1011 | fake_size:200
    ===> label:7 | size:2288 | fake_size:200
    ===> label:8 | size:6518 | fake_size:200
    ===> label:9 | size:10501 | fake_size:200
    ===> label:10 | size:194 | fake_size:200
    ===> label:11 | size:1327 | fake_size:200
    ===> label:12 | size:775 | fake_size:200
    ===> label:13 | size:1053 | fake_size:200
    ===> label:14 | size:3318 | fake_size:200
    ===> label:15 | size:2922 | fake_size:200
    :param root_dir:
    :param type:
    :param choose_size:
    :param bias_to_minor:
    :return: sample_list: json_file['images'], sample_ann_list: [[all anns for image], ...]
    """
    # image_list = glob.glob(os.path.join(root_dir, 'iSAID_patches/{}/images'.format(type), 'P*[0-9].png'))
    # label_list = glob.glob(os.path.join(root_dir, 'iSAID_patches/{}/images'.format(type), 'P*color_RGB.png'))
    # instance_list = glob.glob(os.path.join(root_dir, 'iSAID_patches/{}/images'.format(type), 'P*id_RGB.png'))
    # image_list.sort()
    # label_list.sort()
    # instance_list.sort()

    json_path = os.path.join(root_dir, f'iSAID_patches/{type}/instancesonly_filtered_{type}.json')
    json_file = json.load(open(json_path, 'r'))
    sample_list = json_file['images']
    ann_list = json_file['annotations']
    size = len(sample_list)
    mask = list(range(size))
    if type == 'train' and choose_size is not None:
        if choose_size < 0:
            choose_rest = True
            choose_size = abs(choose_size)
        else:
            choose_rest = False

        if bias_to_minor:
            cls_dict = {}
            with open(os.path.join(root_dir, f'iSAID_patches/{type}/classes_info.txt'), 'r') as f:
                f = f.readlines()
                f = [s.split() for s in f]
                for item in f:
                    cls_dict[int(item[0])] = [int(i) for i in item[1:]]

            mask = []
            for c in range(4):
                cls = (10, 5, 3, 12)[c]
                cls_max = (194, 480, 597, 775)
                for i in range(size):
                    if sample_list[i]['id'] not in cls_dict:
                        continue

                    if len(mask) >= (choose_size * 0.25 * (c + 1)):
                        break
                    # idx = choose_size * i // size
                    idx = i
                    if idx not in mask and any([i in cls_dict[sample_list[idx]['id']] for i in (cls,)]):
                        mask.append(idx)

            print('*** mask size:', len(mask))
            interval = size // choose_size
            fill = [i for i in range(size) if i not in mask]
            mask += fill[0::interval][:choose_size - len(mask)]
            mask.sort()
            print('*** mask size after fill:', len(set(mask)))
            if choose_rest:
                temp = []
                for i in range(size):
                    if i not in mask:
                        temp.append(i)
                mask = temp
        else:
            interval = size // choose_size
            mask = list(range(size))[0::interval][:choose_size]
            if choose_rest:
                temp = []
                for i in range(size):
                    if i not in mask:
                        temp.append(i)
                mask = temp
    mask.sort()
    sample_list = [sample_list[i] for i in mask]
    sample_id_list = [s['id'] for s in sample_list]
    sample_ann_list = {s_id: [] for s_id in sample_id_list}
    for a in ann_list:
        image_id = a['image_id']
        if image_id in sample_ann_list:
            sample_ann_list[image_id].append(a)
    sample_ann_list = [sample_ann_list[s_id] for s_id in sample_id_list]
    for s, sa in zip(sample_list, sample_ann_list):
        s['annotations'] = sa
    """
    json_file['images']['annotations'] = [json_file['annotations']]
    """
    return sample_list


def get_list(args, type):
    print(f'choose data {args.iSAID_get_list}, split:{args.iSAID_get_list_split}')
    if args.iSAID_get_list == 'by_crop':
        sample_list = get_list_by_cropped_images(args.iSAID_data_path, type, choose_size=args.data_choose_size)
    elif args.iSAID_get_list == 'by_original':
        sample_list = get_list_by_original_image(args.iSAID_data_path, type, choose_size=args.data_choose_size,
                                                 split=args.iSAID_get_list_split)
    else:
        raise NotImplementedError
    return sample_list

def convert_one_set_image(image, instance, label):
    sample ={}
    sample['image'] = Image.open(image).convert('RGB')
    sample['im_name'] = os.path.split(image)[1]
    sample['instance'] = Image.open(instance).convert('RGB')
    sample['label'] = Image.open(label).convert('RGB')

    colordecode = ColorDecode()
    normalize = Normalize()
    to_tensor = ToTensor()
    get_obbox = GetOrientedBBoxOneHotTensor(15)
    get_bbox = GetBBoxOneHotTensor(15)

    sample = colordecode(sample)
    sample = to_tensor(sample)
    sample = get_bbox(sample)
    sample = get_obbox(sample)
    sample = normalize(sample)
    return sample

class iSAIDDataset(Dataset):
    Label = namedtuple('Label', [
        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images with train IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
        'm_color',  # The color of this label
    ])

    labels = [
        #     name   id  trainId category  catId  hasInstances ignoreInEval color multiplied color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0), 0),
        Label('ship', 1, 0, 'transport', 1, True, False, (0, 0, 63), 4128768),
        Label('storage_tank', 2, 1, 'transport', 1, True, False, (0, 63, 63), 4144896),
        Label('baseball_diamond', 3, 2, 'land', 2, True, False, (0, 63, 0), 16128),
        Label('tennis_court', 4, 3, 'land', 2, True, False, (0, 63, 127), 8339200),
        Label('basketball_court', 5, 4, 'land', 2, True, False, (0, 63, 191), 12533504),
        Label('Ground_Track_Field', 6, 5, 'land', 2, True, False, (0, 63, 255), 16727808),
        Label('Bridge', 7, 6, 'land', 2, True, False, (0, 127, 63), 4161280),
        Label('Large_Vehicle', 8, 7, 'transport', 1, True, False, (0, 127, 127), 8355584),
        Label('Small_Vehicle', 9, 8, 'transport', 1, True, False, (0, 0, 127), 8323072),
        Label('Helicopter', 10, 9, 'transport', 1, True, False, (0, 0, 191), 12517376),
        Label('Swimming_pool', 11, 10, 'land', 2, True, False, (0, 0, 255), 16711680),
        Label('Roundabout', 12, 11, 'land', 2, True, False, (0, 191, 127), 8371968),
        Label('Soccer_ball_field', 13, 12, 'land', 2, True, False, (0, 127, 191), 12549888),
        Label('plane', 14, 13, 'transport', 1, True, False, (0, 127, 255), 16744192),
        Label('Harbor', 15, 14, 'transport', 1, True, False, (0, 100, 155), 10183680),
    ]

    def __init__(self, args, root_dir='./', type="train", choose_size=None, transform=None, repeat=1):
        assert type in ('train', 'val', 'test')
        self.type = type
        # get image and instance list
        sample_list = get_list(args, type)
        self.sample_list = sample_list * repeat
        self.image_root = os.path.join(root_dir, f'iSAID_patches/{type}/images')
        self.real_size = len(self.sample_list)
        self.transform = transform
        self.colordecode = ColorDecode()
        self.normalize = Normalize()
        self.to_tensor = ToTensor()
        self.get_bbox = GetOrientedBBoxOneHotTensor(15) if 'obox' in args.aug else GetBBoxOneHotTensor(15)

        print(f'iSAID Dataset created: set:{type} | choose_size:{choose_size} | size: {self.__len__()}\n')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        # index = random.randint(0, self.real_size - 1)
        sample_dict = self.sample_list[index]
        id = sample_dict['id']
        file_name = os.path.join(self.image_root, sample_dict['file_name'])
        ins_file_name = os.path.join(self.image_root, sample_dict['ins_file_name'])
        seg_file_name = os.path.join(self.image_root, sample_dict['seg_file_name'])

        sample = {}
        sample['image'] = Image.open(file_name).convert('RGB')
        sample['im_name'] = file_name
        sample['id'] = id
        sample['instance'] = Image.open(ins_file_name).convert('RGB')
        sample['label'] = Image.open(seg_file_name).convert('RGB')
        sample['index'] = index

        # aug
        sample = self.colordecode(sample)
        sample = self.to_tensor(sample)
        sample = self.get_bbox(sample)
        sample = self.normalize(sample)
        # if self.transform is not None:
        #     sample = self.transform(sample)
        return sample

    def get_dict(self, index):
        return self.sample_list[index]

    @classmethod
    def decode_instance(cls, instance: np.array, label: np.array):
        """
        deprecated
        :param instance:
        :param label:
        :return:
        """
        assert instance.ndim == 2
        assert label.ndim == 2
        # instance_new = np.zeros((instance.shape[0], instance.shape[1]), dtype=np.uint8)
        # Note that when you load the PNG as an RGB image, you will need to compute the ids via ids=R+G*256+B*256^2.

        instance_map = np.zeros((instance.shape[0], instance.shape[1]), dtype=np.int32)
        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros((instance.shape[0], instance.shape[1]), dtype=np.int32)

        for c in range(1, len(cls.labels)):
            mask = np.logical_and(label == c, instance > 0)
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(instance[mask])
                instance_map[mask] = ids + np.amax(instance_map)
                # class_map[mask] = i + 1

        return instance_map, label


class BalancediSAIDDataset(Dataset):
    def __init__(self, args, crop_size, fake_size, label, root_dir='./', type="train",
                 choose_size=None, transform=None, repeat=1, sample_list=None):

        self.image_root = os.path.join(root_dir, f'iSAID_patches/{type}/images')
        if sample_list is None:
            sample_list = get_list(root_dir, type, choose_size)

        self.sample_list = sample_list * repeat
        self.real_size = len(self.sample_list)

        self.transform = transform
        self.fake_size = fake_size if fake_size is not None else self.real_size
        self.label = label

        self.cls_dict = {}
        with open(os.path.join(root_dir, f'iSAID_patches/{type}/classes_info.txt'), 'r') as f:
            f = f.readlines()
            f = [s.split() for s in f]
            for item in f:
                self.cls_dict[int(item[0])] = [int(i) for i in item[1:]]

        mask = []
        for i, sample in enumerate(self.sample_list):
            if sample['id'] not in self.cls_dict:
                continue
            classes = self.cls_dict[sample['id']]
            if self.label in classes:
                mask.append(i)
        self.sample_list = [self.sample_list[i] for i in mask]
        self.real_size = len(self.sample_list)
        # Aug
        self.resize = Resize()
        self.choose_label_object = CropRandomObject(label=label, size=crop_size)
        self.hflip = RandomHorizontalFlip(p=0.5)
        self.vflip = RandomVerticalFlip(p=0.5)
        self.colordecode = ColorDecode()
        self.normalize = Normalize()
        self.to_tensor = ToTensor()
        self.get_bbox = GetOrientedBBoxOneHotTensor(15) if 'obox' in args.aug else GetBBoxOneHotTensor(15)

        self.decode_instance = iSAIDDataset.decode_instance
        print(f'===> label:{self.label} | size:{self.real_size} | fake_size:{self.fake_size}')

    def __getitem__(self, index):
        # try:
        index = random.randint(0, self.real_size - 1)
        sample_dict = self.sample_list[index]

        id = sample_dict['id']
        file_name = os.path.join(self.image_root, sample_dict['file_name'])
        ins_file_name = os.path.join(self.image_root, sample_dict['ins_file_name'])
        seg_file_name = os.path.join(self.image_root, sample_dict['seg_file_name'])

        sample = {}
        sample['image'] = Image.open(file_name).convert('RGB')
        sample['im_name'] = file_name
        sample['id'] = id
        sample['instance'] = Image.open(ins_file_name).convert('RGB')
        sample['label'] = Image.open(seg_file_name).convert('RGB')
        # sample['dict'] = sample_dict

        h, w = sample_dict['height'], sample_dict['width']
        # 将不规则patch短边缩放，再进行增广
        if h != 800 or w != 800:
            scale = int(800 / min(h, w) + 1)
            sample = self.resize(sample, (h * scale, w * scale))
        # aug

        sample = self.colordecode(sample)
        sample = self.choose_label_object(sample)
        sample = self.vflip(self.hflip(sample))
        sample = self.to_tensor(sample)
        sample = self.get_bbox(sample)
        sample = self.normalize(sample)

        # if self.transform is not None:
        #     sample = self.transform(sample)
        return sample

        # except Exception:
        #     print(f'*** loading {index} fail, get next...')
        #     return self.__getitem__(random.randint(0, self.__len__()))

    def __len__(self):
        return self.fake_size


class iSAIDTestDataset(Dataset):
    def __init__(self, args, root_dir='./'):
        # get image and instance list
        self.image_root = os.path.join(root_dir, 'iSAID_patches/test/images')
        self.img_list = glob.glob(os.path.join(self.image_root, 'P*.png'))
        self.img_list.sort()
        self.id_list = list(range(len(self.img_list)))
        self.colordecode = ColorDecode()
        self.normalize = Normalize()
        self.to_tensor = ToTensor()
        self.get_bbox = GetOrientedBBoxOneHotTensor(15) if 'obox' in args.aug else GetBBoxOneHotTensor(15)
        print(f'iSAID Dataset created: Test set | size: {self.__len__()}\n')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # index = random.randint(0, self.real_size - 1)
        id = self.id_list[index]
        file_name = self.img_list[index]
        sample = {}
        img = Image.open(file_name).convert('RGB')
        w, h = img.size
        sample['image'] = img
        sample['im_name'] = os.path.split(file_name)[1]
        sample['id'] = id
        sample['instance'] = Image.fromarray(np.zeros((h, w))).convert('RGB')
        sample['label'] = Image.fromarray(np.zeros((h, w))).convert('RGB')
        sample = self.colordecode(sample)
        sample = self.to_tensor(sample)
        sample = self.get_bbox(sample)
        sample = self.normalize(sample)
        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, sample):
        # print(type(img), type(target), type(bbox_target))
        sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        sample['image'] = F.to_tensor(sample['image'])
        sample['instance'] = torch.tensor(sample['instance']).unsqueeze(dim=0)
        sample['label'] = torch.tensor(sample['label']).unsqueeze(dim=0)
        return sample


class CropRandomObject:
    def __init__(self, label, keys=('image', 'instance', 'label'), object_key="instance", size=100):
        self.label = label
        self.keys = keys
        self.object_key = object_key
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, sample):
        # sample['instance'] is numpy array
        object_map = sample[self.object_key]
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]
        # sample['label'] is numpy array
        label = sample['label']
        unique_labels = np.unique(label)
        unique_labels = unique_labels[unique_labels != 0]
        instance_to_label = {int(i): int(np.unique(label[object_map == i])) for i in unique_objects}
        # print('instance_to_label',instance_to_label)
        label_to_instance = {}
        for i in unique_objects:
            i = int(i)
            if label_to_instance.get(instance_to_label[i]) is None:
                label_to_instance[instance_to_label[i]] = []
                label_to_instance[instance_to_label[i]].append(i)
            else:
                label_to_instance[instance_to_label[i]].append(i)
        # print('label_to_instance', label_to_instance)

        if unique_labels.size > 0:
            if self.label in unique_labels:
                choose_label = self.label
            else:
                choose_label = np.random.choice(unique_labels, 1)

            random_id = np.random.choice(label_to_instance[int(choose_label)], 1)

            y, x = np.where(object_map == random_id)
            ym, xm = np.mean(y), np.mean(x)

            i = int(np.clip(ym - self.size[1] / 2, 0, h - self.size[1]))
            j = int(np.clip(xm - self.size[0] / 2, 0, w - self.size[0]))

        else:
            i = random.randint(0, h - self.size[1])
            j = random.randint(0, w - self.size[0])

        sample['image'] = F.crop(sample['image'], i, j, self.size[1], self.size[0])
        sample['instance'] = sample['instance'][i:i + self.size[1], j:j + self.size[0]]
        sample['label'] = sample['label'][i:i + self.size[1], j:j + self.size[0]]
        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            sample['image'] = F.hflip(sample['image'])
            sample['label'] = sample['label'][:, ::-1].copy()
            sample['instance'] = sample['instance'][:, ::-1].copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            sample['image'] = F.vflip(sample['image'])
            sample['label'] = sample['label'][::-1, :].copy()
            sample['instance'] = sample['instance'][::-1, :].copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class iSAIDIDMapping:
    def __init__(self):
        self.id_mapping = self.get_id_mapping()

    def __call__(self, seg: np.array):
        res = np.zeros(seg.shape).astype(np.int32)
        for m_color in self.id_mapping:
            res[seg == m_color] = self.id_mapping[m_color]
        return res

    @staticmethod
    def get_id_mapping():
        d = [i._asdict() for i in iSAIDDataset.labels]
        # print(d)
        id_mapping = {}
        for i in d:
            id_mapping[i['m_color']] = i['id']
        # print(id_mapping)
        return id_mapping


class ColorDecode:
    def __init__(self):
        self.color_mapping = iSAIDIDMapping()

    def __call__(self, sample):
        instance = np.array(sample['instance'])
        label = np.array(sample['label'])

        instance = instance[:, :, 2] * 256 ** 2 + instance[:, :, 1] * 256 + instance[:, :, 0]
        label = label[:, :, 2] * 256 ** 2 + label[:, :, 1] * 256 + label[:, :, 0]
        label = self.color_mapping(label)

        assert instance.ndim == 2
        assert label.ndim == 2
        # Note that when you load the PNG as an RGB image, you will need to compute the ids via ids=R+G*256+B*256^2.
        instance_map = np.zeros((instance.shape[0], instance.shape[1]), dtype=np.int32)
        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg

        for c in range(1, len(iSAIDDataset.labels)):
            mask = np.logical_and(label == c, instance > 0)
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(instance[mask])
                instance_map[mask] = ids + np.amax(instance_map)

        sample['instance'] = instance_map
        sample['label'] = label
        return sample


class Resize:
    def __init__(self, ):
        pass

    def __call__(self, sample, size):
        sample['image'] = F.resize(sample['image'], size, Image.BILINEAR)
        sample['instance'] = F.resize(sample['instance'], size, Image.NEAREST)
        sample['label'] = F.resize(sample['label'], size, Image.NEAREST)
        return sample


def get_iSAID_dataset(args, train=True):
    if 'balance' in args.aug and train:
        aug_save = args.aug
        temp = 'std' if 'std' in args.aug else ''
        temp += '_color' if 'color' in args.aug else ''
        temp += '_bbox' if 'bbox' in args.aug else ''
        temp += '_obox' if 'obox' in args.aug else ''
        args.aug = temp
        # train_transform, val_transform = get_transform(args)
        # it will take long time to read json, so read once and use it 15 times
        sample_list = get_list(args, 'train')
        print('==========')
        print('1 Color decode to numpy')
        print(f'2 Object center crop as {args.crop_size}')
        print('3 RandomHorizontalFlip')
        print('4 RandomVerticalFlip')
        print('5 from numpy to tensor')
        print('6 Normalize')
        print('==========')
        bbox_transform = GetOrientedBBoxOneHotTensor(args.n_class) if 'obox' in args.aug \
            else GetBBoxOneHotTensor(args.n_class)
        data_sets = [
            BalancediSAIDDataset(args,
                                 crop_size=args.crop_size,
                                 fake_size=args.iSAID_fake_size // 15,
                                 label=i,
                                 root_dir=args.iSAID_data_path,
                                 type="train", choose_size=args.data_choose_size,
                                 transform=bbox_transform,
                                 repeat=1, sample_list=sample_list) for i in range(1, 16)]
        non_empty = []
        for i in range(len(data_sets)):
            if data_sets[i].real_size > 0:
                non_empty.append(i)
        data_sets = [data_sets[i] for i in non_empty]

        args.aug = aug_save
        # if args.data_choose_size is None:
        #     fake_size = 3000 // 8
        # else:
        #     fake_size = max([d.real_size for d in data_sets])

        data_set = ConcatDataset(data_sets)
        print(f'===> ConcatDataset size:{len(data_set)}')
        return data_set
    else:
        # train_transform, val_transform = get_transform(args)
        # bbox_transform = GetBBoxOneHotTensor(args.n_class) if 'bbox' in args.aug else None
        #  20200416
        bbox_transform = GetOrientedBBoxOneHotTensor(args.n_class) if 'obox' in args.aug \
            else GetBBoxOneHotTensor(args.n_class)
        if train:
            data_set = iSAIDDataset(args,
                                    root_dir=args.iSAID_data_path, type="train",
                                    choose_size=args.data_choose_size,
                                    transform=bbox_transform, repeat=args.dataset_repeat)
        else:
            data_set = iSAIDDataset(args,
                                    root_dir=args.iSAID_data_path, type="val",
                                    choose_size=args.data_choose_size, transform=bbox_transform)

        return data_set


def get_iSAID_dataloader(args, train=True):
    data_set = get_iSAID_dataset(args, train)
    if train:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.prefetch,
                                 pin_memory=False,
                                 drop_last=True)
    else:
        data_loader = DataLoader(dataset=data_set,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.prefetch,
                                 pin_memory=False,
                                 drop_last=False)
    return data_loader


def get_test_dataloader(args):
    data_set = iSAIDTestDataset(args, root_dir=args.iSAID_data_path)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=args.prefetch,
                             pin_memory=False,
                             drop_last=False)
    return data_loader


def test():
    from train_config import get_config
    from tools.utils_for_weakly import get_bbox_instance
    args = get_config()
    args.aug = 'std_obox'
    # args.aug = 'std_bbox'
    args.n_class = 15
    # data_loader = get_test_dataloader(args)
    data_loader = get_iSAID_dataloader(args, False)
    for i, sample in enumerate(data_loader):
        image = sample['image']
        instance = sample['instance']
        label = sample['label']
        print(sample['id'])
        print(sample['im_name'])
        bbox = sample['obbox']
        # bbox = sample['bbox']
        print(bbox.sum())
        # bbox_instance = get_bbox_instance(instance[0], label[0])
        # bboxes = torch.zeros(1, image.size(-2), image.size(-1))
        # for i in bbox_instance:
        #     box_temp = i['mask'].unsqueeze(dim=0)
        #     bboxes[box_temp > 0] = i['cls']

        transforms.ToPILImage()(image[0]).show()
        # transforms.ToPILImage()(bboxes).show()
        transforms.ToPILImage()(instance[0] * 25).show()
        transforms.ToPILImage()(label[0] * 50).show()
        # transforms.ToPILImage()(bbox_instance[0]*20).show()
        bbox_temp = bbox[0].argmax(dim=0).float()
        transforms.ToPILImage()(bbox_temp * 25).show()
        print(instance.unique())
        print(label.unique())
        # print(bbox_temp.unique())

        print('Next')
        # print(sample)
        pass


def test_weakly_dataset():
    from train_config import get_config
    args = get_config()
    args.data_choose_size = 914
    data_loader_1 = get_iSAID_dataset(args, True)
    args.data_choose_size = -914
    data_loader_2 = get_iSAID_dataset(args, True)

    # check overlap:
    for img1 in data_loader_1.image_list:
        for img2 in data_loader_2.image_list:
            if img1 == img2:
                raise Exception('Overlap!')
    pass


def cat_data_set_test():
    from train_config import get_config
    args = get_config()
    cat_data_set = get_iSAID_dataset(args, train=True)
    print(len(cat_data_set))
    data_loader = DataLoader(cat_data_set, shuffle=True)
    for sample in data_loader:
        print(sample['image'].shape)
        print(sample['image'].type())
        print(sample['label'].type())
        print(sample['instance'].type())
        pass


def test_label_data_set():
    from train_config import get_config
    args = get_config()
    data_set = BalancediSAIDDataset(crop_size=512, fake_size=41, label=6, root_dir=args.iSAID_data_path,
                                    type="train", choose_size=914, transform=None, repeat=1)
    print(data_set.__len__())
    for sample in data_set:
        image = sample['image']
        image.show()


def test_img_list():
    data_path = '/Users/chenlinwei/dataset/iSAID'
    img_list, label_list, instance_list = get_list(data_path, 'val', None)
    print(len(img_list))
    print(len(label_list))
    print(len(instance_list))


def test_test_set():
    from train_config import get_config
    # from tools.utils_for_weakly import get_bbox_instance
    args = get_config()
    data_set = iSAIDTestDataset(root_dir=args.iSAID_data_path)
    # sample = data_set.__getitem__(1631)
    sample = data_set.__getitem__(148)
    print(sample['im_name'])
    Image.open(sample['im_name']).show()

    # val_dataset = get_iSAID_dataset(args, train=False)
    # sample = val_dataset.__getitem__(7425)
    # print(sample['im_name'])
    # print(sample['image'].shape)
    # Image.open(sample['im_name']).show()

    # /Users/chenlinwei/dataset/iSAID/iSAID_patches/val/images/P2150.png
    # torch.Size([3, 1080, 626])


def test_get_list():
    sample_list = get_list(root_dir='/Users/chenlinwei/dataset/iSAID', type='val', choose_size=None, bias_to_minor=True)
    for s in sample_list:
        print(s, s['annotations'])
        for a in s['annotations']:
            assert a['image_id'] == s['id']
    pass


if __name__ == '__main__':
    # cal_forground_weight()
    from train_config import config

    args = config()
    # args.data_choose_size = None
    # dataset = get_iSAID_dataloader(args, train=True)
    # cat_data_set_test()
    # test_label_data_set()
    # test_img_list()
    test()
    # test_test_set()
    # test_get_list()

    pass
