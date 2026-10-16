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

class_ids_to_index = {
    24: 1, 25: 2, 26: 3, 27: 4, 28: 5, 31: 6, 32: 7, 33: 8
}


def get_list(root_dir, type, choose_size, bias_to_6547=True):
    cls_dict = {}
    with open(os.path.join(root_dir, 'classes_info.txt'), 'r') as f:
        f = f.readlines()
        f = [s.split() for s in f]
        for item in f:
            cls_dict[os.path.join(root_dir, item[0])] = [int(i) for i in item[1:]]

    image_list = glob.glob(os.path.join(root_dir, 'leftImg8bit/{}/'.format(type), '*/*.png'))
    image_list.sort()

    instance_list = glob.glob(os.path.join(root_dir, 'gtFine/{}/'.format(type), '*/*instanceIds*.png'))
    instance_list.sort()
    print(len(instance_list), len(image_list))
    assert len(instance_list) == len(image_list)

    size = len(image_list)
    mask = list(range(size))
    if type == 'train' and choose_size is not None:
        if choose_size < 0:
            choose_rest = True
            choose_size = abs(choose_size)
        else:
            choose_rest = False

        if bias_to_6547:
            mask = []
            for c in range(4):
                cls = (6, 5, 4, 7)[c]
                cls_max = (142, 274, 359, 510)
                for i in range(size):
                    if len(mask) >= (choose_size * 0.25 * (c + 1)):
                        break
                    # idx = choose_size * i // size
                    idx = i
                    if idx not in mask and any([i in cls_dict[image_list[idx]] for i in (cls,)]):
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
    return [image_list[i] for i in mask], [instance_list[i] for i in mask]


class CityscapesDataset(Dataset):
    class_names = ('person', 'rider', 'car', 'truck',
                   'bus', 'train', 'motorcycle', 'bicycle')
    class_ids = (24, 25, 26, 27, 28, 31, 32, 33)

    def __init__(self, root_dir='./', type="train", choose_size=None, transform=None, repeat=1):
        assert type in ('train', 'val', 'test')
        self.type = type
        # get image and instance list
        image_list, instance_list = get_list(root_dir, type, choose_size)
        self.image_list = image_list * repeat
        self.instance_list = instance_list * repeat
        self.real_size = len(self.image_list)
        self.transform = transform

        print(f'Cityscapes Dataset created: set:{type} | choose_size:{choose_size} | size: {self.__len__()}\n')

    def __len__(self):
        return self.real_size

    def __getitem__(self, index):
        # index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}
        # print(self.type)
        # load image
        image = Image.open(self.image_list[index])
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = self.decode_instance(instance, class_id=None)
        sample['instance'] = instance
        sample['label'] = label

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, class_id=None):
        pic = np.array(pic, copy=False)

        instance_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        # contains the class of each instance, but will set the class of "unlabeled instances/groups" to bg
        class_map = np.zeros(
            (pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if class_id is not None:
            mask = np.logical_and(pic >= class_id * 1000, pic < (class_id + 1) * 1000)
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])
                instance_map[mask] = ids
                class_map[mask] = 1
        else:
            for i, c in enumerate(cls.class_ids):
                mask = np.logical_and(pic >= c * 1000, pic < (c + 1) * 1000)
                if mask.sum() > 0:
                    ids, _, _ = relabel_sequential(pic[mask])
                    instance_map[mask] = ids + np.amax(instance_map)
                    class_map[mask] = i + 1

        return Image.fromarray(instance_map), Image.fromarray(class_map)


class BalancedCityscapesDataset(Dataset):
    def __init__(self, crop_size, fake_size, label, root_dir='./', type="train",
                 choose_size=None, transform=None, repeat=1):

        image_list, instance_list = get_list(root_dir, type, choose_size)
        self.image_list = image_list * repeat
        self.instance_list = instance_list * repeat
        self.real_size = len(self.image_list)
        self.transform = transform
        self.choose_label_object = CropRandomObject(label=label, size=crop_size)
        self.fake_size = fake_size if fake_size is not None else self.real_size
        self.label = label
        self.cls_dict = {}
        with open(os.path.join(root_dir, 'classes_info.txt'), 'r') as f:
            f = f.readlines()
            f = [s.split() for s in f]
            for item in f:
                self.cls_dict[os.path.join(root_dir, item[0])] = [int(i) for i in item[1:]]
        mask = []
        for i, img in enumerate(self.image_list):
            classes = self.cls_dict[img]
            if self.label in classes:
                mask.append(i)
        self.image_list = [self.image_list[i] for i in mask]
        self.instance_list = [self.instance_list[i] for i in mask]
        self.real_size = len(self.image_list)
        print(f'===> label:{self.label} | size:{self.real_size} | fake_size:{self.fake_size}')

    def __getitem__(self, index):
        # print(self.label)
        index = random.randint(0, self.real_size - 1)
        sample = {}
        # print(self.type)
        # load image
        image = Image.open(self.image_list[index])
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        # load instances
        instance = Image.open(self.instance_list[index])
        instance, label = CityscapesDataset.decode_instance(instance, class_id=None)
        sample['instance'] = instance
        sample['label'] = label
        sample = self.choose_label_object(sample)

        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

    def __len__(self):
        return self.fake_size


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
        object_map = np.array(sample[self.object_key], copy=False)
        h, w = object_map.shape

        unique_objects = np.unique(object_map)
        unique_objects = unique_objects[unique_objects != 0]

        label = np.array(sample['label'])
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

        for k in self.keys:
            assert (k in sample)

            sample[k] = F.crop(sample[k], i, j, self.size[1], self.size[0])

        return sample


def get_cityscapes_dataset(args, train=True):
    if 'balance' in args.aug and train:
        aug_save = args.aug
        temp = 'std' if 'std' in args.aug else ''
        temp += '_color' if 'color' in args.aug else ''
        temp += '_bbox' if 'bbox' in args.aug else ''
        args.aug = temp
        train_transform, val_transform = get_transform(args)
        data_sets = [
            BalancedCityscapesDataset(crop_size=args.crop_size,
                                      fake_size=args.data_choose_size,
                                      label=i,
                                      root_dir=args.cityscapes_data_path,
                                      type="train", choose_size=args.data_choose_size,
                                      transform=train_transform, repeat=1) for i in range(1, 9)]
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

        fake_size = args.cityscapes_fake_size // 8
        for d in data_sets:
            d.fake_size = fake_size

        data_set = ConcatDataset(data_sets)
        print(f'===> ConcatDataset size:{len(data_set)}')
        return data_set
    else:
        train_transform, val_transform = get_transform(args)
        if train:
            data_set = CityscapesDataset(root_dir=args.cityscapes_data_path, type="train",
                                         choose_size=args.data_choose_size, transform=train_transform,
                                         repeat=args.dataset_repeat)
        else:
            data_set = CityscapesDataset(root_dir=args.cityscapes_data_path, type="val",
                                         choose_size=args.data_choose_size, transform=val_transform)

        return data_set


def get_cityscapes_dataloader(args, train=True):
    data_set = get_cityscapes_dataset(args, train)
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
    train_transform, val_transform = get_transform(args)
    data_set = CityscapesDataset(root_dir=args.cityscapes_data_path, type="test",
                                 choose_size=args.data_choose_size, transform=val_transform)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             num_workers=args.prefetch,
                             pin_memory=False,
                             drop_last=False)
    return data_loader


def my_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    # from torch.utils.data.dataloader import de
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    # return (torch.stack(imgs, 0),
    #         torch.stack([i[..., :4] for i in targets], 0),
    #         torch.stack([i[..., 4] for i in targets], 0).long())
    # return imgs, targets
    return torch.stack(imgs, 0), targets


def test():
    from train_config import get_config
    from tools.utils_for_weakly import get_bbox_instance
    args = get_config()
    data_loader = get_cityscapes_dataloader(args, True)
    for i, sample in enumerate(data_loader):
        image = sample['image']
        instance = sample['instance']
        label = sample['label']
        # bbox = sample['bbox']
        bbox_instance = get_bbox_instance(instance[0], label[0])
        bboxes = torch.zeros(1, image.size(-2), image.size(-1))
        for i in bbox_instance:
            box_temp = i['mask'].unsqueeze(dim=0)
            bboxes[box_temp > 0] = i['cls']

        transforms.ToPILImage()(image[0]).show()
        transforms.ToPILImage()(bboxes).show()
        # transforms.ToPILImage()(instance[0]*10).show()
        # transforms.ToPILImage()(label[0]*50).show()
        # transforms.ToPILImage()(bbox_instance[0]*20).show()
        # bbox_temp = bbox[0].argmax(dim=0).float()
        # transforms.ToPILImage()(bbox_temp*50).show()
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
    data_loader_1 = get_cityscapes_dataset(args, True)
    args.data_choose_size = -914
    data_loader_2 = get_cityscapes_dataset(args, True)

    # check overlap:
    for img1 in data_loader_1.image_list:
        for img2 in data_loader_2.image_list:
            if img1 == img2:
                raise Exception('Overlap!')
    pass


def cat_data_set_test():
    from train_config import get_config
    args = get_config()
    cat_data_set = get_cityscapes_dataset(args, train=True)
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
    data_set = BalancedCityscapesDataset(crop_size=512, fake_size=41, label=6, root_dir=args.cityscapes_data_path,
                                         type="train", choose_size=914, transform=None, repeat=1)
    print(data_set.__len__())
    for sample in data_set:
        image = sample['image']
        image.show()


def cal_forground_weight():
    n = [17.9, 1.8, 26.9, 0.5, 0.4, 0.2, 0.7, 3.7]
    s = sum(n)
    p = [i / s for i in n]
    w = [1 / math.log(1.1 + p_c) for p_c in p]
    print(w)
    res_w = [2.72, 7.92, 2.08, 9.61, 9.78, 10.12, 9.31, 6.33]

    pics = [2.324, 1.019, 2.831, 0.359, 0.274, 0.142, 0.510, 1.537]
    # balance_n = [math.ceil(i / p) for i, p in zip(n,pics)]
    balance_n = [(i / p) for i, p in zip(n, pics)]
    print(balance_n)
    balance_s = sum(balance_n)
    balance_p = [i / balance_s for i in balance_n]
    balance_w = [1 / math.log(1.1 + p_c) for p_c in balance_p]
    print(balance_w)
    # when balance_n is int
    res_w = [3.03, 7.55, 2.40, 7.55, 7.55, 7.55, 7.55, 5.94]
    float_res = [3.07, 6.53, 2.68, 7.09, 6.98, 7.06, 7.12, 5.77]


def analyze_inst_and_label():
    from train_config import config
    from tqdm import tqdm
    args = config()
    args.aug = 'std'
    args.batch_size = 1
    args.data_choose_size = None
    dataseg = get_cityscapes_dataloader(args, train=True)
    inst_sum, cls_sum = 0, 0
    for i, sample in tqdm(enumerate(dataseg)):
        inst = len(torch.unique(sample['instance']))
        cls = len(torch.unique(sample['label']))
        inst_sum += inst
        cls_sum += cls
        print(inst, cls)

    print(inst_sum, cls_sum)


if __name__ == '__main__':
    # cal_forground_weight()
    # cat_data_set_test()
    # test_label_data_set()
    analyze_inst_and_label()
