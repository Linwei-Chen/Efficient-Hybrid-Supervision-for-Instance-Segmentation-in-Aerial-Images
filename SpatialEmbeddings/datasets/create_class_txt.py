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
from torchvision.transforms import functional as F
from os import path as osp
from train_config import get_config
from datasets.CityscapesDataset import CityscapesDataset
from tqdm import tqdm


def create_txt():
    args = get_config()
    type = 'train'
    root_dir = args.cityscapes_data_path
    image_list = glob.glob(os.path.join(root_dir, 'leftImg8bit/{}/'.format(type), '*/*.png'))
    image_list.sort()

    instance_list = glob.glob(os.path.join(root_dir, 'gtFine/{}/'.format(type), '*/*instanceIds*.png'))
    instance_list.sort()

    assert len(instance_list) == len(image_list)

    size = len(image_list)
    mask = list(range(size))

    image_list = [image_list[i] for i in mask]
    instance_list = [instance_list[i] for i in mask]
    for image_path, instance_path in tqdm(zip(image_list, instance_list)):
        # image = Image.open(image_path)
        # sample['image'] = image
        # sample['im_name'] = self.image_list[index]

        # load instances
        instance = Image.open(instance_path)
        instance, label = CityscapesDataset.decode_instance(instance, class_id=None)
        label = np.array(label)
        for cls in range(1, 9):
            if cls in label:
                cls_name = CityscapesDataset.class_names[cls - 1]
                with open(osp.join(root_dir, 'each_class', cls_name + '.txt'), 'a+') as f:
                    image_relpath = osp.relpath(image_path, start=root_dir)
                    instance_relpath = osp.relpath(instance_path, start=root_dir)
                    f.write(f'{image_relpath} {instance_relpath}\n')


def create_cls_info():
    args = get_config()
    type = 'train'
    root_dir = args.cityscapes_data_path
    image_list = glob.glob(os.path.join(root_dir, 'leftImg8bit/{}/'.format(type), '*/*.png'))
    image_list.sort()

    instance_list = glob.glob(os.path.join(root_dir, 'gtFine/{}/'.format(type), '*/*instanceIds*.png'))
    instance_list.sort()

    assert len(instance_list) == len(image_list)

    size = len(image_list)
    mask = list(range(size))

    image_list = [image_list[i] for i in mask]
    instance_list = [instance_list[i] for i in mask]
    for image_path, instance_path in tqdm(zip(image_list, instance_list)):
        # image = Image.open(image_path)
        # sample['image'] = image
        # sample['im_name'] = self.image_list[index]

        # load instances
        instance = Image.open(instance_path)
        instance, label = CityscapesDataset.decode_instance(instance, class_id=None)
        label_np = np.array(label)

        label_ids = np.unique(label_np)
        label_ids = label_ids[label_ids != 0]
        label_ids = [int(i) for i in label_ids]

        with open(osp.join(root_dir, 'classes_info.txt'), 'a+') as f:
            relpath = osp.relpath(image_path, start=root_dir)
            for i in label_ids:
                relpath += ' ' + str(i)
            relpath += '\n'
            f.write(relpath)


def parse_txt(txt_path):
    with open(txt_path, 'r') as f:
        f = f.readlines()
        f = [s.split() for s in f]
        return f


if __name__ == '__main__':
    # parse_txt('/Users/chenlinwei/dataset/cityscapes/each_class/bicycle.txt')
    create_cls_info()
