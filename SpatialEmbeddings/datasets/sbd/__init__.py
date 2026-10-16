from datasets.sbd.sbd_instance_segmentation_dataset import SBDInstanceDataset, BalancedSBDInstanceDataset
from datasets.voc.voc_instance_segmentation_dataset import VOCInstanceDataset, BalancedVOCInstanceDataset
from datasets.transforms import get_transform
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image


def get_sbd_instance_dataset(args, include_voc):
    """
    default choose size is None
    :param args:
    :return:
    """
    if 'balance' in args.aug:
        aug_save = args.aug
        temp = 'std' if 'std' in args.aug else ''
        temp += '_color' if 'color' in args.aug else ''
        temp += '_bbox' if 'bbox' in args.aug else ''
        #### 20200330
        temp += 'object_center'
        args.aug = temp
        train_transform, val_transform = get_transform(args)
        data_sets = [
            BalancedSBDInstanceDataset(data_dir=args.sbd_data_path, cls_id=i, fake_size=args.voc_fake_size,
                                       transform=train_transform, choose_size=None,
                                       repeat=1) for i in range(1, 21)]
        if include_voc:
            voc_data_sets = [
                # voc is about 1/7 of sbd
                BalancedVOCInstanceDataset(data_dir=args.voc2012_data_path, cls_id=i, fake_size=args.voc_fake_size // 7,
                                           transform=train_transform, choose_size=None,
                                           repeat=1) for i in range(1, 21)]
            for i in voc_data_sets:
                data_sets.append(i)

        non_empty = []
        for i in range(len(data_sets)):
            if data_sets[i].real_size > 0:
                non_empty.append(i)
        data_sets = [data_sets[i] for i in non_empty]

        args.aug = aug_save

        # fake_size = args.voc_fake_size // 20
        for d in data_sets:
            d.fake_size = d.fake_size // 20

        data_set = ConcatDataset(data_sets)
        print(f'===> ConcatDataset size:{len(data_set)}')
    else:
        train_transform, val_transform = get_transform(args)
        data_set = SBDInstanceDataset(data_dir=args.sbd_data_path, transform=train_transform,
                                      choose_size=args.sbd_data_choose_size, repeat=args.sbd_repeat)
        if include_voc:
            voc_data_set = VOCInstanceDataset(data_dir=args.voc_data_path, transform=train_transform,
                                              choose_size=args.voc_data_choose_size, repeat=args.voc_repeat)
            data_set = ConcatDataset([data_set, voc_data_set])

    return data_set


def get_sbd_instance_dataloader(args, include_voc=False):
    # assert 'balance' not in args.aug

    data_set = get_sbd_instance_dataset(args, include_voc)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.prefetch,
                             pin_memory=False,
                             drop_last=True)
    return data_loader


if __name__ == '__main__':
    from train_config import config
    from torchvision import transforms
    import numpy as np

    args = config()
    data_loader = get_sbd_instance_dataloader(args)
    print(data_loader.__len__())
    for i, sample in enumerate(data_loader):
        imgs = sample['image']
        instances = sample['instance']
        labels = sample['label']
        print(labels.unique())
        print(instances.unique())
        transforms.ToPILImage()(imgs[0]).show()
        transforms.ToPILImage()(instances[0] * 50).show()
        transforms.ToPILImage()(labels[0] * 50).show()

        pass
