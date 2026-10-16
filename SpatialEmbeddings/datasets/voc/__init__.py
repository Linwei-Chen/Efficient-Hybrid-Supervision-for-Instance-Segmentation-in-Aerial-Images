from datasets.voc.voc_instance_segmentation_dataset import VOCInstanceDataset, BalancedVOCInstanceDataset
from datasets.transforms import get_transform
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image


def get_voc_instance_dataset(args, train, repeat):
    if 'balance' in args.aug and train:
        aug_save = args.aug
        temp = 'std' if 'std' in args.aug else ''
        temp += '_color' if 'color' in args.aug else ''
        temp += '_bbox' if 'bbox' in args.aug else ''
        #### 20200330
        temp += 'object_center'
        args.aug = temp
        train_transform, val_transform = get_transform(args)
        data_sets = [
            BalancedVOCInstanceDataset(data_dir=args.voc2012_data_path, cls_id=i, fake_size=args.voc_fake_size,
                                       transform=train_transform, train=True, choose_size=args.data_choose_size,
                                       repeat=1)for i in range(1, 21)]
        non_empty = []
        for i in range(len(data_sets)):
            if data_sets[i].real_size > 0:
                non_empty.append(i)
        data_sets = [data_sets[i] for i in non_empty]

        args.aug = aug_save

        fake_size = args.voc_fake_size // 20
        for d in data_sets:
            d.fake_size = fake_size

        data_set = ConcatDataset(data_sets)
        print(f'===> ConcatDataset size:{len(data_set)}')

    elif train:
        train_transform, val_transform = get_transform(args)
        data_set = VOCInstanceDataset(data_dir=args.voc2012_data_path, transform=train_transform, train=True,
                                      choose_size=args.data_choose_size, repeat=repeat)
    else:
        train_transform, val_transform = get_transform(args)
        data_set = VOCInstanceDataset(data_dir=args.voc2012_data_path, transform=val_transform, train=False)

    return data_set


def get_voc_instance_dataloader(args, train=True, repeat=1):
    data_set = get_voc_instance_dataset(args, train, repeat)
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


def test():
    from train_config import config
    from torchvision import transforms
    import numpy as np

    args = config()
    args.n_class = 20
    # args.crop_size = 256
    args.aug = 'balance_object_center_obox'
    args.data_choose_size = 200
    args.batch_size = 1
    data_loader = get_voc_instance_dataloader(args, train=True, repeat=1)
    print(data_loader.__len__())
    for i, sample in enumerate(data_loader):
        imgs = sample['image']
        instances = sample['instance']
        labels = sample['label']
        obbox = sample['obbox']
        print('label:', labels.unique())
        print('instance:', instances.unique())
        print('obbox', obbox[0][1:].max(dim=0)[0].sum())
        transforms.ToPILImage()(imgs[0]).show()
        transforms.ToPILImage()(instances[0] * 50).show()
        # transforms.ToPILImage()(labels[0] * 50).show()
        # transforms.ToPILImage()((labels[0].squeeze() == 255) * 255).show()
        # print('255:', (labels[0].squeeze() == 255).unique())
        transforms.ToPILImage()((obbox[0][1:].max(dim=0)[0])).show()
        pass
        # break

if __name__ == '__main__':
    test()
    pass
