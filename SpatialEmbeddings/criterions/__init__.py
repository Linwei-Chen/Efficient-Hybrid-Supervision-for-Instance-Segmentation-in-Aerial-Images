from criterions.multi_cls_my_loss import SpatialEmbLoss as MultiSpatialEmbLoss
from criterions.my_loss import SpatialEmbLoss
from criterions.weakly_loss import WeaklySpatialEmbLoss
from torch import nn
from train_utils import get_device
from criterions.my_loss import LovaszSoftMaxWithCE, FocalLoss
from criterions.spatial_density_loss import SpatialDensityEmbLoss
from criterions.multi_offset_loss import SpatialEmbLossMultiOffset


def get_segloss(args):
    if args.seg_criterion == 'l+ce':
        if args.n_class in (8, 15, 20):
            criterion = LovaszSoftMaxWithCE(n_class=args.n_class + 1)
        elif args.n_class in (9, 16, 21):
            criterion = LovaszSoftMaxWithCE(n_class=args.n_class)
        else:
            raise NotImplementedError
    elif args.seg_criterion == 'fbce':
        criterion = FocalLoss(n_class=args.n_class + 1, mode='bce')
    else:
        raise NotImplementedError

    criterion = nn.DataParallel(criterion).to(get_device(args))
    return criterion


def get_criterion(args):
    if 'weakly' in args.criterion:
        if args.n_class == 8:
            criterion = WeaklySpatialEmbLoss(n_class=8, to_center='to_center' in args.criterion,
                                             n_sigma=args.n_sigma, foreground_weight=args.foreground_weight,
                                             **args.loss_w)
        elif args.n_class == 9:
            raise NotImplementedError

    elif 'bce' in args.criterion:
        if args.n_class in (8, 15, 20):
            criterion = SpatialEmbLoss(args)
        elif args.n_class == 9:
            criterion = MultiSpatialEmbLoss(n_class=9, to_center='to_center' in args.criterion, mode='bce', strict=True,
                                            n_sigma=args.n_sigma, foreground_weight=args.foreground_weight,
                                            **args.loss_w)
        else:
            raise NotImplementedError

    elif 'ce' in args.criterion:
        assert args.n_class == 9
        criterion = MultiSpatialEmbLoss(n_class=9, to_center='to_center' in args.criterion, mode='ce', strict=False,
                                        n_sigma=args.n_sigma, foreground_weight=args.foreground_weight,
                                        **args.loss_w)
    elif 'density' in args.criterion:
        criterion = SpatialDensityEmbLoss(args)
    elif 'multioffset' in args.criterion:
        criterion = SpatialEmbLossMultiOffset(args)
    else:
        raise NotImplementedError
    print(f'===> instance criterion:{criterion.__class__.__name__} | n_class: {args.n_class}')
    device = get_device(args)
    criterion = nn.DataParallel(criterion)
    criterion = criterion.to(device)
    return criterion


def _test():
    from train_config import get_config
    args = get_config()
    criterion = get_criterion(args)


if __name__ == '__main__':
    _test()
