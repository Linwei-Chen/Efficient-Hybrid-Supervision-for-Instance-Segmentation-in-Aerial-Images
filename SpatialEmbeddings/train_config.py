"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

import torch
from datasets import transforms as my_transforms
import argparse

CITYSCAPES_DIR = os.environ.get('CITYSCAPES_DIR')

import os.path as osp


def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)


def config():
    parser = argparse.ArgumentParser(description='SpacialEmbbeding',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--voc2012_data_path', type=str, default='/Users/chenlinwei/dataset/VOCdevkit/VOC2012',
                        help='Root for the voc dataset.')
    parser.add_argument('--voc_repeat', type=int, default=10, help='Repeat for the voc dataset in train_voc_weakly.py')
    parser.add_argument('--sbd_repeat', type=int, default=1, help='Repeat for the voc dataset in train_voc_weakly.py')
    parser.add_argument('--sbd_data_path', type=str, default='/Users/chenlinwei/dataset/SBD_FULL11355/dataset',
                        help='Root for the voc dataset.')
    parser.add_argument('--cityscapes_data_path', type=str, default=osp.expanduser('~/dataset/cityscapes'),
                        help='Root for the cityscape dataset.')
    parser.add_argument('--iSAID_data_path', type=str, default=osp.expanduser('~/dataset/iSAID'),
                        help='Root for the cityscape dataset.')
    parser.add_argument('--iSAID_get_list', type=str, default='by_crop', choices=['by_crop', 'by_original'],
                        help='the way of choose mask dataset')
    parser.add_argument('--iSAID_get_list_split', type=int, default=(1,), nargs='+',
                        help='the way of choose mask dataset')

    parser.add_argument('--cityscapes_fake_size', type=int, default=3000)
    parser.add_argument('--iSAID_fake_size', type=int, default=3000)
    parser.add_argument('--voc_fake_size', type=int, default=3000)
    parser.add_argument('--data_choose_size', type=int, default=None,
                        help='choose strong data size for semi superveised')

    parser.add_argument('--sbd_data_choose_size', type=int, default=None,
                        help='choose strong data size for semi superveised')

    parser.add_argument('--seg_model_path', type=str, default=osp.expanduser('~/dataset/iSAID'),
                        help='Root for the cityscape dataset.')
    parser.add_argument('--seg_preprocessing', type=str, default='onehot', choices=['softmax', 'onehot',
                                                                                    'ignore', 'gt_mask'],
                        help='Root for the cityscape dataset.')

    parser.add_argument('--dataset_repeat', type=int, default=1,
                        help='choose strong data size for semi superveised')
    parser.add_argument('--aug', type=str, default='balance_std', help='DataAugmentation')
    parser.add_argument('--weakly_train', type=str, default='weakly strong',
                        help='weakly, strong means using ~ dataset, finetune means do not train with bbox')

    parser.add_argument('--crop_size', type=int, default=512, help='Choose crop size of RandomCrop.')
    parser.add_argument('--stride', type=int, default=32, help='the stride of CNN model.')
    parser.add_argument('--inbox255', type=int, default=1, help='mark pseudo mask inbox255')

    # Optimization options
    parser.add_argument('--optimizer', '-op', type=str, default='Adam', help='Optimizer to train model.')
    parser.add_argument('--scheduler', type=str, default='poly_epoch',
                        help='scheduler to train model, None means MultiSteps')

    parser.add_argument('--criterion', type=str, default='bce', help='criterion to train model.')
    parser.add_argument('--seg_criterion', type=str, default='l+ce', choices=['l+ce', 'fbce'],
                        help='criterion to train model.')

    parser.add_argument('--area_choose', type=float, default=1.0,
                        help='for weakly_smask_bbox.utils.BboxSmaskClusterRefine.')
    parser.add_argument('--imask_refine', type=str, default='normal', choices=['normal', 'exoverlap', 'seedmapmax'],
                        help='for weakly_smask_bbox.utils.BboxSmaskClusterRefine.')

    parser.add_argument('--loss_w', type=dict,
                        default={
                            'w_inst': 1,
                            'w_var': 10,
                            'w_seed': 1,
                        }, help='criterion to train model.')
    parser.add_argument('--foreground_weight', nargs='+', type=float, default=None, help='foreground_weight.')
    parser.add_argument('--peek_loss', type=str, default='orignal', help='peek loss for bbox, smask weakly training')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    # SpacialEmbedding default
    parser.add_argument('--decay', '-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')

    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--epoch_update', type=int, default=None, help='Reset or updae n_epoch.')
    parser.add_argument('--val_epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--test_batch_size', '-tb', type=int, default=1, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='The Learning Rate.')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='The Learning Rate of discriminator')
    parser.add_argument('--lr_update', type=float, default=None, help='The Learning Rate update.')
    parser.add_argument('--freeze', type=int, default=0, help='freeze batchnorm or not.')

    parser.add_argument('--cluster', type=str, default='for_each',
                        choices=['max', 'for_each', 'for_each_iter', 'gt',
                                 'density_for_each', 'density_max', 'density_gt',
                                 'multioffset_for_each'],
                        help='clustering configuration, utils.get_cluster_result')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold to clustering')

    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--resume_path', type=str, default=None, help='path to load checkpoints file.')
    parser.add_argument('--test_save', type=int, default=1, help='if or not to save eval result.')
    parser.add_argument('--eval_save', type=str, default=None, help='Folder to save eval result.')
    parser.add_argument('--test_mode', type=str, default='normal', choices=['anc', 'weakly', 'test'], help='test mode')
    parser.add_argument('--display', type=int, default=1, help='')
    parser.add_argument('--display_iter', type=int, default=1, help='')

    # Architecture
    parser.add_argument('--model_name', type=str, default='erfnet', help='backbone name')
    parser.add_argument('--n_class', type=int, default=8, help='classes number to classify.')
    parser.add_argument('--n_sigma', type=int, default=2, help='classes number to classify.')

    # Acceleration
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu or not')

    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')
    parser.add_argument('--std', type=int, default=0, help='standardize the data distribution.')

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--schedule', type=int, nargs='+', default=[25, 45],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--milestones', type=int, nargs='+', default=[25, 45],
                        help='multi_step_LR')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    args = parser.parse_args()

    args.optimizer = args.optimizer.lower()  # + f'_{args.scheduler}' if args.scheduler == 'poly' else ''
    args.model_name = args.model_name.lower()

    if args.foreground_weight is not None:
        assert len(args.foreground_weight) in (1, 8, 20)
        if len(args.foreground_weight) == 1:
            args.foreground_weight = args.foreground_weight * args.n_class
    else:
        if args.n_class > 8:
            args.foreground_weight = [100] * args.n_class
        else:
            if 'balance' in args.aug:
                args.foreground_weight = [3.07, 6.53, 2.68, 7.09, 6.98, 7.06, 7.12, 5.77]
            else:
                args.foreground_weight = [2.72, 7.92, 2.08, 9.61, 9.78, 10.12, 9.31, 6.33]
    if args.save is None:
        args.save = f'../{args.model_name}'
    if args.log is None:
        args.log = f'../{args.model_name}'

    if args.save is not None:
        create_dir(args.save)
    if args.eval_save is not None:
        create_dir(args.eval_save)
    else:
        args.eval_save = osp.join(args.save, 'eval/')
        create_dir(args.eval_save)
    args.resume_path = args.resume_path if args.resume_path is not None else args.save + '/checkpoint.pth'

    if args.scheduler is None:
        args.scheduler_name = f'{args.optimizer}_scheduler'
    else:
        args.scheduler_name = f'{args.optimizer}_{args.scheduler}_scheduler'

    # print args one by one ########
    for arg in vars(args):
        print(arg, getattr(args, arg))
    return args


def get_args():
    args = config()
    # cfg = copy.deepcopy(default_config)

    if args.aug == 'object_center':
        train_transform = my_transforms.get_transform([
            {
                'name': 'CropRandomObject',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'size': (args.crop_size, args.crop_size),
                }
            },
            {
                'name': 'ToTensor',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                }
            },
        ])
    else:
        train_transform = my_transforms.get_transform([
            {
                'name': 'RandomCrop',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'size': (args.crop_size, args.crop_size),
                }
            },
            {
                'name': 'ToTensor',
                'opts': {
                    'keys': ('image', 'instance', 'label'),
                    'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                }
            },
        ])
    cfg = dict(

        cuda=bool(args.gpu),
        display=bool(args.display),
        display_it=args.display_iter,

        save=True,
        eval_save=args.eval_save,
        test_save=True,
        save_dir=args.save,
        resume_path=args.resume_path,
        val_epochs=args.val_epochs,
        bkgd=args.n_class[1] == 9,

        train_dataset={
            'name': 'cityscapes',
            'kwargs': {
                'root_dir': args.cityscapes_data_path,
                # 'type': 'crops',
                'type': 'train',
                'size': 3000,
                'transform': train_transform
            },
            'batch_size': args.batch_size,
            'workers': args.prefetch
        },

        val_dataset={
            'name': 'cityscapes',
            'kwargs': {
                'root_dir': args.cityscapes_data_path,
                'type': 'val',
                'transform': my_transforms.get_transform([
                    {
                        'name': 'ToTensor',
                        'opts': {
                            'keys': ('image', 'instance', 'label'),
                            'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                        }
                    },
                ]),
            },
            'batch_size': args.test_batch_size,
            'workers': args.prefetch
        },

        model={
            'name': 'branched_erfnet',
            'kwargs': {
                'num_classes': args.n_class
            }
        },

        lr=5e-4,
        n_epochs=args.epochs,

        # loss options
        loss_opts={
            'to_center': False,
            'n_sigma': 2,
            'n_class': 8,
            'foreground_weight': 10,
            'strict': True if 'strict' in args.criterion else False,
            'mode': 'bce' if 'bce' in args.criterion else 'ce'
        },
        loss_w={
            'w_inst': 1,
            'w_var': 10,
            'w_seed': 1,
        },
    )

    return cfg


def get_config():
    return config()


if __name__ == '__main__':
    args = get_config()
    print(args)
    pass
