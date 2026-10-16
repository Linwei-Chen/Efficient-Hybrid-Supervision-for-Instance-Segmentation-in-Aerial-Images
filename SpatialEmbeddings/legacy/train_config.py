"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os

import torch
from datasets import transforms as my_transforms
import argparse
import os.path as osp

CITYSCAPES_DIR = os.environ.get('CITYSCAPES_DIR')


def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.mkdir(dir_path)


def config():
    parser = argparse.ArgumentParser(description='Trains FCN on SBD and VOC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--cityscapes_data_path', type=str, default='/Users/chenlinwei/dataset/cityscapes',
                        help='Root for the cityscape dataset.')
    parser.add_argument('--aug', type=str, default='object_center', help='DataAugmentation')

    parser.add_argument('--crop_size', type=int, default=320, help='Choose crop size of RandomCrop.')
    parser.add_argument('--stride', type=int, default=32, help='the stride of CNN model.')

    # Optimization options
    parser.add_argument('--optimizer', '-op', type=str, default='Adam', help='Optimizer to train model.')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler to train model, None means MultiSteps')
    parser.add_argument('--criterion', type=str, default='bce', help='criterion to train model.')

    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--val_epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--test_batch_size', '-tb', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0005, help='The Learning Rate.')

    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=2e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_bs', type=int, default=10)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--schedule', type=int, nargs='+', default=[25, 45],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--eval_save', type=str, default=None, help='Folder to save eval result.')

    parser.add_argument('--save_steps', '-ss', type=int, default=50, help='steps to save checkpoints.')
    parser.add_argument('--display', type=int, default=1, help='')
    parser.add_argument('--display_iter', type=int, default=5, help='')

    # Architecture
    parser.add_argument('--model_name', type=str, default='erfnet', help='')
    parser.add_argument('--n_class', nargs='+', type=int, default=[4, 8], help='classes number to classify.')
    parser.add_argument('--n_sigma', type=int, default=2, help='classes number to classify.')

    # Acceleration
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')
    parser.add_argument('--gpu', type=int, default=0, help='use gpu or not')

    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')
    parser.add_argument('--std', type=int, default=2, help='standardize the data distribution.')
    args = parser.parse_args()

    args.optimizer = args.optimizer.lower()  # + f'_{args.scheduler}' if args.scheduler == 'poly' else ''
    args.model_name = args.model_name.lower()

    if args.save is None:
        args.save = f'../{args.model_name}'
    if args.log is None:
        args.log = f'../{args.model_name}'

    if args.save is not None:
        create_dir(args.save)
    if args.eval_save is not None:
        create_dir(args.eval_save)
    args.resume_path = args.resume_path if args.resume_path is not None else args.save + '/checkpoint.pth'

    if args.scheduler is None:
        args.scheduler_name = f'{args.optimizer}_scheduler'
    else:
        args.scheduler_name = f'{args.optimizer}_{args.scheduler}_scheduler'
    print(args)
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
        bkgd=args.n_class[1] == 9,

        save=True,
        eval_save=args.eval_save,
        test_save=True,
        save_dir=args.save,
        resume_path=args.resume_path,
        val_epochs=args.val_epochs,
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
            'n_sigma': args.n_sigma,
            'n_class': args.n_class[1],
            'foreground_weight': 10,
        },
        loss_w={
            'w_inst': 1,
            'w_var': 10,
            'w_seed': 1,
        },
    )

    return cfg


if __name__ == '__main__':
    args = get_args()
    print(args)
    pass
