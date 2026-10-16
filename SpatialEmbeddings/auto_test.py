"""
test the modelone by one.
"""

import os
import time
import argparse
from train_config import get_config, create_dir
import os.path as osp
from eval_cityscapes import clear_txt, parse_txt
import shutil


def config():
    parser = argparse.ArgumentParser(description='Trains FCN on SBD and VOC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--test_data_set', type=str, default='cityscapes', choices= ['cityscapes', 'voc2012'],
                        help='Root for the cityscape dataset.')

    parser.add_argument('--cityscapes_data_path', type=str, default='/Users/chenlinwei/dataset/cityscapes',
                        help='Root for the cityscape dataset.')
    parser.add_argument('--voc2012_data_path', type=str, default='/Users/chenlinwei/dataset/VOCdevkit/VOC2012',
                        help='Root for the voc dataset.')
    parser.add_argument('--sbd_data_path', type=str, default='/Users/chenlinwei/dataset/SBD_FULL11355/dataset',
                        help='Root for the voc dataset.')
    parser.add_argument('--data_choose_size', type=int, default=None,
                        help='choose strong data size for semi superveised')
    parser.add_argument('--list', nargs='+', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--test_save', type=int, default=1, help='if or not to save eval result.')
    parser.add_argument('--eval_save', type=str, default=None, help='Folder to save eval result.')
    parser.add_argument('--performance_save', type=str, default='../model_AP', help='Folder to save eval result.')
    # if 'bbox' in aug, it means usingweakly model
    parser.add_argument('--aug', type=str, default='', help='Folder to save eval result.')
    parser.add_argument('--test_mode', type=str, default='normal',
                        choices=['normal','anc', 'weakly', 'test', 'seg', 'anc_seg',
                                 'multi_scale', 'flip', 'multi_scale_flip',
                                 'recurrent'],
                        help='test mode')

    parser.add_argument('--display', type=int, default=1, help='display or not')
    parser.add_argument('--display_iter', type=int, default=1, help='')
    parser.add_argument('--epoch_update', type=int, default=None, help='Reset or updae n_epoch.')

    parser.add_argument('--model_name', type=str, default='erfnet', help='backbone name')
    parser.add_argument('--n_class', type=int, default=8, help='classes number to classify.')
    parser.add_argument('--n_sigma', type=int, default=2, help='classes number to classify.')

    # Acceleration
    parser.add_argument('--prefetch', type=int, default=0, help='Pre-fetching threads.')

    parser.add_argument('--std', type=int,nargs='+', default=0, help='standardize the data distribution.')

    parser.add_argument('--cluster', type=str, default='for_each',
                        choices=['max', 'for_each', 'for_each_iter', 'gt',
                                 'density_for_each', 'density_max', 'density_gt',
                                 'multioffset_for_each', 'multioffset_for_each_iter'],
                        help='clustering configuration')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold to clustering')
    args = parser.parse_args()

    create_dir(args.performance_save)
    args.eval_save = osp.join(args.performance_save, 'eval')
    create_dir(args.eval_save)

    args.gpu = 1
    args.test_save = 1
    args.test_batch_size = 1

    print(args)
    return args


def auto_eval():
    from eval_cityscapes import test
    from time import sleep

    args = config()
    aug = args.aug
    print(args.list)

    for std, save in zip(args.std, args.list):
        # std or not
        if std > 0 and 'std' not in aug:
            args.aug = aug + '_std'
        else:
            args.aug = aug

        if '.pth' not in save:
            args.resume_path = osp.join(save, 'checkpoint.pth')
            args.save = save
        else:
            args.resume_path = save
            args.save = osp.split(save)[0]
        print(args)
        if args.test_data_set == 'cityscapes':
            assert args.n_class in (8, 9)
            if osp.exists(args.resume_path):
                # create mask image in ./performance/eval
                test(args)
            else:
                print(f'*** {args.resume_path} not exist! continue to next one.')
                continue

            # run citiyscapes script
            CITYSCAPES_DATASET = os.path.abspath(args.cityscapes_data_path)
            CITYSCAPES_RESULTS = os.path.abspath(args.eval_save)
            os.system(f'CITYSCAPES_DATASET={CITYSCAPES_DATASET} CITYSCAPES_RESULTS={CITYSCAPES_RESULTS} '
                      f'python ../cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py | tee temp.txt | tee -a res.txt ')

            clear_txt()
            ap, ap50 = parse_txt()
            perfomance_path = osp.join(args.save, f'AP_{ap:.4f}_AP50_{ap50:.4f}_std{std}.pth')
            shutil.copyfile(args.resume_path, perfomance_path)
        elif args.test_data_set == 'voc2012':
            assert args.n_class in (20, 21)
            from eval_voc import eval_voc
            mAP50, mAP75 = eval_voc(args)
            perfomance_path = osp.join(args.save, f'mAP_{mAP50:.4f}_mAP75_{mAP75:.4f}_std{std}.pth')
            shutil.copyfile(args.resume_path, perfomance_path)
        else:
            raise NotImplementedError

    sleep(1)

if __name__ == '__main__':
    auto_eval()
    # os.system(f'CITYSCAPES_DATASET=/Users/chenlinwei/dataset/cityscapes '
    #           f'CITYSCAPES_RESULTS=/Users/chenlinwei/code/SpatialEmbeddings/eval '
    #           f'python ../cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py | tee temp.txt | tee -a res.txt ')

    pass
