import os.path as osp
import os
import sys

sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
sys.path.append(osp.join(sys.path[0], '../../../'))
from eval_cityscapes import clear_txt, parse_txt
import shutil
import torchvision
import argparse
from train_config import get_config, create_dir
from tqdm import tqdm
import torch
from train_utils import get_device
from datasets import get_cityscapes_dataloader, get_test_dataloader
from models import get_model, get_weakly_model
from tools.utils import VisualizeTool, get_cluster, get_cluster_result
from models import get_model
from train_utils import load, model_accelerate
from torch import nn
from time import sleep
from logger import ModelSaver
from PIL import Image
import numpy as np
from torchvision import transforms


def config():
    parser = argparse.ArgumentParser(description='Trains FCN on SBD and VOC',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--test_data_set', type=str, default='cityscapes', choices=['cityscapes', 'voc2012'],
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
                        choices=['normal', 'anc', 'weakly', 'test', 'seg', 'anc_seg',
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

    parser.add_argument('--std', type=int, nargs='+', default=0, help='standardize the data distribution.')

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


def test(args):
    from eval_cityscapes import from_seg_pred_to_one_hot
    device = get_device(args)
    cluster_func = get_cluster(args)
    visualizetool = VisualizeTool(args)

    print(f'*** test mode : {args.test_mode}')
    if args.test_mode == 'test':
        data_loader = get_test_dataloader(args)
    else:
        data_loader = get_cityscapes_dataloader(args, train=False)

    if args.test_mode in ('anc', 'weakly', 'recurrent'):
        model = get_weakly_model(args)
    elif args.test_mode in ('normal', 'test'):
        model = get_model(args)
    elif args.test_mode in ('seg', 'anc_seg'):
        model = get_weakly_model(args)
        model.seg = True
    else:
        raise NotImplementedError

    model = model_accelerate(args, model)
    _, _, model, _, _ = load(args, model)

    # seg model - 19 class
    seg_model = get_model(args, name='deeplabv3plusseg', n_class=19)
    seg_model = model_accelerate(args, seg_model)
    if not isinstance(seg_model, nn.DataParallel):
        seg_model = nn.DataParallel(seg_model)
    # seg_model.load_state_dict(torch.load('../deeplabv3plusxception_miou_0.7840_pa_0.9608.pkl', map_location='cpu'))
    ModelSaver.load_tool(load_path='../deeplabv3plusxception_miou_0.7840_pa_0.9608.pkl', model=seg_model)

    model.eval()
    seg_model.eval()

    data_loader = tqdm(data_loader)
    with torch.no_grad():
        for sample in data_loader:

            im = sample['image'].to(device)
            instance = sample['instance'].to(device)
            class_label = sample['label'].to(device)

            # check
            batch_size = im.size(0)
            if args.test_save > 0:
                assert batch_size == 1

            if args.test_mode == 'anc':
                bbox = sample['bbox'].to(device)
                output = model(im, bbox)
            elif args.test_mode in ('normal', 'test', 'weakly'):
                output = model(im)
            elif args.test_mode == 'recurrent':
                '''
                seg = seg_model(im)
                score, seg_max = seg.max(dim=1, keepdim=True)
                seg_one_hot = from_seg_pred_to_one_hot(seg)
                for i in range(10):
                    seg_one_hot[:, 10][seg_one_hot[:, i] > 0] = 1
                output = model(im, bbox=seg_one_hot[:, 10:])
                '''
                anc_seg = seg_model(im).detach()
                seg_input = anc_seg.softmax(dim=1)
                for i in range(10):
                    seg_input[:, 10] += seg_input[:, i]
                seg_input = seg_input[:, 10:]
                output = model(im, bbox=seg_input)
            elif 'seg' in args.test_mode:
                if 'anc' in args.test_mode:
                    bbox = sample['bbox'].to(device)
                    output, seg = model(im, bbox)
                else:
                    output, _ = model(im)
                    seg = seg_model(im)
                score, seg_max = seg.max(dim=1, keepdim=True)
                seg_one_hot = from_seg_pred_to_one_hot(seg)
                # seg_one_hot = seg > 0.4
                # output[:, 4:] = output[:, 4:] * (seg[:, 0:1] - 1).abs()
                output[:, 4:] = output[:, 4:] * seg_one_hot[:, -8:].float()
                pred_label = seg_max.squeeze(dim=0)
            else:
                raise NotImplementedError

            assert args.cluster != 'gt'

            instance_map, predictions, instances_cls = get_cluster_result(args, cluster_func, output[0], instance[0])
            visualizetool.visualize(image=im[0], label=class_label[0],
                                    instance=instance[0],
                                    prediction=output[0].detach(), instance_pred=instance_map)
            data_loader.write(args.save)
            if args.test_save > 0:

                base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

                txt_file = os.path.join(args.eval_save, base + '.txt')
                with open(txt_file, 'w') as f:

                    # loop over instances
                    for id, pred in enumerate(predictions):
                        im_name = base + '_{:02d}.png'.format(id)
                        im = torchvision.transforms.ToPILImage()(
                            pred['mask'].unsqueeze(0))

                        # write image
                        im.save(os.path.join(args.eval_save, im_name))

                        # write to file
                        cl = pred['cls']
                        score = pred['score']
                        f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
    del visualizetool, data_loader
    torch.cuda.empty_cache()


def run_auto_test():
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
        perfomance_path = osp.join(args.save, f'AP_{ap}_AP50_{ap50}_std{std}.pth')
        shutil.copyfile(args.resume_path, perfomance_path)
    sleep(1)
    pass


def _load_test():
    args = config()
    seg_model = get_model(args, name='deeplabv3plus3branch', n_class=18)
    seg_model = model_accelerate(args, seg_model)
    if not isinstance(seg_model, nn.DataParallel):
        seg_model = nn.DataParallel(seg_model)
    # seg_model.load_state_dict(torch.load('../deeplabv3plusxception_miou_0.7840_pa_0.9608.pkl', map_location='cpu'))
    ModelSaver.load_tool(load_path='../deeplabv3plusxception_miou_0.7840_pa_0.9608.pkl', model=seg_model, strict=False)
    img = Image.open('/Users/chenlinwei/Desktop/city.png').resize((512, 256)).convert('RGB')
    img_np = np.array(img).transpose(2, 0, 1)
    x = torch.from_numpy(img_np).float()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    x = normalize(x).unsqueeze(dim=0)
    seg_model.eval()
    seg_model.module.seg = True
    _, y = seg_model(x)
    _, res = y.max(dim=1)

    transforms.ToPILImage()(res.squeeze().float() * 15).show()
    pass


if __name__ == '__main__':
    # _load_test()
    run_auto_test()
    pass
