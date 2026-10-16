"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import shutil
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from train_config import get_args
from criterions.multi_cls_my_loss import SpatialEmbLoss
from datasets import get_dataset
from models import get_model
from tools.utils import AverageMeter, Cluster, Logger, Visualizer
from eval_cityscapes import test
# torch.backends.cudnn.benchmark = True

args = get_args()

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

# set device
device = torch.device("cuda" if args['cuda'] else "cpu")

# train dataloader
train_dataset = get_dataset(
    args['train_dataset']['name'], args['train_dataset']['kwargs'])
train_dataset_it = torch.utils.data.DataLoader(
    train_dataset, batch_size=args['train_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# val dataloader
val_dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
val_dataset_it = torch.utils.data.DataLoader(
    val_dataset, batch_size=args['val_dataset']['batch_size'], shuffle=True, drop_last=True,
    num_workers=args['train_dataset']['workers'], pin_memory=True if args['cuda'] else False)

# set model
model = get_model(args['model']['name'], args['model']['kwargs'])
model.init_output(args['loss_opts']['n_sigma'])
model = torch.nn.DataParallel(model).to(device)

# set criterion
criterion = SpatialEmbLoss(**args['loss_opts'])
criterion = torch.nn.DataParallel(criterion).to(device)

# set optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=args['lr'], weight_decay=1e-4)


def lambda_(epoch):
    return pow((1 - ((epoch) / args['n_epochs'])), 0.9)


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda_, )

# clustering
cluster = Cluster(bkgd=True)

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed', 'cls', 'label'))

# Logger
logger = Logger(('train', 'val', 'iou', 'ap', 'ap50'), 'loss')

# resume
start_epoch = 0
best_ap = 0
if args['resume_path'] is not None and os.path.exists(args['resume_path']):
    print('Resuming model from {}'.format(args['resume_path']))
    try:
        state = torch.load(args['resume_path'], map_location='cpu')
        start_epoch = state['epoch'] + 1
        best_ap = state['best_ap']
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optim_state_dict'])
        logger.data = state['logger_data']
    except Exception:
        print('load {} fail!'.format(args['resume_path']))


def train(epoch):
    # define meters
    loss_meter = AverageMeter()

    # put model into training mode
    model.train()

    for param_group in optimizer.param_groups:
        print('learning rate: {}'.format(param_group['lr']))

    for i, sample in enumerate(tqdm(train_dataset_it)):

        im = sample['image']
        instances = sample['instance'].squeeze()
        class_labels = sample['label'].squeeze()

        output = model(im)

        loss = criterion(output, instances, class_labels, **args['loss_w'])
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        seed_map = output[0][2 + args['loss_opts']['n_sigma']:].detach()
        prob, cls = seed_map.max(dim=0)
        print('cls:', cls.unique().data)
        prob[cls == 0] = 0
        cls = (cls.float()) / 10

        if args['display'] and i % args['display_it'] == 0:
            with torch.no_grad():

                # show img
                visualizer.display(im[0], 'image')

                # show label
                visualizer.display([cls.cpu(), prob.cpu(), class_labels[0].cpu()], 'label')

                # show cls map
                # visualizer.display(cls.cpu(), 'cls')

                # show prediction
                # predictions = cluster.cluster_with_gt(output[0], instances[0], n_sigma=args['loss_opts']['n_sigma'])
                # visualizer.display([predictions.cpu(), instances[0].cpu()], 'pred')

                instance_map, predictions, instances_cls = cluster.cluster(output[0],
                                                                           n_sigma=args['loss_opts']['n_sigma'],
                                                                           threshold=0.9)

                visualizer.display([instance_map.cpu(), instances[0].cpu()], 'pred')

                # show sigma refine by GT
                sigma = output[0][2:4].mean(dim=0).cpu()
                sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                sigma[instances[0] == 0] = 0
                # visualizer.display(sigma, 'sigma')

                # show seed
                seed = torch.sigmoid(output[0][3]).cpu()
                # visualizer.display(seed, 'seed')

        print(f'loss:{loss.item()}')
        loss_meter.update(loss.item())

    return loss_meter.avg


def val(epoch):
    # define meters
    loss_meter, iou_meter = AverageMeter(), AverageMeter()

    # put model into eval mode
    model.eval()

    with torch.no_grad():

        for i, sample in enumerate(tqdm(val_dataset_it)):

            im = sample['image']
            instances = sample['instance'].squeeze()
            class_labels = sample['label'].squeeze()

            output = model(im)
            loss = criterion(output, instances, class_labels, **
            args['loss_w'], iou=True, iou_meter=iou_meter)
            loss = loss.mean()

            seed_map = output[0][2 + args['loss_opts']['n_sigma']:].detach()
            prob, cls = seed_map.max(dim=0)
            print('cls:', cls.unique().data)
            prob[cls == 0] = 0
            cls = (cls.float()) / 10

            if args['display'] and i % args['display_it'] == 0:
                with torch.no_grad():
                    visualizer.display(im[0], 'image')

                    # predictions = cluster.cluster_with_gt(output[0], instances[0],
                    #                                       n_sigma=args['loss_opts']['n_sigma'])

                    instance_map, predictions, instances_cls = cluster.cluster(output[0],
                                                                               n_sigma=args['loss_opts']['n_sigma'],
                                                                               threshold=0.9)
                    visualizer.display([instance_map.cpu(), instances[0].cpu()], 'pred')

                    # sigma = output[0][2].cpu()
                    # sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
                    # sigma[instances[0] == 0] = 0
                    # visualizer.display(sigma, 'sigma')

                    # seed = torch.sigmoid(output[0][3]).cpu()
                    # visualizer.display(seed, 'seed')
            loss_meter.update(loss.item())

    return loss_meter.avg, iou_meter.avg


def save_checkpoint(state, is_best, name='checkpoint.pth'):
    print('=> saving checkpoint')
    file_name = os.path.join(args['save_dir'], name)
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, os.path.join(
            args['save_dir'], 'best_ap_model.pth'))


for epoch in range(start_epoch, args['n_epochs']):

    print('Starting epoch {}'.format(epoch))
    scheduler.step(epoch)

    # train_loss = train(epoch)
    # val_loss, ap = val(epoch)
    if epoch % args['val_epochs'] == 0:
        ap, ap50 = test(args, model)
        # print('===> train loss: {:.2f}'.format(train_loss))
        print('===> ap: {:.2f}, ap50: {:.2f}'.format(ap, ap50))

        # logger.add('train', train_loss)
        logger.add('ap', ap)
        logger.add('ap50', ap)
        # logger.plot(save=args['save'], save_dir=args['save_dir'])

        is_best = ap > best_ap
        best_ap = max(ap, best_ap)

        if args['save']:
            state = {
                'epoch': epoch,
                'best_ap': best_ap,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'logger_data': logger.data
            }
            save_checkpoint(state, is_best)
