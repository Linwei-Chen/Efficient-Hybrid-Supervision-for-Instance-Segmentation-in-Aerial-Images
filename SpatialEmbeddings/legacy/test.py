"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from train_config import get_args
import torch
from models import get_model
from tools.utils import Cluster, Visualizer
from torchvision import transforms

torch.backends.cudnn.benchmark = True

args = get_args()

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

if args['save']:
    if not os.path.exists(args['eval_save']):
        os.makedirs(args['eval_save'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['val_dataset']['name'], args['val_dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

# load snapshot
if os.path.exists(args['resume_path']):
    try:
        state = torch.load(args['resume_path'], map_location='cpu')
        model.load_state_dict(state['model_state_dict'], strict=True)
    except Exception:
        print('load {} fail'.format(args['resume_path']))
        # assert(False, 'checkpoint_path {} does not exist!'.format(args['checkpoint_path']))

model.eval()

# cluster module
cluster = Cluster(bkgd=args['bkgd'])

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed', 'cls'))

with torch.no_grad():
    for sample in tqdm(dataset_it):

        im = sample['image'].to(device)
        instances = sample['instance'].squeeze().to(device)

        output = model(im)
        # cls_map = transforms.ToPILImage(output[0].argmax(dim=0)*20)

        seed_map = output[0][2 + args['loss_opts']['n_sigma']:].detach()
        prob, cls = seed_map.max(dim=0)
        cls = cls + 1
        cls[prob < 0.5] = 0
        print('cls:', cls.unique())
        cls = cls.float() / 10
        cls = seed_map[2]

        instance_map, predictions, instances_cls = cluster.cluster_for_each(output[0],
                                                                            n_sigma=args['loss_opts']['n_sigma'],
                                                                            threshold=0.5)
        print(f'instance_map:{instance_map.shape}')
        # print(f'instance_map:{instance_map.unique()}')
        # print('predictions:', predictions[0]['mask'].unique())

        if args['display']:
            visualizer.display(im.cpu(), 'image')

            visualizer.display([instance_map.cpu(), instances.cpu()], 'pred')
            visualizer.display(cls.cpu(), 'cls')

            sigma = output[0][2].cpu()
            sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
            sigma[instances == 0] = 0
            # visualizer.display(sigma, 'sigma')

            seed = torch.sigmoid(output[0][3]).cpu()
            # visualizer.display(seed, 'seed')

        if args['test_save']:

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

            txt_file = os.path.join(args['eval_save'], base + '.txt')
            with open(txt_file, 'w') as f:

                # loop over instances
                for id, pred in enumerate(predictions):
                    im_name = base + '_{:02d}.png'.format(id)
                    im = torchvision.transforms.ToPILImage()(
                        pred['mask'].unsqueeze(0))

                    # write image
                    im.save(os.path.join(args['eval_save'], im_name))

                    # write to file
                    cl = pred['cls']
                    score = pred['score']
                    f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))
