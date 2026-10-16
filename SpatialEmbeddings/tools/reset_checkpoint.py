import torch

load_path = '/Users/chenlinwei/code/SpatialEmbeddings/std_weakly/checkpoint.pth'
state_dict = torch.load(load_path, map_location='cpu')
state_dict['best_iou'] = 0
torch.save(state_dict, load_path)
