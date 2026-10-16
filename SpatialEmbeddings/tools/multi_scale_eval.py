"""
Author: Charles Chen
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import collections
import os
import threading

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tools.utils_for_density import DensityCluster
from tools.utils_for_multi_offset import ClusterForMultiOffset

import torch
import torch.nn.functional as F
import math


def modify_out(out: torch.Tensor):
    if out.size(-3) in (4 + 8, 4 + 20):
        pass
    else:
        raise NotImplementedError


def scale_with_strde(size: tuple, scale: float, stride=16):
    assert len(size) == 2
    size = [int(math.ceil(i * scale / stride) * stride) for i in size]
    return size


def multi_scale_eval(img, model, multi_scale=False, flip=False):
    """
    img, model should be on same device
    :return:
    """
    scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()
        N, c, h, w = img.shape
        assert N == 1
        out = model(img)
        outs = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        counter = 1
        if multi_scale:
            for scale in scales:
                counter += 1
                img_temp = F.interpolate(img, size=scale_with_strde((h, w), scale), mode='bilinear',
                                         align_corners=True)
                # print(f'scale img.shape:{img.shape}')
                out = F.interpolate(model(img_temp), size=(h, w), mode='bilinear',align_corners=True)
                # del img_temp
                # print(f'flip img.shape:{out.shape}')
                outs += out
                # del out
            pass
        if flip:
            counter += 1
            img_temp = torch.flip(img, dims=[3])
            # print(f'flip img.shape:{img.shape}')
            out = F.interpolate(torch.flip(model(img_temp), dims=[3]), size=(h, w), mode='bilinear', align_corners=True)
            # del img_temp
            # print(f'flip img.shape:{out.shape}')
            outs += out
            # del out
            pass
    return outs / float(counter)
