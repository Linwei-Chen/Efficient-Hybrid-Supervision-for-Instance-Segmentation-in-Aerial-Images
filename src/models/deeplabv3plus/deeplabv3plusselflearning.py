# ----------------------------------------
# Written by charles
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from models.deeplabv3plus.backbone import build_backbone
from models.deeplabv3plus.ASPP import ASPP


class ModelBase(nn.Module):
    """
    normal: do not use the aucillary info like bbox, label
    info: use the ancillary info
    refine-learning: refine the output by label or bbox first and then softmax, as for bbox, bkgd should set as zero first
    refine: output the reuslt refined by label or bbox
    self-learning: softmax first and then refine the ouput by label or bbox

    """

    def __init__(self):
        super().__init__()
        self.mode = 'normal'

    def normal(self):
        self.mode = 'normal'
        # self.train()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def wBoxAttBoxFltr(self):
        self.mode = 'BoxAtt+BoxFltr'
        # self.train()
        print(f'---> mode:{self.mode} | trianing:{self.training}')

    def wBoxAtt(self):
        self.mode = 'BoxAtt'
        # self.eval()
        print(f'---> mode:{self.mode} | trianing:{self.training}')


class deeplabv3plus(ModelBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = None
        self.backbone_layers = None
        input_channel = 2048
        self.aspp = ASPP(dim_in=input_channel,
                         dim_out=cfg.MODEL_ASPP_OUTDIM,
                         rate=16 // cfg.MODEL_OUTPUT_STRIDE,
                         bn_mom=cfg.TRAIN_BN_MOM)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.MODEL_OUTPUT_STRIDE // 4)

        indim = 256
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_SHORTCUT_KERNEL, 1,
                      padding=cfg.MODEL_SHORTCUT_KERNEL // 2, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_SHORTCUT_DIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM + cfg.MODEL_SHORTCUT_DIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1,
                      bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_ASPP_OUTDIM, 3, 1, padding=1, bias=True),
            SynchronizedBatchNorm2d(cfg.MODEL_ASPP_OUTDIM, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.bbox_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES, input_channel, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.bbox_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES, indim, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.mode = 'normal'

    def forward(self, x, bbox_target_one_hot=None):
        _, _, ih, iw = x.shape
        x = F.interpolate(x, size=(ih - ih % 16, iw - iw % 16), mode='bilinear', align_corners=True)

        if bbox_target_one_hot is None:
            bbox_target_one_hot = torch.zeros([x.size(0), self.cfg.MODEL_NUM_CLASSES, x.size(2), x.size(3)]).to(
                x.device)
        elif not isinstance(bbox_target_one_hot, torch.FloatTensor):
            bbox_target_one_hot = bbox_target_one_hot.float()

        if self.mode == 'normal':
            bbox_target_one_hot = torch.zeros(bbox_target_one_hot.shape).to(bbox_target_one_hot.device)

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()

        bbox_size1 = F.interpolate(bbox_target_one_hot, size=layers[-1].shape[2:], mode='nearest')
        bbox_size2 = F.interpolate(bbox_target_one_hot, size=layers[0].shape[2:], mode='nearest')

        bbox_attention1 = self.bbox_attention1(bbox_size1)
        bbox_attention2 = self.bbox_attention2(bbox_size2)
        layers[-1] = bbox_attention1 * layers[-1]
        layers[0] = bbox_attention2 * layers[0]

        # for l in layers:
        #     print(l.shape)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        # print(feature_aspp.shape)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)

        result = F.interpolate(result, size=(ih, iw), mode='bilinear', align_corners=True)

        # 78.91 vs 79.56
        #     probs = result.softmax(dim=1)
        #     bbox_target_one_hot[:, 0] = 1.
        #     probs = bbox_target_one_hot * probs

        # 79.18 vs 79.56
        if self.mode == 'BoxAtt+BoxFltr':
            bbox_target_one_hot[:, 0] = 1.
            result = bbox_target_one_hot * result
            probs = result.softmax(dim=1)
            return probs
        elif self.mode in ('normal', 'BoxAtt'):
            return result

        else:
            raise NotImplementedError


if __name__ == '__main__':
    from models.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plus(cfg)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)
    print(model)
