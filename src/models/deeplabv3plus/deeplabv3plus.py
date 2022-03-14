# ----------------------------------------
# Written by Linwei Chen
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d
from torch.nn import init
from models.deeplabv3plus.backbone import build_backbone
from models.deeplabv3plus.ASPP import ASPP


# from models import ISEG_MODEL


# @MODEL.register()
class deeplabv3plus(nn.Module):
    def __init__(self, cfg):
        super(deeplabv3plus, self).__init__()
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
        self.offset_conv = Decoder(cfg.MODEL_AUX_OUT)
        self.seed_map_conv = Decoder(cfg.MODEL_NUM_CLASSES)

        self.bbox_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, indim, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.bbox_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 1, input_channel, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )
        # self.cls_conv = nn.Conv2d(cfg.MODEL_ASPP_OUTDIM, cfg.MODEL_NUM_CLASSES, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.backbone = build_backbone(cfg.MODEL_BACKBONE, os=cfg.MODEL_OUTPUT_STRIDE)
        self.backbone_layers = self.backbone.get_layers()
        self.init_output()
        self.apply(self.init_bn)

    def init_output(self, n_sigma=2):
        with torch.no_grad():
            output_conv = self.offset_conv.output_conv
            print('initialize last layer with size: ', output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(1)
            if self.cfg.MODEL_AUX_OUT == 6:
                output_conv.weight[:, 2 + n_sigma:4 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 + n_sigma:4 + n_sigma].fill_(0)
                pass
            elif self.cfg.MODEL_AUX_OUT == 5:
                output_conv.weight[:, 2 + n_sigma:3 + n_sigma, :, :].fill_(0)
                output_conv.bias[2 + n_sigma:3 + n_sigma].fill_(0)
                pass

    @staticmethod
    def init_bn(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            m.eps = 0.001
            m.momentum = 0.1
            print(f'*** init with m.eps:{m.eps} m.momentum:{m.momentum}')

    @staticmethod
    def init_bn_for_seg(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            # eps=1e-05, momentum=0.0003
            m.eps = 1e-05
            m.momentum = 0.0003
            print(f'*** init with m.eps:{m.eps} m.momentum:{m.momentum}')

    def apply_init_bn(self):
        self.apply(self.init_bn)

    def apply_init_bn_for_seg(self):
        self.apply(self.init_bn_for_seg)

    def forward(self, x, bbox=None):
        _, _, ih, iw = x.shape
        x = F.interpolate(x, size=(ih - ih % 16, iw - iw % 16), mode='bilinear', align_corners=True)

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        _, _, h0, w0 = layers[0].shape
        _, _, h2, w2 = layers[2].shape
        b, _, _, _ = x.shape

        if bbox is not None:
            bbox_1 = F.interpolate(bbox, size=(h0, w0))
            bbox_2 = F.interpolate(bbox, size=(h2, w2))

            bbox_attention_1 = self.bbox_attention1(bbox_1)
            bbox_attention_2 = self.bbox_attention2(bbox_2)
            layers[0] = layers[0] * bbox_attention_1
            layers[2] = layers[2] * bbox_attention_2

        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        feature = self.cat_conv(feature_cat)

        offset = self.offset_conv(feature)
        seed_map = self.seed_map_conv(feature)

        offset = F.interpolate(offset, size=(ih, iw), mode='bilinear', align_corners=True)
        # y, x, sigma
        offset[:, 0] = offset[:, 0] * float(ih / (ih - ih % 16))
        offset[:, 1] = offset[:, 1] * float(iw / (iw - iw % 16))
        seed_map = F.interpolate(seed_map, size=(ih, iw), mode='bilinear', align_corners=True)

        return torch.cat([offset, seed_map], dim=1)



def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = SynchronizedBatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class BasicBlock(nn.Module):
    expansion = 1
    bn_mom = 0.0003

    def __init__(self, inplanes, planes, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, atrous)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = SynchronizedBatchNorm2d(planes, momentum=self.bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = SynchronizedBatchNorm2d(planes, momentum=self.bn_mom)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(256, 128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))
        self.output_conv = nn.ConvTranspose2d(
            128, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


class DecoderThick(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(256, 128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))
        self.layers.append(BasicBlock(inplanes=128, planes=128))
        self.output_conv = nn.ConvTranspose2d(
            128, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


if __name__ == '__main__':
    from models.deeplabv3plus import Configuration

    cfg = Configuration()
    model = deeplabv3plus(cfg)
    print(model)
    print(model.__class__.__name__)
    model.eval()
    # model.seg = True
    x = torch.randn(2, 3, 65, 65)
    y = model(x)
    for item in y:
        print(item.shape)

