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


class deeplabv3pluswithbboxandseg(nn.Module):
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
        ########## 20200330 modified
        # +2 means background and 255 ignore
        self.seg_attention1 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 2, indim, 3, 1, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.seg_attention2 = nn.Sequential(
            nn.Conv2d(cfg.MODEL_NUM_CLASSES + 2, input_channel, 3, 1, padding=1, bias=True),
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
            # print(f'*** init with m.eps:{m.eps} m.momentum:{m.momentum}')

    @staticmethod
    def init_bn_for_seg(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            # eps=1e-05, momentum=0.0003
            m.eps = 1e-05
            m.momentum = 0.0003
            # print(f'*** init with m.eps:{m.eps} m.momentum:{m.momentum}')

    def apply_init_bn(self):
        self.apply(self.init_bn)

    def apply_init_bn_for_seg(self):
        self.apply(self.init_bn_for_seg)

    def forward(self, x, bbox=None, seg=None):
        _, _, ih, iw = x.shape
        x = F.interpolate(x, size=(ih - ih % 16, iw - iw % 16), mode='bilinear', align_corners=True)

        x_bottom = self.backbone(x)
        layers = self.backbone.get_layers()
        _, _, h0, w0 = layers[0].shape
        _, _, h2, w2 = layers[2].shape
        b, _, _, _ = x.shape
        #### for args.preprocessing == 'softmax'
        if seg is not None and seg.size(1) == self.cfg.MODEL_NUM_CLASSES + 1:
            zero_temp = torch.zeros(b, 1, ih, iw).to(seg.device)
            seg = torch.cat([seg, zero_temp], dim=1)

        if bbox is not None and seg is not None:
            bbox_1 = F.interpolate(bbox, size=(h0, w0))
            bbox_2 = F.interpolate(bbox, size=(h2, w2))
            bbox_attention_1 = self.bbox_attention1(bbox_1)
            bbox_attention_2 = self.bbox_attention2(bbox_2)

            seg_1 = F.interpolate(seg, size=(h0, w0))
            seg_2 = F.interpolate(seg, size=(h2, w2))

            seg_attention_1 = self.seg_attention1(seg_1)
            seg_attention_2 = self.seg_attention2(seg_2)

            layers[0] = layers[0] * (bbox_attention_1 + seg_attention_1)
            layers[2] = layers[2] * (bbox_attention_2 + seg_attention_2)
        elif bbox is not None:
            bbox_1 = F.interpolate(bbox, size=(h0, w0))
            bbox_2 = F.interpolate(bbox, size=(h2, w2))

            bbox_attention_1 = self.bbox_attention1(bbox_1)
            bbox_attention_2 = self.bbox_attention2(bbox_2)
            layers[0] = layers[0] * bbox_attention_1
            layers[2] = layers[2] * bbox_attention_2
        elif seg is not None:
            seg_1 = F.interpolate(seg, size=(h0, w0))
            seg_2 = F.interpolate(seg, size=(h2, w2))

            seg_attention_1 = self.seg_attention1(seg_1)
            seg_attention_2 = self.seg_attention2(seg_2)
            layers[0] = layers[0] * seg_attention_1
            layers[2] = layers[2] * seg_attention_2

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


class _MultiscaleEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        """
        branch 1~4 is Atrous Conv with dilation rate  [0 (conv-1x1), 6, 12, 18] * rate
        branch 5 is global pooling
        :param dim_in: in_channel
        :param dim_out: out_channel
        :param rate: dilation rate coefficient
        :param bn_mom:
        """
        super().__init__()
        dim_min = max(dim_out // 4, 32)
        self.dim_out = dim_out
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True),
        )
        self.bn1 = nn.Sequential(
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate, dilation=rate, bias=True),
        )
        self.bn2 = nn.Sequential(
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        # self.branch3 = nn.Sequential(
        #     nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
        # )
        # self.bn3 = nn.Sequential(
        #     SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
        #     nn.ReLU(inplace=False),
        # )
        #
        # self.branch4 = nn.Sequential(
        #     nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
        # )
        # self.bn4 = nn.Sequential(
        #     SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
        #     nn.ReLU(inplace=False),
        # )
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim_out, dim_min, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(dim_min, dim_out, 1, 1, padding=0, bias=False)
        self.fc2 = nn.Conv2d(dim_min, dim_out, 1, 1, padding=0, bias=False)
        # self.fc3 = nn.Conv2d(dim_min, dim_out, 1, 1, padding=0, bias=False)
        # self.fc4 = nn.Conv2d(dim_min, dim_out, 1, 1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.category_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
        )
        self.category_max = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
        )
        self.category_fc = nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, bias=True)

    def forward(self, x):
        [b, c, h, w] = x.size()
        conv1x1 = self.branch1(x)
        _conv1x1 = self.bn1(conv1x1)

        conv3x3 = self.branch2(x)
        _conv3x3 = self.bn2(conv3x3)

        # conv5x5 = self.branch3(x)
        # _conv5x5 = self.bn3(conv5x5)
        #
        # conv7x7 = self.branch4(x)
        # _conv7x7 = self.bn4(conv7x7)

        fuse = _conv1x1 + _conv3x3  # + _conv5x5 + _conv7x7
        gp = self.avg_pooling(fuse)
        v = self.fc(gp)
        v1 = self.fc1(v)
        v2 = self.fc2(v)
        # v3 = self.fc3(v)
        # v4 = self.fc4(v)
        # vc = torch.cat([v1, v2, v3, v4], dim=1).view(b, 4, self.dim_out, 1, 1)
        vc = torch.cat([v1.unsqueeze(dim=1), v2.unsqueeze(dim=1)], dim=1).view(b, 2, self.dim_out, 1, 1)
        att = self.softmax(vc)
        # conv_cat = torch.cat([conv1x1, conv3x3, conv5x5, conv7x7], dim=1).view(b, 4, self.dim_out, h, w) * att
        conv_cat = torch.cat([conv1x1.unsqueeze(dim=1),
                              conv3x3.unsqueeze(dim=1)], dim=1).view(b, 2, self.dim_out, h, w) * att
        spatial_att = conv_cat.sum(dim=1).sigmoid()

        ca = self.category_avg(x)
        cm = self.category_max(x)
        category_att = (self.mlp(ca) + self.mlp(cm)).sigmoid()
        return spatial_att + category_att


class MultiscaleEncoder(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        """
        branch 1~4 is Atrous Conv with dilation rate  [0 (conv-1x1), 6, 12, 18] * rate
        branch 5 is global pooling
        :param dim_in: in_channel
        :param dim_out: out_channel
        :param rate: dilation rate coefficient
        :param bn_mom:
        """
        super().__init__()
        dim_min = max(dim_out // 16, 32)
        self.dim_min = dim_min
        self.dim_out = dim_out

        self.pre_conv = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=rate, dilation=rate, bias=True),
        )
        self.pre_conv_bn = nn.Sequential(
            SynchronizedBatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=False),
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(dim_out, dim_min, 1, 1, padding=0, bias=True),
            SynchronizedBatchNorm2d(dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.psp1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim_min, dim_min, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.psp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(dim_min, dim_min, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.psp3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(dim_min, dim_min, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.psp4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(dim_min, dim_min, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Conv2d(50 * dim_min, 4 * dim_min, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(4 * dim_min, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.psp_att = nn.Sequential(
            nn.Conv2d(4 * dim_min, dim_out, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        [b, c, h, w] = x.size()
        pre_conv = self.pre_conv(x)
        _pre_conv = self.pre_conv_bn(pre_conv)
        reduce = self.reduce_conv(_pre_conv)
        psp1 = self.psp1(reduce)
        psp2 = self.psp2(reduce)
        psp3 = self.psp3(reduce)
        psp4 = self.psp4(reduce)
        psp = torch.cat([psp1.view(b, 1 * self.dim_min, 1, 1),
                         psp2.view(b, 4 * self.dim_min, 1, 1),
                         psp3.view(b, 9 * self.dim_min, 1, 1),
                         psp4.view(b, 36 * self.dim_min, 1, 1)], dim=1)
        fc = self.fc1(psp)
        psp_att = self.psp_att(fc)
        return (pre_conv + pre_conv * psp_att).sigmoid()


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

