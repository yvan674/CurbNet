"""CurbNetD.

A deep neural network designed to identify and segment urban images for curbs
and curb cuts. This version is based on DeepLab v3.

References:
    Rethinking Atrous Convolution for Semantic Image Segmentation.
        arXiv:1706.05587 [cs.CV]

Based on the implementation found at:
    https://github.com/jfzhang95/pytorch-deeplab-xception
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .deeplab.batchnorm import SynchronizedBatchNorm2d
from .deeplab.aspp import build_aspp
from .deeplab.decoder import build_decoder
from .deeplab.backbone import build_backbone

class CurbNetD(nn.Module):
    def __init__(self, backbone='drn', output_stride=8, num_classes=3,
                 sync_bn=True, freeze_bn=False):
        super(CurbNetD, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input: torch.Tensor):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear',
                          align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
