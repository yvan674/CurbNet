from . import drn

def build_backbone(backbone, output_stride, BatchNorm, pretrained):
    if backbone == 'drn':
        return drn.drn_d_54(BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
