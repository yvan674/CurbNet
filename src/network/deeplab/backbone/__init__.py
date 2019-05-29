from . import drn

def build_backbone(backbone, output_stride, BatchNorm, pretrained,
                   input_channels):
    if backbone == 'drn':
        return drn.drn_d_54(BatchNorm, input_channels, pretrained)
    else:
        raise NotImplementedError
