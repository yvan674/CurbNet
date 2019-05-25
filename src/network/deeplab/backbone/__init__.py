from . import drn

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    else:
        raise NotImplementedError
