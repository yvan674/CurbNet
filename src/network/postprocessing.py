"""Postprocessing.

This module post processes the output of the neural network using
CRF
"""
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral,\
    create_pairwise_gaussian


def crf(generated_batch):
    """Applies dense CRF to a network generated batch."""
    shape = generated_batch[0].shape

    # Setup DCRF
    d = dcrf.DenseCRF2D(shape[1], shape[0], 3)

    for item in generated_batch:
        # Get unary potentials
        u = unary_from_labels(item, 3, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(u)

        # Add color independent term
        features = create_pairwise_gaussian(sdims=(3, 3), shape=shape[:2])
        d.addPairwiseGaussian(features, compat=3, kernal=dcrf.DIAG_KERNAL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Add color dependent term

# TODO Understand this
