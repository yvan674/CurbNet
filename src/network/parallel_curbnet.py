"""Parallel CurbNet.

This module allows for CurbNet to be run in parallel on multiple GPUs using the
nn.DataParallel module.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch.nn as nn
from network.curbnet import CurbNet


class ParallelCurbNet(nn.Module):
    def __init__(self):
        super(ParallelCurbNet, self).__init__()

        self.parallelized = CurbNet()
        self.parallelized = nn.DataParallel(self.parallelized)

    def forward(self, x):
        """Runs the CurbNet network in parallel.

        Args:
            x (torch.Tensor): The input image as a tensor.

        Returns:
            (torch.Tensor): The segmentation of the image.
        """
        x = self.parallelized(x)
        return x
