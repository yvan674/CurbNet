"""Parallelizer.

This module allows for any network to be run in parallel on multiple GPUs using
the nn.DataParallel module.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch.nn as nn


class Parllelizer(nn.Module):
    def __init__(self, network):
        """Parllelizer allows for any network to be run in parallel.

        Args:
            network (nn.Module): The torch module that is to be parallelized
        """
        super(Parllelizer, self).__init__()

        self.parallelized = network
        self.parallelized = nn.DataParallel(self.parallelized)


    def forward(self, x):
        """Runs the CurbNetG network in parallel.

        Args:
            x (torch.Tensor): The input image as a tensor.

        Returns:
            (torch.Tensor): The segmentation of the image.
        """
        x = self.parallelized(x)
        return x
