# -*- coding: UTF-8 -*-
"""CurbNet.

A deep neural network designed to identify and segment urban images for curbs
and curb cuts.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import torch.nn as nn
import torch.functional as F


class CurbNet(nn.Module):
    def __init__(self):
        """A neural network that identifies and segments curbs and curb cuts."""
        super(CurbNet, self).__init__()  # Initialize the superclass

        self.fc1 = nn.Linear(5, 5)

        # Nothing here yet

    def forward(self, img):
        """Describes the architecture of the neural network.

        Args:
            img (torch.Tensor): The input image as a tensor.

        Returns:
            (torch.Tensor): The segmentation of the image.
        """
        out = self.fc1(img)
        return out
