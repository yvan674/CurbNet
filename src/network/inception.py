"""Inception.

Implementation of the Inception network module as described in "Going deeper
with convolutions".

References:
    Going Deeper with Convolutions
    arXiv:1409.4842 [cs.CV] 17 Sep 2014
"""
import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels: int, n1x1: int, n3x3red: int, n3x3: int,
                 n5x5red: int, n5x5: int, pool_channels: int):
        """Inception module as described in "Going Deeper with Convolutions."

        This is a Pytorch implementation of the inception module that is used as
        the basic building block of the GoogLeNet encoder network described in
        the aforementioned paper.

        Args:
            in_channels: The number of channels in the input.
            n1x1: The number of channels the 1x1 conv should output.
            n3x3red: The number of channels the 1x1 to 3x3 conv reduction
            should output.
            n3x3: The number of channels the 3x3 conv branch should
            output.
            n5x5red: The number of channels the 1x1 to 5x5 conv reduction
            should output.
            n5x5: The number of channels the 5x5 conv branch should
            output.
            pool_channels: The number of channels the max pool 1x1 conv
            should output.
        """
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_channels, kernel_size=1),
            nn.BatchNorm2d(pool_channels),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs a forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor after being passed through the inception
            module.
        """
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat((y1, y2, y3, y4), 1)
