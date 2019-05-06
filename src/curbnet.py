# -*- coding: UTF-8 -*-
"""CurbNet.

A deep neural Network designed to identify and segment urban images for curbs
and curb cuts.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    Fully Convolutional Networks for Semantic Segmentation
    arXiv:1605.06211v1 [cs.CV] 20 May 2016

    Going Deeper with Convolutions
    arXiv:1409.4842 [cs.CV] 17 Sep 2014
"""
import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5,
                 pool_planes):
        """Inception module as described in Going Deeper with Convolutions."""
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
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
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat((y1, y2, y3, y4), 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        """GoogLeNet architecture described in Going Deeper with Convolutions.

        This is the decoder architecture we'll try to use. It is an
        implementation of the architecture described in Going Deeper with
        Convolutions known as GoogLeNet. It uses the Inception module described
        above.
        """
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)

        # Skips
        self.skip1 = None
        self.skip2 = None

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        return out


class CurbNet(nn.Module):
    def __init__(self):
        """A neural Network that identifies and segments curbs and curb cuts."""
        super(CurbNet, self).__init__()  # Initialize the superclass

        # Currently use the same architecture from the AIS project, i.e.
        # GoogLeNet encoder with FC decoders

        self.encoder = GoogLeNet()
        classifier_conv = nn.Conv2d(1024, 3, 1)
        self._normal_initialization(classifier_conv)
        self.classifier_conv = classifier_conv
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _normal_initialization(layer):
        """Initializes a layer with some weights.
        Args:
            layer (nn.Module): The layer to be initialized.
        """
        layer.weight.data.normal_(0, 0.01)

        layer.bias.data.zero_()

    def forward(self, x):
        """Describes the architecture of the neural Network.

        Args:
            x (torch.Tensor): The input image as a tensor.

        Returns:
            (torch.Tensor): The segmentation of the image.
        """
        # Calculate the spatial dimension for output later
        input_spatial_dim = x.size()[2:]

        # Run the Network
        out = self.encoder(x)
        out = self.classifier_conv(out)

        out = nn.functional.interpolate(input=out, size=input_spatial_dim,
                                        mode="bilinear")
        out = self.softmax(out)

        return out
