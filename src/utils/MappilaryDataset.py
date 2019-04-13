#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
"""Mappilary Dataset.

This module creates a class that extends the torch Dataset object to parse the
mappilary dataset images. Each data point in the dataset contains a dictionary
containing the raw image, the segmentation, and the panoptic segmentation.

Authors:
    Yvan Satyawan <ys88@saturn.uni-freiburg.de>
"""
import torch
from torch.utils.data import Dataset
from skimage import io
import imgaug as ia
from imgaug import augmenters as iaa

# Here we define probabilities
class prob:
    @staticmethod
    def st(aug):
        """Defines the "sometimes" probability value."""
        return iaa.Sometimes(0.4, aug)

    @staticmethod
    def oc(aug):
        """Defines the "occasionally" probability value."""
        return iaa.Sometimes(0.3, aug)

    @staticmethod
    def rl(aug):
        """Defines the "rarely" probability value."""
        return iaa.Sometimes(0.09, aug)

    @staticmethod
    def seq(self):
        """Returns a sequential object that outputs an augmentation."""
        seq = iaa.Sequential([
            # blur images with a sigma between 0 and 1.5
            self.rl(iaa.GaussianBlur((0, 1.5))),
            # randomly remove up to X% of the pixels
            self.oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),
            # randomly remove up to X% of the pixels
            self.oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),
                                 per_channel=0.5)),
            # change brightness of images (by -X to Y of original value)
            self.oc(iaa.Add((-40, 40), per_channel=0.5)),
            # change brightness of images (X-Y% of original value)
            self.st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),
            # improve or worsen the contrast
            self.rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
            ],
            random_order=True)
        return seq

class MappilaryDataset(Dataset):
    def __init__(self, path):
        """Dataset object that contains the mappilary dataset.

        Args:
            path (string): The path to the mappilary dataset folder.
        """
        super().__init__()

        # TODO Do stuff

    def __len__(self):
        """Returns the number of items in the dataset."""
        # TODO

    def __getitem__(self, idx):
        """Returns the items in the dataset with the specified idx.

        Args:
            idx (int): The id of the item number that is requested.

        Returns:
            A dictionary with the keys (raw, segment, panoptic), each with their
            corresponding image as a tensor.
        """
        # TODO
