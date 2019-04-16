# -*- coding: UTF-8 -*-
"""Mappilary Dataset.

This module creates a class that extends the torch Dataset object to parse the
mapillary dataset images. Each data point in the dataset contains a dictionary
containing the raw image, the segmentation, and the panoptic segmentation.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
from os import listdir
from os.path import join

from torch.utils.data import Dataset
from skimage import io
from imgaug import augmenters as iaa


# Here we define probabilities
class augment:
    @staticmethod
    def augment(image):
        """Augments an image in the dataset.

        Args:
            image (imageio.core.util.Array): The image to be augmented.

        Returns:
            imageio.core.util.Array: The augmented image.
        """
        # Define the probabilities
        def rl(aug):
            """Defines the "rarely" probability value."""
            return iaa.Sometimes(0.09, aug)

        def oc(aug):
            """Defines the "occasionally" probability value."""
            return iaa.Sometimes(0.3, aug)

        def st(aug):
            """Defines the "sometimes" probability value."""
            return iaa.Sometimes(0.4, aug)

        # set up the sequential augmentation
        seq = iaa.Sequential([
            # blur images with a sigma between 0 and 1.5
            rl(iaa.GaussianBlur((0, 1.5))),
            # randomly remove up to X% of the pixels
            oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),
            # randomly remove up to X% of the pixels
            oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),
                                 per_channel=0.5)),
            # change brightness of images (by -X to Y of original value)
            oc(iaa.Add((-40, 40), per_channel=0.5)),
            # change brightness of images (X-Y% of original value)
            st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),
            # improve or worsen the contrast
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
            ],
            random_order=True)

        # Return the augmented image
        return seq.augment_image(image)


class MappilaryDataset(Dataset):
    def __init__(self, path, with_aug):
        """Dataset object that contains the mapillary dataset.

        The dataset takes all the types of images within the specified training,
        testing, or validation dataset. The items returned are given in a
        dictionary containing the raw image, segmentation, and panoptic
        segmentation.

        Args:
            path (string): The path to the specific mapillary dataset folder.
            with_aug (bool): Whether or not to use image augmentation.
        """
        super().__init__()

        # Save the args
        self.path = path
        self.with_aug = with_aug
        self.images_dir = join(self.path, "images")
        self.seg_dir = join(self.path, "labels")
        self.pan_dir = join(self.path, "panoptic")

        # Save a list of all the image names
        self.images = listdir(self.images_dir)

        # TODO Do stuff

    def __len__(self):
        """Returns the number of items in the dataset.

        This assumes that the number of images in the images directory is equal
        to the number of image in every on eof the other directories.

        Returns:
            int: The number of images in the dataset.
        """
        return len(listdir(self.images_dir))


    def __getitem__(self, idx):
        """Returns the items in the dataset with the specified idx.

        Args:
            idx (int): The id of the item number that is requested.

        Returns:
            dict: A sample with keys (raw, segment, panoptic), each with their
            corresponding image as a tensor.
        """
        # Set paths


        # Get the dict images, processed to tensors
        out = {
            "raw": self.to_tensor(io.imread(
                join(self.images_dir, self.images[idx]))),
            "segmented": self.to_tensor(io.imread(
                join(self.seg_dir, self.images[idx]))),
            "panoptic": self.to_tensor(io.imread(
                join(self.pan_dir, self.images[idx])))
        }

    def to_tensor(self, image):
        """Transform the given image to a tensor and augments it.

        Args:
            image (imageio.core.util.Array): The image to be turned into a
            tensor.

        Returns:
            torch.Tensor: The image as a torch tensor.
        """
        if self.with_aug:
            # apply image augmentation sequential
            image = augment.augment(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image).to(dtype=torch.float)
