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
from os.path import join, splitext

from torch.utils.data import Dataset
from skimage import io
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image


# Here we define augmentation
class Augment:
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


class MapillaryDataset(Dataset):
    def __init__(self, path, with_aug):
        """Dataset object that contains the mapillary dataset.

        The dataset takes all the types of images within the specified training,
        testing, or validation dataset. The items returned are given in a
        dictionary containing the raw image, segmentation, and panoptic
        segmentation.

        Args:
            path (string): The path to the specific mapillary dataset subfolder.
            with_aug (bool): Whether or not to use image augmentation.
        """
        super().__init__()

        # Save the args
        self.path = path
        self.with_aug = with_aug
        self.images_dir = join(self.path, "images")
        self.seg_dir = join(self.path, "labels")
        self.pan_dir = join(self.path, "panoptic")

        self.images = []

        # Read viability list
        with open(join(path, "viable.txt"), mode='r') as viability:
            for line in viability:
                self.images.append(line)

    def __len__(self):
        """Returns the number of items in the dataset.

        This assumes that the number of images in the images directory is equal
        to the number of image in every on eof the other directories.

        Returns:
            int: The number of images in the dataset.
        """
        return len(listdir(self.seg_dir))

    def __getitem__(self, idx):
        """Returns the items in the dataset with the specified idx.

        Args:
            idx (int): The id of the item number that is requested.

        Returns:
            dict: A sample with keys (raw, segmented), raw having the image as a
            tensor and segmented having the image as a numpy array.
        """
        raw = self._to_tensor(io.imread(join(self.images_dir,
                                             self.images[idx])))

        segmented = Image.open(join(self.seg_dir, splitext(self.images[idx])[0]
                                   + ".png"))
        segmented = np.array(segmented)
        segmented = self._process_segmented(segmented)

        # Get the dict images, processed to tensors
        out = {
            "raw": raw,
            "segmented": segmented
        }
        return out

    def _to_tensor(self, image):
        """Transform the given image to a tensor and augments it.

        Args:
            image (imageio.core.util.Array): The image to be turned into a
            tensor.

        Returns:
            torch.Tensor: The image as a torch tensor.
        """
        if self.with_aug:
            # apply image augmentation sequential
            image = Augment.augment(image)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image).to(dtype=torch.float)

    @staticmethod
    def _process_segmented(image):
        """Processes segmentation image to classes from a given config file.

        Each pixel in the image should be given a label that corresponds to if
        it is irrelevant, a curb, or a cut curb. 0 is irrelevant, 1 is curb, and
        2 is curb cut.

        Args:
            image (np.array): The image to be processed.

        Returns:
            np.array: An array of the images with one-hot labelling.
        """
        out_array = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        out_array[image == 2] = 1  # Label for curb
        out_array[image == 9] = 2  # Label for curb cut

        return out_array
