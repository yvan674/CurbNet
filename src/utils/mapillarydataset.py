# -*- coding: UTF-8 -*-
"""Mappilary Dataset.

This module creates a class that extends the torch Dataset object to parse the
mapillary dataset images. Each data point in the dataset contains a dictionary
containing the raw image, the segmentation, and the panoptic segmentation.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from os.path import join, exists
from PIL import Image
import constants


def augment(image, iaa):
    """Augments a given input image.

    Args:
        image (imageio.core.util.Array): The image to be augmented.

    Returns:
        imageio.core.util.Array: The augmented image.
    """

    # Define the probabilities
    def rl(aug):
        """Defines the "rarely" probability value."""
        return iaa.Sometimes(0.03, aug)

    def oc(aug):
        """Defines the "occasionally" probability value."""
        return iaa.Sometimes(0.1, aug)

    def st(aug):
        """Defines the "sometimes" probability value."""
        return iaa.Sometimes(0.2, aug)

    # set up the sequential augmentation
    seq = iaa.Sequential([
        # Blur images with a sigma between 0 and 1.5
        rl(iaa.GaussianBlur((0, 1.5))),

        # Pepper X% of the pixels
        oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),

        # Pepper X% of the pixels
        oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),
                             per_channel=0.5)),

        # Brightness (by -X to Y of original value)
        oc(iaa.Add((-20, 20), per_channel=0.5)),

        # Brightness (X-Y% of original value)
        st(iaa.Multiply((0.5, 1.5), per_channel=0.2)),

        # Contrast
        rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),
    ],
        random_order=True)

    # Return the augmented image
    return seq.augment_image(image)


class MapillaryDataset(Dataset):
    def __init__(self, path, with_aug, iaa):
        """Dataset object that contains the mapillary dataset.

        The dataset takes all the types of images within the specified training,
        testing, or validation dataset. The items returned are given in a
        dictionary containing the raw image, segmentation, and panoptic
        segmentation.

        Args:
            path (string): The path to the specific mapillary dataset.
            with_aug (bool): Whether or not to use image augmentation.
        """
        super().__init__()

        self.iaa = iaa

        self.resized_available = False
        # Set mode (i.e. if small files available, load all images to memory)
        if exists(path + "_resized"):
            self.path = path + "_resized"
            self.resized_available = True
        else:
            self.path = path

        # Save the args
        self.with_aug = with_aug
        self.images_dir = join(self.path, "images")
        self.seg_dir = join(self.path, "labels")
        self.pan_dir = join(self.path, "panoptic")

        self.images = []
        self.loaded_images = None
        self.loaded_labels = None

        # Read viability list
        with open(join(path, "viable.txt"), mode='r') as viability:
            for line in viability:
                self.images.append(line[:-1])

        if self.resized_available:
            # If resized folder available, to load images into memory
            self.loaded_images = []
            self.loaded_labels = []
            for image in self.images:
                # Open the images
                image_file = Image.open(join(self.images_dir, image + ".jpg"))
                label_file = Image.open(join(self.seg_dir, image + ".png"))

                # Load them to memory
                self.loaded_images.append(np.array(image_file))
                self.loaded_labels.append(np.array(label_file))

                # Close the images
                image_file.close()
                label_file.close()

    def __len__(self):
        """Returns the number of items in the dataset.

        Calculates by getting a len of the self.images list, which contains all
        the values found in viable.txt

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Returns the items in the dataset with the specified idx.

        Args:
            idx (int): The id of the item number that is requested.

        Returns:
            dict: A sample with keys (raw, segmented), raw having the image as a
            tensor and segmented having the image as a numpy array.
        """
        if self.resized_available:
            raw = self._process_raw(self.loaded_images[idx])
            segmented = self._process_segmented(self.loaded_labels[idx])

        else:
            # Otherwise load them from the hard drive
            raw = Image.open(join(self.images_dir, self.images[idx] + ".jpg"))
            raw = np.array(raw)
            raw = self._process_raw(raw)

            segmented = Image.open(join(self.seg_dir,
                                        self.images[idx] + ".png"))
            segmented = np.array(segmented)
            segmented = self._process_segmented(segmented)

        # Turn the selected file into a dict
        out = {
            "raw": raw,
            "segmented": segmented
        }
        return out

    def _process_raw(self, image):
        """Transform the given image to a tensor and augments it.

        This also adds the INDICES array, to allow the network to also
        learn from pixel coordinates.

        Args:
            image (numpy.array): The image to be turned into a tensor as a numpy
                array.

        Returns:
            torch.Tensor: The image as a torch tensor.
        """
        # Crop the image to 4:3 ratio
        image = self._resize(image, "cubic", self.iaa)

        if self.with_aug:
            # apply image augmentation sequential
            image = augment(image, self.iaa)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        # Here we rescale the input image to [-0.5, 0.5] to improve numerical
        # stability
        image = image.astype(dtype=float)
        image /= 255.
        image -= 0.5

        return torch.from_numpy(image).to(dtype=torch.float)

    def _process_segmented(self, image):
        """Processes segmentation image to classes from a given config file.

        Each pixel in the image should be given a label that corresponds to if
        it is irrelevant, a curb, or a cut curb. 0 is irrelevant, 1 is curb, and
        2 is curb cut.

        Args:
            image (np.array): The image to be processed as a numpy array.

        Returns:
            np.array: An array of the images with one-hot labelling.
        """
        # First resize the image
        image = self._resize(image, "nearest", self.iaa)

        # Create an out array
        out_array = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        out_array[image == 2] = 1  # Label for curb
        out_array[image == 9] = 2  # Label for curb cut

        # All road labels, including any markings
        # 10: parking
        # 13: road
        # 14: service lane
        # 23: Crosswalk
        # 24: Marking
        # 41: Manhole
        # 43: Pothole
        labels = [10, 13, 14, 23, 24, 41, 43]
        for i in labels:
            out_array[image == i] = 3

        return out_array

    @staticmethod
    def _resize(image, interpolation, iaa):
        """Resizes a given input image to DIMENSIONS

        Args:
            image (imageio.core.util.Array): The image to be resized.
            interpolation (str): Sampling when resizing. Either 'cubic' or
                                 'nearest'.

        Returns:
            imageio.core.util.Array: The resized image.
        """
        # If image is already the right size, don't do anything.
        if image.shape[0] == constants.DIMENSIONS[1] \
                and image.shape[1] == constants.DIMENSIONS[0]:
            return image

        # Calculate what the cropped size should be based on dimensions
        if 4 * image.shape[1] >= 3 * image.shape[0]:
            # the height is the smaller element
            cropped_height = image.shape[0]
            cropped_width = int(image.shape[0] * 1.33333333333333)
        else:
            cropped_height = int(image.shape[1] * .75)
            cropped_width = image.shape[1]

        # Define the transformations
        seq = iaa.Sequential([
            # Crop to aspect ratio
            iaa.CropToFixedSize(cropped_width, cropped_height,
                                position='center'),

            # Resize to proper size
            iaa.Resize({"width": constants.DIMENSIONS[0],
                        "height": constants.DIMENSIONS[1]},
                       interpolation=interpolation)
        ])

        return seq.augment_image(image)
