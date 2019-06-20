"""Masked Cross Entropy Loss.

This custom loss function uses cross entropy loss as well as ground truth
data to calculate a loss specific to this use case scenario. It calculates a
regular cross entropy loss, but additionally heavily penalizes any curb
classification that is not around the perimeter of known roads.

The perimeter around known roads is calculated by using a binary dilation on B.
B is a b x b matrix, with b being 0.1 * image width.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import constants
from scipy.ndimage.morphology import binary_dilation
import numpy as np


class MCELoss(nn.CrossEntropyLoss):
    def __init__(self, weight_normal=None, weight_penalized=None,
                 size_average=None, ignore_index=-100, reduce=None,
                 reduction='mean'):
        """Cross Entropy loss with a masked applied for different weights."""
        super(MCELoss, self).__init__(weight_normal, size_average, reduce,
                                      reduction)
        self.ignore_index = ignore_index
        self.register_buffer('weight_penalized', weight_penalized)

        # Calculate the size of the b matrix.
        self.b_size = int(constants.DIM_WIDTH * 0.05)  # Chosen based on
        # manually viewing the dataset
        # B is created later since it will have to depend on the batch size.

    def forward(self, input, target):
        """Requires target to also have the road mask.

        Args:
            input (torch.Tensor): The predicted segmentation
            target (torch.Tensor): The ground truth segmentation. The road mask
                should be given as class k, where the network predicts k
                classes. E.g. given 3 classes (0, 1, 2), class (3) should be the
                road mask.
        """
        # Create a mask, that's really the entire image
        target_mask = torch.ones(target.shape).to(dtype=torch.uint8,
                                                  device=constants.DEVICE)
        # Prepare the mask for the 3 channel input by copying it basically
        # Assuming that the batch is relatively small so this shouldn't take
        # much time even if it is a nested for loop
        input_mask = torch.zeros(input.shape).to(constants.DEVICE)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                input_mask[i, j] = target_mask[i]

        # Send all the masks to the proper device
        target_mask = target_mask.to(device=constants.DEVICE, dtype=torch.uint8)
        input_mask = input_mask.to(device=constants.DEVICE, dtype=torch.uint8)

        # The values within the mask are now selected as well as the inverse.
        # The loss is then calculated separately for those in the mask and not
        # in the mask.
        perimeter_target = torch.masked_select(target, target_mask).\
            reshape(target.shape)

        # Set roads to 0
        perimeter_target[perimeter_target == 3] = 0
        perimeter_predicted = torch.masked_select(input, input_mask).\
            reshape(input.shape)

        return F.cross_entropy(perimeter_predicted, perimeter_target,
                               weight=self.weight,
                               ignore_index=self.ignore_index,
                               reduction=self.reduction)

    @staticmethod
    def flip_tensor(tensor):
        """Flips values of 0 and 1 in a given tensor."""
        flipped = tensor.clone()
        flipped[tensor == 0] = 1
        flipped[tensor == 1] = 0
        return flipped
