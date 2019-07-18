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
        self.b_size = int(constants.DIM_WIDTH * 0.03)  # Chosen based on
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
        # Extract the road mask from the target
        mask = torch.zeros(target.shape, dtype=torch.uint8,
                           device=constants.DEVICE)
        mask[target == 3] = 1.

        # Create b
        b = np.ones((self.b_size, self.b_size))

        # Calculate the road perimeter mask
        # After testing, element-wise is significantly faster than a single
        # statement for some reason.
        target_mask = np.zeros(target.shape)
        mask = mask.detach().cpu().numpy()

        for i in range(target_mask.shape[0]):
            target_mask[i] = binary_dilation(mask[i], b)

        target_mask = torch.from_numpy(target_mask).to(dtype=torch.uint8,
                                                       device=constants.DEVICE)
        # Remove the road so we get only the perimeter
        target_mask[target == 3] = 0

        # Turn road back into other for loss classification
        target[target == 3] = 0

        # Prepare the mask for the 3 channel input by copying it basically
        # Assuming that the batch is relatively small so this shouldn't take
        # much time even if it is a nested for loop
        input_mask = torch.zeros(input.shape).to(constants.DEVICE)
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                input_mask[i, j] = target_mask[i]

        # Get the inverted mask as well
        inverted_target_mask = self.flip_tensor(target_mask)
        inverted_input_mask = self.flip_tensor(input_mask)

        # Send all the masks to the proper device
        target_mask = target_mask.to(device=constants.DEVICE, dtype=torch.uint8)
        input_mask = input_mask.to(device=constants.DEVICE, dtype=torch.uint8)
        inverted_target_mask = inverted_target_mask.to(device=constants.DEVICE,
                                                       dtype=torch.uint8)
        inverted_input_mask = inverted_input_mask.to(device=constants.DEVICE,
                                                     dtype=torch.uint8)

        # Create a single length zero tensor once
        zero = torch.zeros(1).to(device=constants.DEVICE)

        # The values within the mask are now selected as well as the inverse.
        # The loss is then calculated separately for those in the mask and not
        # in the mask.
        # We use torch.where to preserve the shape of the mask

        perimeter_target = torch.where(target_mask, target.long(),
                                       zero.long())
        perimeter_predicted = torch.where(input_mask, input, zero)

        other_target = torch.where(inverted_target_mask, target.long(),
                                   zero.long())
        other_predicted = torch.where(inverted_input_mask, input, zero)

        perimeter_loss = F.cross_entropy(perimeter_predicted, perimeter_target,
                                         weight=self.weight,
                                         ignore_index=self.ignore_index,
                                         reduction=self.reduction)
        other_loss = F.cross_entropy(other_predicted, other_target,
                                     weight=self.weight_penalized,
                                     ignore_index=self.ignore_index,
                                     reduction=self.reduction)
        return perimeter_loss + other_loss

    @staticmethod
    def flip_tensor(tensor):
        """Flips values of 0 and 1 in a given tensor."""
        flipped = tensor.clone()
        flipped[tensor == 0] = 1
        flipped[tensor == 1] = 0
        return flipped
