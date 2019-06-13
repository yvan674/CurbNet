"""Class-based Cross Entropy Loss.

This custom loss function uses cross entropy loss as well as ground truth
data to calculate a loss specific to this use case scenario. It calculates a
regular cross entropy loss, but additionally heavily penalizes any curb
classification that is not around the perimeter of known roads.

The perimeter around known roads is calculated by using a binary dilation on B.
B is a b x b matrix, with b being 0.05 * image width.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import torch
import torch.nn as nn
import constants


class CCELoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CCELoss, self).__init__(weight, size_average, reduce,
                                               reduction)
        self.ignore_index = ignore_index

        # Create the matrix B
        b_size = int(constants.DIM_WIDTH * 0.05)
        b = torch.ones((b_size))

