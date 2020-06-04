"""Preprocessing.

Allows all networks to use the same preprocessing functions, if necessary.
The preprocessing functions are:
    - Add pixel coordinates to the input tensor.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import constants
import torch


# Constant INDICES to save a operations
INDICES = torch.zeros((1, 2, constants.DIMENSIONS[1], constants.DIMENSIONS[0]))\
    .to(device=constants.DEVICE)


def change_indices_batch_size(batch_size):
    # First create a new tensor with a "batch size" of whatever is needed, with
    # 2 channels (x, y coordinates), and a size equivalent to the input tensor)
    global INDICES
    INDICES = torch.zeros((batch_size, 2, constants.DIMENSIONS[1],
                           constants.DIMENSIONS[0])).to(device=constants.DEVICE)
    for i in range(batch_size):
        # Then fill it with the normalized indices from constants
        INDICES[i] = constants.NORMALIZED_INDICES


# Initialize INDICES with 1 item batch size
change_indices_batch_size(1)


class Preprocessing:
    @staticmethod
    def append_px_coordinates(input_tensor):
        """Appends pixel coordinates to a given input tensor array.

        Args:
            input_tensor (torch.Tensor): The input tensor to be appended.
        """
        # change INDICES batch size if it's not the same as the input tensor
        if INDICES.shape[0] != input_tensor.shape[0]:
            change_indices_batch_size(input_tensor.shape[0])

        return torch.cat((input_tensor, INDICES), dim=1)
