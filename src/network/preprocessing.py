"""Preprocessing.

Allows all networks to use the same preprocessing functions, if necessary.
The preprocessing functions are:
    - Add pixel coordinates to the input tensor.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import constants
import numpy as np


class Preprocessing:
    @staticmethod
    def append_px_coordinates(input_tensor):
        """Appends pixel coordinates to a given input tensor array.

        Args:
            input_tensor (torch.Tensor): The input tensor to be appended.
        """
        return np.append(input_tensor, constants.NORMALIZED_INDICES, axis=0)