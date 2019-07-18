"""Search Worker.

Searches for the optimal hyperparameters using HpBandSter, which is an
implementation of BOHB from the AutoML group at Uni Freiburg.

The hyperparameters we will be optimizing are as follows:

+----------------+----------------+-----------------+------------------------+
| Parameter name | Parameter type |      Range      |        Comment         |
+----------------+----------------+-----------------+------------------------+
| Learning rate  | float          | [1e-5, 1e-2]    | varied logarithmically |
| Optimizer      | categorical    | {'adam', 'sgd'} | choose one             |
| SGD momentum   | float          | [0, 0.99]       | only active when       |
|                |                |                 | optimizer == 'sgd'     |
| Adam epsilon   | float          | [1e-2, 1]       | using step size 5e-2   |
| sync batchnorm | bool           | {True, False}   | using synchronized     |
|                |                |                 | batchnorm or regular   |
|                |                |                 | batchnorm              |
| loss weight    | float          | {2, 6}          | intervals of .25        |
| ratio          |                |                 |                        |
+----------------+----------------+-----------------+------------------------+

Note:
    Loss weight ratio is the ratio between the weights of the curb to curb cuts.
    The ratio between the other class and both curb and curb cut class is 1:10.
Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    BOHB: Robust and Efficient Hyperparameter Optimization at Scale
        <http://proceedings.mlr.press/v80/falkner18a.html>
"""
# Torch imports
import torch
from torch.utils.data.dataloader import DataLoader

# numpy
import numpy as np

# Python built in imports
import time

import ConfigSpace as CS
from hpbandster.core.worker import Worker


class SearchWorker(Worker):
    def __init__(self, *args, **kwargs):
        super(SearchWorker, self).__init__(*args, **kwargs)

    def compute(self, config, budget, **kwargs):
        """Runs the training session.

        Args:
            config (dict): Dictionary containing the configuration by the optimizer
            budget (int): Amount of epochs the model can use to train.

        Returns:
            dict: dictionary with fields 'loss' (float) and 'info' (dict)
        """
        counter = 0
        self.tracker = PlotCSV()


    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()

    @staticmethod
    def _calculate_loss_weights(ratio):
        """Calculates the loss weights based on a given ratio.

        Args:
            ratio (float): The ratio between the curb and curb cut weights.

        Returns:
            list: A normalized class-wise weight list for MCE loss.
        """
        curb_weight = 1.
        curb_cut_weight = ratio
        other_weight = (curb_weight + curb_cut_weight) / 10

        normalization_constant = 1 / (curb_weight + curb_cut_weight
                                      + other_weight)
        return [other_weight * normalization_constant,
                curb_weight * normalization_constant,
                curb_cut_weight * normalization_constant]
