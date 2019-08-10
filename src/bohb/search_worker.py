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
| Sync batchnorm | bool           | {True, False}   | using synchronized     |
|                |                |                 | batchnorm or regular   |
|                |                |                 | batchnorm              |
| Loss weight    | float          | {2, 6}          | intervals of .25        |
| ratio          |                |                 |                        |
+----------------+----------------+-----------------+------------------------+

+-------------------+----------------+
|  Parameter name   | Name in config |
+-------------------+----------------+
| Learning Rate     | lr             |
| Optimizer         | optimizer      |
| SGD momentum      | momentum       |
| Adam epsilon      | epsilon        |
| Sync batchnorm    | sync_bn        |
| Loss weight ratio | weight_ratio   |
+-------------------+----------------+


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
from torch.optim import Adam, SGD
from utils.mapillarydataset import MapillaryDataset
from utils import calculate_accuracy
import numpy as np
from os import path
import ConfigSpace as CS
# import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
from network.curbnet_d import CurbNetD
from network.mce_loss import MCELoss
from network.parallelizer import Parallelizer
from constants import BATCH_SIZE

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SearchWorker(Worker):
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)

        self.training_loader = self._load_dataset(
            path.join(data_path, "training"), True, BATCH_SIZE)
        self.validation_loader = self._load_dataset(
            path.join(data_path, "validation"), True, BATCH_SIZE)
        self.device = torch.device("cuda")
        self.run_count = 0

    def compute(self, config, budget, **kwargs):
        """Runs the training session.

        Args:
            config (dict): Dictionary containing the configuration by the optimizer
            budget (int): Amount of epochs the model can use to train.

        Returns:
            dict: dictionary with fields 'loss' (float) and 'info' (dict)
        """
        print("\nStarting run {} with config:".format(self.run_count))
        print("    lr:           {}".format(config['lr']))
        print("    optimizer:    {}".format(config['optimizer']))
        print("    sync_bn:      {}".format(config['sync_bn']))
        print("    weight ratio: {}".format(config['weight_ratio']))
        if config['optimizer'] is "adam":
            print("    epsilon:      {:.2f}".format(config['epsilon']))
        else:
            print("    momentum:     {}".format(config['momentum']))

        self.run_count += 1

        network = Parallelizer(CurbNetD(sync_bn=config['sync_bn']).cuda())
        criterion = MCELoss(self._calculate_loss_weights(
            config['weight_ratio']))

        if config['optimizer'] == 'adam':
            optimizer = Adam(network.parameters(), lr=config['lr'],
                             eps=config['epsilon'])
        elif config['optimizer'] == 'sgd':
            optimizer = SGD(network.parameters(), lr=config['lr'],
                            momentum=config['momentum'])
        else:
            raise ValueError("Illegal optimizer value used.")

        for epoch in range(int(budget)):
            for data in enumerate(self.training_loader):
                network.train()
                optimizer.zero_grad()

                # Rename for easy access
                raw_image = data[1]["raw"]
                target_image = data[1]["segmented"]

                # Run network
                out = network(raw_image.to(self.device, non_blocking=True))

                loss = criterion(out, target_image.to(self.device,
                                                      dtype=torch.long,
                                                      non_blocking=True))
                loss.backward()
                optimizer.step()

                del out
                del raw_image
                del target_image

        validation_accuracy, loss = self.evaluate_network(
            network, criterion, self.validation_loader)

        average_validation_acc = (validation_accuracy[0]
                                  + validation_accuracy[1]
                                  + validation_accuracy[2]) / 3.

        return {'loss': 1 - average_validation_acc,
                'info': {'validation accuracy': validation_accuracy.tolist(),
                         'validation loss': loss
                         }
                }

    def evaluate_network(self, network, criterion, data_loader):
        """Evaluate network accuracy on a specific data set.

        Returns:
            list: Element-wise accuracy
            float: Average loss
        """
        # Set to eval and set up variables
        network.eval()
        accuracy = np.array([0., 0., 0.])
        loss = 0.
        total_values = float(len(data_loader))

        # Use network but without updating anything
        with torch.no_grad():
            for data in enumerate(data_loader):
                raw_image = data[1]["raw"].to(self.device, non_blocking=True)
                target_image = data[1]["segmented"]

                out = network.forward(raw_image)

                loss += criterion(out,
                                  target_image.to(self.device,
                                                  non_blocking=True)).item()

                out_argmax = torch.argmax(out.cpu().detach(), dim=1)

                accuracy += self._calculate_batch_accuracy(target_image,
                                                           out_argmax,
                                                           64)


            del raw_image
            del target_image
            del out
            del out_argmax

        # Average accuracy and loss
        loss /= total_values
        accuracy /= total_values

        return accuracy, loss

    def _calculate_batch_accuracy(self, ground_truth, predicted, batch_size):
        """Calculates accuracy of the batch using intersection over union.

        We use a separate function to make it easier to follow and for the
        actual training code easier to maintain.

        Args:
            ground_truth (torch.*Tensor): The batch of the ground truth data.
            predicted (torch.*Tensor): The batch of the predicted segmentation
                                       generated by the network.
            batch_size (int): The batch size used.

        returns:
            float: Average accuracy of the batch.
        """
        accuracy = np.array([0., 0., 0.])
        for idx, item in enumerate(ground_truth):
            accuracy += calculate_accuracy(item, predicted[idx])

        return accuracy / batch_size

    @staticmethod
    def get_configspace():
        """Builds the config space as described in the header docstring."""
        cs = CS.ConfigurationSpace()

        lr = CS.UniformFloatHyperparameter('lr',
                                           lower=1e-5,
                                           upper=1e-2,
                                           default_value=1e-4,
                                           log=True)

        optimizer = CS.CategoricalHyperparameter('optimizer', ['adam', 'sgd'])
        momentum = CS.UniformFloatHyperparameter('momentum', lower=0.,
                                                 upper=1.00,
                                                 default_value=0.9,
                                                 q=5e-2)
        epsilon = CS.UniformFloatHyperparameter('epsilon', lower=1e-2,
                                                upper=1.,
                                                default_value=0.1,
                                                q=5e-2)
        sync_bn = CS.CategoricalHyperparameter('sync_bn', [True, False])
        weight_ratio = CS.UniformFloatHyperparameter('weight_ratio',
                                                     lower=2.,
                                                     upper=6.,
                                                     default_value=5.,
                                                     q=.25)

        cs.add_hyperparameters([lr, optimizer, momentum, epsilon, sync_bn,
                               weight_ratio])
        cs.add_condition(CS.EqualsCondition(momentum, optimizer, 'sgd'))
        cs.add_condition(CS.EqualsCondition(epsilon, optimizer, 'adam'))

        return cs

    @staticmethod
    def _calculate_loss_weights(ratio):
        """Calculates the loss weights based on a given ratio.

        Args:
            ratio (float): The ratio between the curb and curb cut weights.

        Returns:
            torch.Tensor: A normalized class-wise weight list for MCE loss.
        """
        curb_weight = 1.
        curb_cut_weight = ratio
        other_weight = (curb_weight + curb_cut_weight) / 10

        normalization_constant = 1 / (curb_weight + curb_cut_weight
                                      + other_weight)



        return torch.tensor([
            other_weight * normalization_constant,
            curb_weight * normalization_constant,
            curb_cut_weight * normalization_constant]).cuda()

    @staticmethod
    def _load_dataset(data_path, augmentation, batch_size):
        """Loads a dataset and creates the data loader.

        Args:
            data_path (str): The path of the mapillary folder.
            augmentation (bool): Augment the images or not.
            batch_size (int): Size of each batch

        Returns:
            torch.utils.data.DataLoader: The data loader for the dataset
        """
        if path.split(data_path)[1] == "":
            # Deal with edge case where there's a "/" at the end of the path.
            data_path = path.split(data_path)[0]

        dataset = MapillaryDataset(data_path, augmentation)
        data_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True)

        return data_loader
