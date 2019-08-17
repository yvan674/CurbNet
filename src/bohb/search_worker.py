"""Search Worker.

Searches for the optimal hyperparameters using HpBandSter, which is an
implementation of BOHB from the AutoML group at Uni Freiburg.

The hyperparameters we will be optimizing are as follows:

+----------------+----------------+------------------+------------------------+
| Parameter name | Parameter type |      Range       |        Comment         |
+----------------+----------------+------------------+------------------------+
| Learning rate  | float          | [1e-5, 1e-2]     | varied logarithmically |
| Optimizer      | categorical    | {'adam', 'sgd'}  | choose one             |
| SGD momentum   | float          | [0, 0.99]        | only active when       |
|                |                |                  | optimizer == 'sgd'     |
| Adam epsilon   | float          | [1e-2, 1]        | using step size 5e-2   |
| Sync batchnorm | bool           | {True, False}    | using synchronized     |
|                |                |                  | batchnorm or regular   |
|                |                |                  | batchnorm              |
| Loss weight    | float          | {2, 6}           |                        |
| ratio          |                |                  |                        |
| Loss criterion | categorical    | {'cross_entropy',|                        |
|                |                |  'mce'}          |                        |
+----------------+----------------+------------------+------------------------+

+-------------------+----------------+
|  Parameter name   | Name in config |
+-------------------+----------------+
| Learning Rate     | lr             |
| Optimizer         | optimizer      |
| SGD momentum      | momentum       |
| Adam epsilon      | epsilon        |
| Sync batchnorm    | sync_bn        |
| Loss weight ratio | weight_ratio   |
| Loss criterion    | loss_criterion |
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
from torch.nn import CrossEntropyLoss
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
from tqdm import tqdm
from datetime import datetime
from utils.plotcsv import PlotCSV

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class SearchWorker(Worker):
    def __init__(self, data_path, iaa, logging_dir, **kwargs):
        """Initializes the search worker.

        Args:
            data_path (str): Path to the data directory.
            iaa: The image augmentation module imported. This is necessary
                because sometimes, importing it here doesn't work but importing
                it in a different module does.
            logging_dir (str): Path to the logging directory. Used for logging
                configuration, loss, accuracy. Ideally, this is a subdirectory
                from the output directory.
            **kwargs:
        """
        super().__init__(**kwargs)

        self.iaa = iaa

        self.training_loader = self._load_dataset(
            path.join(data_path, "training"), True, BATCH_SIZE, self.iaa)
        self.validation_loader = self._load_dataset(
            path.join(data_path, "validation"), True, BATCH_SIZE, self.iaa)

        self.device = torch.device("cuda")
        self.run_count = 0

        self.data_path = data_path

        # Logging
        self.logging = PlotCSV()

        self.logging_dir = logging_dir

    def compute(self, config, budget, **kwargs):
        """Runs the training session.

        This training session will also save all the data on its runs (e.g.
        config, loss, accuracy) into the logging dir

        Args:
            config (dict): Dictionary containing the configuration by the
                optimizer
            budget (int): Amount of epochs the model can use to train.

        Returns:
            dict: dictionary with fields 'loss' (float) and 'info' (dict)
        """
        # Start with printouts
        print("\nStarting run {} with config:".format(self.run_count))
        print("    Start time:   {}"
              .format(datetime.now().strftime("%a, %-d %b at %H:%M:%S")))
        print("    lr:           {}".format(config['lr']))
        print("    optimizer:    {}".format(config['optimizer']))
        print("    sync_bn:      {}".format(config['sync_bn']))
        print("    weight ratio: {}".format(config['weight_ratio']))
        if config['optimizer'] == "adam":
            print("    epsilon:      {:.2f}".format(config['epsilon']))
        else:
            print("    momentum:     {}".format(config['momentum']))

        # Set network and loss criterion
        network = Parallelizer(CurbNetD(sync_bn=config['sync_bn'],
                                        px_coordinates=False).cuda())
        loss_weights = self._calculate_loss_weights(config['weight_ratio'])

        if config['loss_criterion'] == 'cross_entropy':
            criterion = CrossEntropyLoss(loss_weights)
        elif config['loss_criterion'] == 'mce':
            criterion = MCELoss(self._calculate_loss_weights(
                config['weight_ratio']))
        else:
            raise ValueError("Illegal loss criterion value used.")

        # Setup optimizers
        if config['optimizer'] == 'adam':
            optimizer = Adam(network.parameters(), lr=config['lr'],
                             eps=config['epsilon'])
        elif config['optimizer'] == 'sgd':
            optimizer = SGD(network.parameters(), lr=config['lr'],
                            momentum=config['momentum'])
        else:
            raise ValueError("Illegal optimizer value used.")

        # Prepare logging
        self.logging.reset()
        self.logging.configure({
            'run number': self.run_count,
            'plot path': self.logging_dir,
            'training data path': self.data_path,
            'optimizer': optimizer,
            'batch size': BATCH_SIZE,
            'epochs': budget,
            'network': network,
            'loss criterion': config['loss_criterion']
        })

        # Increment run count number
        self.run_count += 1

        csv_data = []  # To keep its scope outside of the for loop

        # Start actual training loop
        for epoch in range(int(budget)):
            for data in tqdm(enumerate(self.training_loader)):

                print('time: ', datetime.now() , 'epoch: ', epoch,
                      '  iteration: ', data[0], ' / ',
                      len(self.training_loader))

                # Make sure network is in train and optimizer has been zeroed
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

                # Calculate class-wise accuracy
                accuracy = np.array([0., 0., 0.])
                for idx, item in enumerate(target_image):
                    accuracy += calculate_accuracy(item, out[idx])
                accuracy / BATCH_SIZE

                # Append the data to a list first before writing to the file
                csv_data.append({"loss": loss.item(),
                                 "accuracy other": accuracy[0],
                                 "accuracy curb": accuracy[1],
                                 "accuracy curb cut": accuracy[2],
                                 "validation loss": ""})


                loss.backward()
                optimizer.step()

                # flush from memory
                del out
                del raw_image
                del target_image

        validation_accuracy, loss = self.evaluate_network(
            network, criterion, self.validation_loader)

        average_validation_acc = (validation_accuracy[0]
                                  + validation_accuracy[1]
                                  + validation_accuracy[2]) / 3.

        csv_data[-1]["validation loss"] = loss
        self.logging.write_data(csv_data)
        self.logging.close()

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

        print("\nFinished at {}"
              .format(datetime.now().strftime("%a, %-d %b at %H:%M:%S")))
        print("=====================================================")
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
                                                     lower=1.,
                                                     upper=6.,
                                                     default_value=5.)
        loss_criterion = CS.CategoricalHyperparameter('loss_criterion',
                                                      ['cross_entropy', 'mce'])

        cs.add_hyperparameters([lr, optimizer, momentum, epsilon, sync_bn,
                               weight_ratio, loss_criterion])
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
    def _load_dataset(data_path, augmentation, batch_size, iaa):
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

        dataset = MapillaryDataset(data_path, augmentation, iaa)
        data_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True)

        return data_loader
