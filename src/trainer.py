# -*- coding: UTF-8 -*-
"""Trainer.

This class implements the necessary functions to train the network.

Also creates a current status file. This file is to be able to check the status
of the program from the command line without needing any sort of communication
protocol. It works by just simply being a file that has the current state
written to every 10 steps. It's located as a file that always has the same name
within the plotting path.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
# Torch imports
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

# numpy
import numpy as np

# Python built in imports
import time
from time import strftime, localtime
from os import path
import datetime
import warnings

# Custom classes imports
from ui.training_gui import TrainingGUI
from ui.training_cmd import TrainingCmd
from utils.mapillarydataset import MapillaryDataset
from utils.plotcsv import PlotCSV

# network imports
from network.parallelizer import Parllelizer as Network
from network.curbnet_e import CurbNetE
from network.curbnet_f import CurbNetF
from network.curbnet_g import CurbNetG

# import network.postprocessing as post


class Trainer:
    def __init__(self, lr: float=0.01, optimizer: str="sgd",
                 loss_weights: list=None, cmd_line: bool=False,
                 network: str="g") -> None:
        """Training class used to train the CurbNetG network.

        Args:
            lr (float): Learning rate of the network
            optimizer (str): The optimizer to be used. Defaults to sgd.
            Possible optimizers are:
                - sgd: Stochastic Gradient Descent.
                - Adam: Adaptive moment estimation.
            loss_weights (list): A list of floats with length 3 that are the
            weight values for each class used by the loss function.
            cmd_line (bool): Whether or not to use the command line interface.
                             Defaults to False.
            network (str): The network to be used. Defaults to GoogLeNet.
            Possible options are:
                - e: ENET based network.
                - f: FCN based network.
                - g: GoogLeNet based encoder network.
        """
        if loss_weights is None:
            # To avoid mutable default values
            loss_weights = [0.00583, 0.49516, 0.49902]

        # Set up the different networks
        if network == "e":
            self.px_coordinates = False
            network = CurbNetE()
        elif network == "f":
            self.px_coordinates = True
            network = CurbNetF()
        elif network == "g":
            # G is the only network so far working with pixel coordinates
            self.px_coordinates = True
            network = CurbNetG()

        # Initialize the network
        self.network = Network(network)

        # Parameters
        self.lr = lr,
        self.optimizer = optimizer

        if torch.cuda.is_available():
            # Check if cuda is available and use it if it is
            self.device = torch.device("cuda")
            self.network.cuda()
        else:
            self.device = torch.device("cpu")
            warnings.warn("CUDA compatible GPU not found. Running on CPU",
                          ResourceWarning)

        # Set the loss criterion according to the recommended for pixel-wise
        # classification. We use weights so that missing curbs
        # will be more heavily penalized
        loss_weights = torch.tensor(loss_weights,
                                    dtype=torch.float).to(device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=loss_weights)

        # Set the optimizer according to the arguments
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)
        else:
            raise ValueError("Illegal optimizer value: only SGD and Adam "
                             "optimizers are currently supported.")

        # Set the network to train
        self.network.train()

        # Create UI
        if cmd_line:
            self.ui = TrainingCmd()
        else:
            self.ui = TrainingGUI()

        self.cmd_line = cmd_line

        # Set the status file path variable
        self.status_file_path = None

        # Creates logging tracker
        self.tracker = PlotCSV()

    def train(self, data_path, batch_size, num_epochs, plot_path, weights_path,
              augmentation=True):
        """Start training the network.

        Args:
            data_path (str): Path to the specific dataset directory.
            batch_size (int): Number of images to run per batch.
            num_epochs (int): Number of epochs to run the trainer for.
            plot_path (str): Path to save the loss plot. This should be a
                             directory.
            weights_path (str): The path to the weights.
            augmentation (bool): Whether or not to use augmentation. Defaults to
                                 True.

        Returns:
            None
        """
        # Stat variables
        counter = 0
        rate = 0

        self.tracker.configure({
            "plot path": plot_path,
            "weights path": weights_path,
            "training data path": data_path,
            "optimizer": self.optimizer,
            "batch size": batch_size,
            "epochs": num_epochs,
            "augmentation": augmentation,
            "network": self.network
        })

        # Set up the status file
        self.status_file_path = path.join(plot_path, "status.txt")


        # Load the dataset
        start_time = time.time()
        dataset = MapillaryDataset(data_path, augmentation, self.px_coordinates)
        data_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True)

        self._update_status("Dataset loaded. ({} ms)".format(
            int(time.time() - start_time) * 1000))

        # Load the state dictionary
        start_time = time.time()
        if path.isfile(weights_path):
            self.network.load_state_dict(torch.load(weights_path))
            self._update_status("Loaded weights into state dictionary. ({} ms)"
                                .format(int(time.time() - start_time) * 1000))
        else:
            self._update_status("Warning: Weights do not exist. "
                                "Running with random weights.")

        # Start training
        start_time = time.time()
        absolute_start_time = time.time()
        self._update_status("Starting training on {} GPU(s)."
                            .format(torch.cuda.device_count()))
        for epoch in range(num_epochs):
            # Figure out number of max steps for info displays
            self.ui.set_max_values(len(data_loader), num_epochs)

            for data in enumerate(data_loader):
                # Grab the raw and target images
                raw_image = data[1]["raw"]
                target_image = data[1]["segmented"]

                # Run the network, but make sure the tensor is in the right
                # format
                out = self.network(raw_image.to(self.device, non_blocking=True))

                # Make sure it's been run properly
                if out is None:
                    raise ValueError("forward() has not been run properly.")

                # Since out is used multiple times, we detach it once
                detached_out = out.cpu().detach()

                # Start post processing
                # Create an argmax version here to avoid making it in the GUI
                # class and since we need it for crf as well
                d_out_argmax = torch.argmax(detached_out, dim=1)

                # Calculate loss, converting the tensor if necessary
                loss = self.criterion(out, target_image.to(self.device,
                                                           dtype=torch.long,
                                                           non_blocking=True))

                # Zero out the optimizer
                self.optimizer.zero_grad()

                # Backprop and perform optimization
                loss.backward()
                self.optimizer.step()

                counter += 1



                # Calculate accuracy
                accuracy = self._calculate_batch_accuracy(target_image,
                                                          detached_out,
                                                          batch_size)

                # Calculate time per step every 2 steps
                if counter % 2 == 0:
                    rate = float(counter) / (time.time() - start_time)

                loss_value = loss.item()

                self.ui.update_data(step=data[0] + 1,
                                    epoch=epoch + 1,
                                    accuracy=accuracy,
                                    loss=loss_value,
                                    rate=rate,
                                    status_file_path=self.status_file_path)

                if not self.cmd_line:
                    self.ui.update_image(target=target_image[0],
                                         generated=d_out_argmax[0])

                # Write to the plot file every step
                self.tracker.write_data({"loss": loss_value,
                                         "accuracy": accuracy})

        # Save the weights
        torch.save(self.network.state_dict(),
                   weights_path)

        torch.cuda.empty_cache()

        self._update_status(
            "Finished training in {}.".format(datetime.timedelta(
                seconds=int(time.time() - absolute_start_time))))

        # Now save the loss and accuracy file
        self.tracker.close()

        if not self.cmd_line:
            self.ui.mainloop()

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
        accuracy = 0
        for idx, item in enumerate(ground_truth):
            accuracy += self._calculate_accuracy(item, predicted[idx])

        return accuracy / batch_size

    @staticmethod
    def _calculate_accuracy(ground_truth, predicted):
        """Calculate the intersection over union accuracy of the segmentation."

        Notes:
            Also tried a function where the union was calculated by getting the
            maximum array (i.e. np.maximum(predicted, ground_truth) and the
            zeros were calculated using np.count_nonzero(max_array == 0) but
            after benchmarking, it was found that this method was approximately
            8-10% slower than the currently implemented algorithm.

        Args:
            ground_truth (torch.*Tensor): The ground truth data.
            predicted (torch.*Tensor): The predicted segmentation generated by
                                       the network.

        Returns:
            float: Accuracy of the segmentation based on intersection over
            union accuracy.
        """
        # First turn the arrays into numpy arrays for processing
        ground_truth = ground_truth.numpy()
        predicted = predicted.numpy()

        # Then get the argmax value from the prediction
        predicted = predicted.transpose((1, 2, 0))
        predicted = predicted.argmax(axis=2)

        # Calculate union
        union = np.count_nonzero(np.maximum(ground_truth, predicted))

        # Get zeros
        zeros = (ground_truth.shape[0] * ground_truth.shape[1]) - union

        # Calculate intersect
        intersect = np.sum(ground_truth == predicted) - zeros

        # Return intersect over union, with special handling for union = 0
        if union == 0:
            if intersect == 0:
                return 1.
            else:
                return 0.
        else:
            return intersect / union

    def _update_status(self, message):
        """Updates the status of the program in the GUI, console, and log.

        Args:
            message (str): The message to be written.
            ui (ui.TrainingUI): The UI object used.
            plot_csv (utils.plotcsv.PlotCSV): The PlotCSV object used
        """
        message = "[{}] {}".format(strftime("%H:%M:%S", localtime()), message)
        self.tracker.write_log(message)
        self.ui.update_status(message)
