# -*- coding: UTF-8 -*-
"""Trainer.

This class implements the necessary functions to train the network.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
# Network import
from curbnet import CurbNet

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

# Custom classes imports
from gui.training_gui import TrainingGUI
from utils.mapillarydataset import MappilaryDataset
from utils.plotcsv import PlotCSV


class Trainer:
    def __init__(self, lr=0.01, optimizer="sgd", loss_weights=None):
        """Training class used to train the CurbNet network.

        Args:
            lr (float): Learning rate of the network
            optimizer (str): The optimizer to be used.
            loss_weights (list): A list of floats with length 3 that are the
                                 weight values for each class used by the loss
                                 function.
        """
        if loss_weights is None:
            # To avoid mutable default values
            loss_weights = [0.05, 5., 8.]

        # Initialize the network
        self.network = CurbNet()

        # Parameters
        self.lr = lr,
        self.optimizer = optimizer

        if torch.cuda.is_available():
            # Check if cuda is available and use it if it is
            self.device = torch.device("cuda")
            self.network.cuda()
        else:
            self.device = torch.device("cpu")
            print("Warning: CUDA compatible GPU not found. Running on CPU")

        # Set the loss criterion according to the recommended for pixel-wise
        # classification. We use weights so that missing curbs
        # will be more heavily penalized
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(loss_weights).cuda())

        # Set the optimizer according to the arguments
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)
        else:
            raise ValueError("Illegal optimizer value: only SGD and Adam"
                             "optimizers are currently supported.")

        # Set the network to train
        self.network.train()

    def train(self, data_path, batch_size, num_epochs, plot_path, weights_path,
              augmentation):
        """Start training the network.

        Args:
            data_path (str): Path to the specific dataset directory.
            batch_size (int): Number of images to run per batch.
            num_epochs (int): Number of epochs to run the trainer for.
            plot_path (str): Path to save the loss plot. This should be a
                             directory.
            weights_path (str): The path to the weights,
            augmentation (bool): Whether or not to use augmentation,
        """
        # Stat variables
        counter = 0
        rate = 0

        tracking = PlotCSV(plot_path, {"weights path": weights_path,
                                       "training data path": data_path,
                                       "optimizer": self.optimizer,
                                       "batch size": batch_size,
                                       "epochs": num_epochs,
                                       "augmentation": augmentation})

        # Create GUI
        gui = TrainingGUI(num_epochs)

        # Load the dataset
        start_time = time.time()
        dataset = MappilaryDataset(data_path, augmentation)
        data_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True)

        self._update_status("Dataset loaded. ({} ms)".format(
            int(time.time() - start_time) * 1000), gui, tracking)

        # Load the state dictionary
        start_time = time.time()
        if path.isfile(weights_path):
            self.network.load_state_dict(torch.load(weights_path))
            self._update_status("Loaded weights into state dictionary. ({} ms)"
                                .format(int(time.time() - start_time) * 1000),
                                gui, tracking)
        else:
            self._update_status("Warning: Weights do not exist. "
                                "Running with random weights.", gui, tracking)

        # Start training
        start_time = time.time()
        absolute_start_time = time.time()
        self._update_status("Starting training.", gui, tracking)
        for epoch in range(num_epochs):
            # Figure out number of max steps for info displays
            gui.set_max_step(len(data_loader))

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

                # Since out is used multiple times, we detach it once
                detached_out = out.cpu().detach()

                # Calculate accuracy
                accuracy = self._calculate_batch_accuracy(target_image,
                                                          detached_out,
                                                          batch_size)

                # Calculate time per step every 2 steps
                if counter % 2 == 0:
                    rate = float(counter)/(time.time() - start_time)

                loss_value = loss.item()

                gui.update_data(target=target_image[0],
                                generated=detached_out[0],
                                step=data[0] + 1,
                                epoch=epoch,
                                accuracy=accuracy,
                                loss=loss_value,
                                rate=rate)

                # Write to the plot file every step
                tracking.write_data({"loss": loss_value,
                                     "accuracy": accuracy})

        # Save the weights
        torch.save(self.network.state_dict(),
                   weights_path)

        torch.cuda.empty_cache()

        self._update_status(
            "Finished training in {}."
                .format(datetime.timedelta(seconds=int(time.time()
                                                   - absolute_start_time))),
            gui, tracking)

        # Now save the loss and accuracy file
        tracking.close()

        gui.mainloop()

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

    def _update_status(self, message, gui, plot_csv):
        """Updates the status of the program in the GUI, console, and log.

        Args:
            message (str): The message to be written.
            gui (gui.training_gui.TrainingGUI): The GUI object used.
            plot_csv (utils.plotcsv.PlotCSV): The PlotCSV object used
        """
        message = "[{}] {}".format(strftime("%H:%M:%S", localtime()), message)
        plot_csv.write_log(message)
        gui.update_status(message)

    @staticmethod
    def _human_time_duration(start_time, end_time=time.time()):
        """Calculates and returns time delta in a human readable format.

        Args:
            start_time (float): Starting time.
            end_time (float): End time. Defaults to now.

        Returns:
            str: The time delta in a human readable format.
        """
        td = int(end_time - start_time)
        days = td // 86400

        td -= (days * 86400)
        hours =  td // 3600

        td -= (hours * 3600)
        minutes = td // 60

        td -= (minutes * 60)
        seconds = int(td)

        return_str = ""
        if days > 0:
            return_str += "{} ".format(days)
            if days > 1:
                return_str += "days "
            else:
                return_str += "day "

        if hours > 0:
            return_str += "{} ".format(hours)
            if hours > 1:
                return_str += "hours "
            else:
                return_str += "hour "

        if minutes > 0:
            return_str += "{} ".format(minutes)
            if minutes > 1:
                return_str += "minutes "
            else:
                return_str += "minute"

        if seconds > 0:
            return_str += "{} ".format(seconds)
            if seconds > 1:
                return_str += "seconds"
            else:
                return_str += "second"

        return return_str
