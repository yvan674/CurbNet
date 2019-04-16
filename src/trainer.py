# -*- coding: UTF-8 -*-
"""Trainer.

This class implements the necessary functions to train the network.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""

from curbnet import CurbNet
import torch
import torch.nn as nn
from time import strftime, gmtime
from os import path
from gui.training_gui import TrainingGUI
import time


class Trainer:
    def __init__(self, lr=0.01, optimizer="sgd"):
        self.network = CurbNet()

        if torch.cuda.is_available():
            # Check if cuda is available and use it if it is
            self.device = torch.device("cuda")
            self.network.cuda()
        else:
            self.device = torch.device("cpu")
            print("Warning: CUDA compatible GPU not found. Running on CPU")

        # Set the loss criterion according to the recommended for pixel-wise
        # classification
        self.criterion = nn.CrossEntropyLoss

        # # Set the optimizer according to the arguments
        # if optimizer == "adam":
        #     self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        # elif optimizer == "sgd":
        #     self.optimizer = torch.optim.SGD(self.network.parameters(), lr=lr)
        # else:
        #     raise ValueError("Illegal optimizer value: only SGD and Adam"
        #                      "optimizers are currently supported.")

        # Set the network to train
        self.network.train()

    def train(self, data_path, batch_size, num_epochs, plot_path, save_path):
        """Start training the network.

        Args:
            data_path (str): Path to the specific dataset directory.
            batch_size (int): Number of images to run per batch.
            num_epochs (int): Number of epochs to run the trainer for.
            plot_path (str): Path to save the loss plot. This should be a
                             directory.
            save_path (str): The path to the weights
        """
        counter = 0

        # Plot save location. This is a plot of the accuracy and loss over time.
        # The plot should be saved every 10 batches
        plot_path = path.join(plot_path, strftime("%Y_%m_%d_%H-%M-%S", gmtime())
                             + '-loss_data.csv')

        # Create GUI
        gui = TrainingGUI(num_epochs)

        # Load the parameters list



