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

# Python built in imports
import time
from os import path

# Custom classes imports
from gui.training_gui import TrainingGUI
from utils.mapillarydataset import MappilaryDataset
from utils.plotcsv import PlotCSV


class Trainer:
    def __init__(self, lr=0.01, optimizer="sgd"):
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
        # classification
        self.criterion = nn.CrossEntropyLoss

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
        start_time = time.time()

        tracking = PlotCSV(plot_path, {"weights path": weights_path,
                                       "training data path": data_path,
                                       "learning rate": self.lr,
                                       "optimizer": self.optimizer,
                                       "batch size": batch_size,
                                       "epochs": num_epochs,
                                       "augmentation": augmentation})

        # Create GUI
        gui = TrainingGUI(num_epochs)

        # Load the dataset
        dataset = MappilaryDataset(data_path, augmentation)
        data_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True)

        gui.update_status("Dataset loaded.")

        # Load the state dictionary
        if path.isfile(weights_path):
            self.network.load_state_dict(torch.load(weights_path))
            gui.update_status("Loaded weights into state dictionary.")
        else:
            gui.update_status("Warning: Weights do not exist. "
                              "Running with random weights.")

        # Start training
        for epoch in range(num_epochs):
            gui.update_status("Starting training.")

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
                                                           non_blocking=True))

                # Zero out the optimizer
                self.optimizer.zero_grad()

                # Backprop and perform optimization
                loss.backward()
                self.optimizer.step()

                counter += 1

                # TODO Track accuracy
                accuracy = 0

                # Calculate time per step every 2 steps
                if counter % 2 == 0:
                    rate = float(counter)/(time.time() - start_time)

                loss_value = loss.item()

                gui.update_data(target=target_image[0],
                                generated=out.cpu().detach()[0],
                                step=data[0] + 1,
                                epoch=epoch,
                                accuracy=accuracy,
                                loss=loss_value,
                                rate=rate)

                # Write to the plot file every step
                tracking.write_data({"step": data[0] + 1,
                                     "loss": loss_value,
                                     "accuracy": accuracy})

        # Now save the loss and accuracy file
        tracking.close()

        # Save the weights
        torch.save(self.network.state_dict(),
                   weights_path)

        torch.cuda.empty_cache()

        gui.update_status("Done training.")
        gui.mainloop()
