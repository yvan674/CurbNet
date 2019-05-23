"""Training UI.

This is the abstract class for all training UIs. Any versions of the training UI
will inherit from this class to allow interchangeability of the UI between using
command line and a tkinter GUI.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com
"""
from abc import ABC, abstractmethod


class TrainingUI(ABC):
    @abstractmethod
    def update_data(self, step, epoch, accuracy, loss, rate, status_file_path):
        """Updates the strings within the UI.

        Args:
            step (int): The current step of the training process.
            epoch (int): The current epoch of the training process.
            accuracy (float): The accuracy of the network at the current step.
            loss (float): The loss of the network at the current step.
            rate (float): The rate the network is running at in steps per
                          second.
            status_file_path (str): The path to save the status file to.
        """
        pass

    @abstractmethod
    def update_status(self, message):
        """Updates the status message within the UI.

        Args:
            message (str): The new message that should be displayed.
        """
        pass

    @abstractmethod
    def set_max_values(self, total_steps, total_epochs):
        """Sets the number of steps and epochs during this training session.

        Args:
            total_steps (int): The total number of steps.
            total_epochs (int): The total number of epochs.
        """
