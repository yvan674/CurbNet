"""Training Command Line.

Enables training through a command-line interface. This allows for training to
be done through the command line, e.g. when using SSH.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from ui.training_ui import TrainingUI


class TrainingCmd(TrainingUI):
    def __init__(self):
        """Initializes a curses based training UI."""
        self.max_steps = 0
        self.max_epochs = 0

    def update_data(self, step, epoch, accuracy, loss, rate):
        """Updates the strings within the UI.

        Args:
            step (int): The current step of the training process.
            epoch (int): The current epoch of the training process.
            accuracy (float): The accuracy of the Network at the current step.
            loss (float): The loss of the Network at the current step.
            rate (float): The rate the Network is running at in steps per
                          second.
        """
        print("Step: {}/{}, Epoch: {}/{}, Rate: 0 S/s, Time left: 0 seconds")

    def update_status(self, message):
        """Updates the status message within the UI.

        Args:
            message (str): The new message that should be displayed.
        """
        pass

    def set_max_values(self, total_steps, total_epochs):
        """Sets the number of steps and epochs during this training session.

        Args:
            total_steps (int): The total number of steps.
            total_epochs (int): The total number of epochs.
        """
        self.max_steps = total_steps
        self.max_epochs = total_epochs
