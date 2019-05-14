"""Training Command Line.

Enables training through a command-line interface. This allows for training to
be done through the command line, e.g. when using SSH.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from ui.training_ui import TrainingUI
import datetime
try:
    import curses
except ImportError:
    pass


class TrainingCmd(TrainingUI):
    def __init__(self):
        """Initializes a curses based training UI."""
        self.max_step = 0
        self.max_epoch = 0
        self.stdscr = curses.initscr()
        self.stdscr.clear()

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
        # Calculate time left
        if rate == 0:
            time_left = "NaN"
        else:
            time_left = int(((self.max_step * self.max_epoch)
                             - ((float(step) + 1.)
                                + (self.max_step * epoch))) / rate)
            time_left = str(datetime.timedelta(seconds=time_left))

        # Clear only the top part of the screen
        for i in range(3):
            self.stdscr.addstr(i, 0,
                               "                              "
                               "                              ")

        self.stdscr.addstr(0, 0, "Step: {} / {}".format(step, self.max_step))
        self.stdscr.addstr(0, 30, "Epoch: {}/ {}".format(epoch, self.max_epoch))

        self.stdscr.addstr(1, 0, "Loss: {:.3f}".format(loss))
        self.stdscr.addstr(1, 30, "Accuracy: {:.3f}%".format(accuracy))

        self.stdscr.addstr(2, 0, "Rate: {:.3f} Steps/s".format(rate))
        self.stdscr.addstr(2, 30, "Time left: {}".format(time_left))

    def update_status(self, message):
        """Updates the status message within the UI.

        Args:
            message (str): The new message that should be displayed.
        """
        self.stdscr.addstr(3, 0, message)

    def set_max_values(self, total_steps, total_epochs):
        """Sets the number of steps and epochs during this training session.

        Args:
            total_steps (int): The total number of steps.
            total_epochs (int): The total number of epochs.
        """
        self.max_step = total_steps
        self.max_epoch = total_epochs
