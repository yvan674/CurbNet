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

        # String variables
        self.step_var = ""
        self.epoch_var = ""
        self.loss_var = ""
        self.acc_var = ""
        self.rate_var = ""
        self.time_var = ""
        self.status_var = ""

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
        # Calculate time left
        # Calculate time left
        if rate == 0:
            time_left = "NaN"
        else:
            steps_total = float((self.max_step * self.max_epoch))
            steps_done_this_epoch = float(step + 1)
            steps_times_epochs_done = float(self.max_step * (epoch - 1))
            steps_left = (steps_total - steps_done_this_epoch
                          - steps_times_epochs_done)

            time_left = int(steps_left / rate)
            time_left = str(datetime.timedelta(seconds=time_left))

        self.step_var = "Step: {} / {}".format(step, self.max_step)
        self.epoch_var = "Epoch: {}/ {}".format(epoch, self.max_epoch)
        self.loss_var = "Loss: {:.3f}".format(loss)
        self.acc_var = "Accuracy: {:.3f}%".format(accuracy)
        self.rate_var = "Rate: {:.3f} Steps/s".format(rate)
        self.time_var = "Time left: {}".format(time_left)

        # Now write the status file
        if step % 10 == 0:
            with open(status_file_path, 'w') as status_file:
                lines = ["Step: {}\n".format(step),
                         "Epoch: {}\n".format(epoch),
                         "Accuracy: {}\n".format(accuracy),
                         "Loss: {:.3f}\n".format(loss),
                         "Rate: {:.3f} steps/s\n".format(rate),
                         "Time left: {}\n".format(time_left)]

                status_file.writelines(lines)

        if step == self.max_step:
            with open(status_file_path, 'w') as status_file:
                status_file.write("Finished training.")

        self._update_screen()

    def update_status(self, message):
        """Updates the status message within the UI.

        Args:
            message (str): The new message that should be displayed.
        """
        self.status_var = message
        self._update_screen()

    def set_max_values(self, total_steps, total_epochs):
        """Sets the number of steps and epochs during this training session.

        Args:
            total_steps (int): The total number of steps.
            total_epochs (int): The total number of epochs.
        """
        self.max_step = total_steps
        self.max_epoch = total_epochs

    def _update_screen(self):
        """Updates the screen with the current string variables."""
        # Clear only the top part of the screen
        for i in range(4):
            self.stdscr.addstr(i, 0,
                               "                              "
                               "                              "
                               "                              ")
        self.stdscr.addstr(0, 0, self.step_var)
        self.stdscr.addstr(0, 30, self.epoch_var)

        self.stdscr.addstr(1, 0, self.loss_var)
        self.stdscr.addstr(1, 30, self.acc_var)

        self.stdscr.addstr(2, 0, self.rate_var)
        self.stdscr.addstr(2, 30, self.time_var)
        self.stdscr.addstr(3, 0, self.status_var)
        self.stdscr.refresh()
