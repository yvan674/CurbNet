"""Plot CSV.

This module is responsible for tracking loss and accuracy and writing that
information to a CSV file in a specified location. The reason this is a separate
module is to ensure that no code has to be rewritten and to ensure a more robust
and modular approach to the architecture.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import os
import csv
from time import strftime, gmtime
from utils.plotit import PlotIt


class PlotCSV:
    def __init__(self):
        """Writes training parameters and the loss and accuracy data to a CSV.
        """
        # Initialize variables
        self.log_file = None
        self.file_path = None
        self.csv_file = None
        self.csv_writer = None
        self.configured = False

        # Create a queue of lines to be written to avoid hdd writes every line
        self.queued_lines = []

    def configure(self, parameters):
        """Configures the parameters for the module.

        Args:
            parameters (dict): Required parameters:
                - "plot path" (str): Where to put the plot file.
                - "weights path" (str): Where the weights will be saved.
                - "training data path" (str): Where the training data is.
                - "optimizer" (str): The optimizer used to train.
                - "batch size" (int): The size of the batch in this session.
                - "epochs" (int): Number of epochs to train for.
                - "augmentation" (bool): Whether or not to use augmentation.
        """
        # Reassign the plot path
        directory = parameters["plot path"]
        # Remove it from the parameters list so it isn't printed to the log.
        parameters.pop("plot path")

        # Create the directory if it doesn't exist
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # Set the file location to the current time and date for future
        # referencing
        current_time = strftime("%Y_%m_%d_%H-%M-%S", gmtime())

        # Set up the log file
        log_path = os.path.join(directory,
                                current_time + '-log.txt')
        self.log_file = open(log_path, 'a')

        # Write out the parameters of this test.
        self.log_file.write("Training Parameters:\n====================\n")
        for key in parameters.keys():
            self.log_file.write("{}: {}\n".format(key, parameters[key]))

        self.log_file.write("\n\nTraining Log\n============\n")

        # Prepare csv file
        self.file_path = os.path.join(directory, current_time
                                      + '-loss_data.csv')
        self.csv_file = open(self.file_path, 'a', newline='')
        # Create the writer for the csv
        self.csv_writer = csv.DictWriter(self.csv_file, dialect=csv.excel,
                                         fieldnames=["loss", "accuracy"],
                                         restval="", extrasaction="ignore")

        self.csv_writer.writeheader()
        self.configured = True

    def write_data(self, data):
        """Queues data for writing to the CSV file.

        Args:
            data (dict): Data to be written to the CSV file.
        """
        if not self.configured:
            raise Exception("PlotCSV has not been properly configured.")

        self.queued_lines.append(data)

        # Writes every 10 lines to reduce hdd dependency time
        if len(self.queued_lines) == 10:
            self.csv_writer.writerows(self.queued_lines)
            self.queued_lines = []

    def write_log(self, message):
        """Writes a message to the log.

        Args:
            message (str): The message to be written
        """
        if not self.configured:
            raise Exception("PlotCSV has not been properly configured.")

        self.log_file.write("{}\n".format(message))

    def close(self):
        """Writes any last lines, closes open files, and shows the plot."""
        # Deal with the csv file
        self.csv_writer.writerows(self.queued_lines)
        self.csv_file.close()

        # Deal with the log file
        self.log_file.close()

        # Open the plot
        PlotIt(self.file_path)
