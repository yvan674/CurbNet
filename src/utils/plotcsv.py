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
    def __init__(self, directory, parameters):
        """Writes training parameters and the loss and accuracy data to a CSV.

        Args:
            directory (str): The directory that the data should be saved in.
            parameters (dict): The training hyper parameters to be stored.
        """
        # Create the directory if it doesn't exist
        if not os.path.isdir(directory):
            os.mkdir(directory)

        # Set the file location to the current time and date for future
        # referencing
        current_time = strftime("%Y_%m_%d_%H-%M-%S", gmtime())
        self.file_path = os.path.join(directory, current_time
                                      + '-loss_data.csv')

        param_path = os.path.join(directory, current_time + '-parameters.txt')

        # Write out the parameters of this test.
        with open(param_path, 'w') as params:
            for key in parameters.keys():
                params.write("{}: {}\n".format(key, parameters[key]))

        self.csv_file = open(self.file_path, 'a', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, dialect=csv.excel,
                                         fieldnames=["loss", "accuracy"],
                                         restval="", extrasaction="ignore")

        self.csv_writer.writeheader()

        self.queued_lines = []

    def write_data(self, data):
        """Queues data for writing to the CSV file.

        Args:
            data (dict): Data to be written to the CSV file.
        """
        self.queued_lines.append(data)

        # Writes every 10 lines to reduce hdd dependency time
        if len(self.queued_lines) == 10:
            self.csv_writer.writerows(self.queued_lines)
            self.queued_lines = []

    def close(self):
        """Writes any last lines, closes the csv file, and shows the plot."""
        self.csv_writer.writerows(self.queued_lines)
        self.csv_file.close()
        PlotIt(self.file_path)
