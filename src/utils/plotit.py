# -*- coding: utf-8 -*-
"""PlotIt.

This module visualizes the plot of the loss function created while training the
network.

Authors:
    Maximilian Roth <max@die-roths.de>
    Nina Pant
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import argparse
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import csv
import numpy as np


def parse_args():
    """Parses command line arguments."""
    description = "Plots loss data from DriveNet"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-l', '--loss', action='store_false',
                        help='do not display the loss')
    parser.add_argument('-a', '--accuracy', action='store_false',
                        help='do not display the accuracy')
    parser.add_argument('-v', '--validation-loss', action='store_false',
                        help='do not display the validation loss')
    parser.add_argument('path', metavar='P', type=str, nargs='?',
                        help='path of the loss data to be plotted.')
    parser.add_argument('period', metavar='T', type=int, nargs='?',
                        help='period of the moving average calculation')
    return parser.parse_args()


class PlotIt:
    def __init__(self, period, loss_bool=True, accuracy_bool=True,
                 validation_bool=True, plot_location=None):
        """Generates plot of loss function from .csv file.

        The plot gives the moving average over a number of iterations of both
        the loss as well as the accuracy.

        The data should have loss and accuracy at every line. If the data has
        validation loss, then the validation loss at each iteration should be
        blank except for when validation loss is actually calculated. When this
        happens, then that single row should have 3 values: Loss, accuracy, and
        validation loss.

        Args:
            period (int): The period of the moving average calculation. Defaults
                to 100.
            loss_bool (bool): Whether or not to plot the loss. Defaults to True.
            accuracy_bool (bool): Whether or not to plot the accuracy. Defaults
                to True.
            validation_bool (bool): Whether or not to plot the validation loss.
                Defaults to True.
            plot_location (str): The path of the loss file to be plotted. Is
                none is given, then a file selector window will open. Defaults
                to None.
        """
        if period is None or period == "":
            period = 100

        if plot_location is None:
            root = tk.Tk()
            root.withdraw()
            file_names = askopenfilenames(filetypes=[('CSV files', '*.csv')])
            file_names = root.tk.splitlist(file_names)
            root.destroy()
        else:
            if type(plot_location) != list:
                file_names = [plot_location]
            else:
                file_names = plot_location

        if file_names[0] == "":
            print("No file selected. Exiting program.")
            return

        loss_data = []
        acc_data = np.array([[], [], []])
        has_validation_loss = False
        validation_loss = []
        validation_index = []

        # Read csv data
        index_counter = 0
        for file in file_names:
            with open(file, "r") as csv_file:
                # Get csv as a list to iterate through
                data = list(csv.reader(csv_file))

                # If data has 3 columns, i.e. has validation loss
                has_validation_loss = True
                for i in range(1, len(data)):
                    # using for i in range since we want the index
                    loss_data.append(float(data[i][0]))
                    acc_data[0].append(float(data[i][1]))
                    acc_data[1].append(float(data[i][2]))
                    acc_data[2].append(float(data[i][3]))
                    if data[i][4] != '':
                        # If the data has a validation loss value in that
                        # row
                        validation_loss.append(float(data[i][4]))
                        validation_index.append(index_counter)
                    index_counter += 1

        # Calculate their moving averages, ma = moving average
        ma_loss, ma_loss_idx = self._calculate_moving_average(loss_data, period)

        ma_acc = [0., 0., 0.]

        for i in range(3):
            ma_acc[i], ma_acc_idx = self._calculate_moving_average(acc_data[i],
                                                                   period)

        # Turn accuracy to percentage
        ma_acc = np.array(ma_acc)
        ma_acc = ma_acc * 100
        # Instantiate the subplots
        fig, loss_axis = plt.subplots(sharey="col")

        # Set window title
        fig.canvas.set_window_title('Loss and Accuracy Plot')

        # Setup colors
        loss_color = 'red'
        acc_color = ['blue', 'green', 'yellow']


        # Setup the first axis
        loss_axis.set_xlabel('step number')
        loss_axis.set_ylabel('loss')

        # Instantiate a second axes that shares the x-axis
        acc_axis = loss_axis.twinx()

        # Setup the second axis
        # X label not necessary due to sharing it with loss_axis
        acc_axis.set_ylabel('accuracy (%)')

        # Set up legend
        lines = []
        labels = []
        # Plot in order
        if accuracy_bool:
            for i in range(3):
                acc_line, = acc_axis.plot(ma_acc_idx,
                                          ma_acc[i],
                                          color=acc_color[i])
                lines.append(acc_line)
                labels.append("Class {} acc".format(i))
        if loss_bool:
            loss_line, = loss_axis.plot(ma_loss_idx,
                                        ma_loss,
                                        color=loss_color)
            lines.append(loss_line)
            labels.append("Training Loss")

        # Add validation loss to legend if the data has validation loss
        if has_validation_loss and validation_bool:
            validation_color = 'green'
            validation_line, = loss_axis.plot(validation_index,
                                              validation_loss,
                                              color=validation_color)
            lines.append(validation_line)
            labels.append("Validation Loss")

        fig.legend(lines, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.96))

        # Tight layout so acc_axis label isn't clipped
        fig.tight_layout()

        # Show it
        plt.show()

    @staticmethod
    def _calculate_moving_average(list, period):
        """Calculates the moving average given a list of values.

        Args:
            list (list): A list of numbers.
            period (int): The period of the moving average.

        Returns:
            list: The moving averages.
            list: The indices of each moving average
        """
        # Set the first values
        ma = []
        indices = []

        if len(list) <= period:
            # If the list is too short, set the first object as the starting
            # value
            ma.append(list.pop(0))
            indices.append(0)

        # Iterate through the list
        counter = 0  # Count resets each period
        total_counter = 0  # Counts total to check list length and index
        sum = 0
        for value in list:
            sum += value
            counter += 1
            total_counter += 1

            if counter == period:
                # At each period, or the end of the list, count average and
                # reset counters
                ma.append(sum / period)
                indices.append(total_counter)
                sum = 0
                counter = 0
            elif total_counter == len(list):
                ma.append(sum / counter)
                indices.append(total_counter)
        return ma, indices

if __name__ == "__main__":
    print("hi im nina")
    args = parse_args()
    PlotIt(args.period, args.loss, args.accuracy, args.validation_loss,
           args.path)
