# -*- coding: utf-8 -*-
"""PlotIt.

This module visualizes the plot of the loss function created while training the
Network.

Authors:
    Maximilian Roth <max@die-roths.de>
    Nina Pant
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import argparse
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename
import csv


def parse_args():
    """Parses command line arguments."""
    description = "Plots loss data from DriveNet"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('path', metavar='P', type=str, nargs='?',
                        help='path of the loss data to be plotted.')
    return parser.parse_args()


class PlotIt:
    def __init__(self, plot_location=None):
        """Generates plot of loss function from .csv file."""
        if plot_location is None:
            root = tk.Tk()
            root.withdraw()
            file_name = askopenfilename()
            root.destroy()
        else:
            file_name = plot_location

        loss_data = []
        acc_data = []

        # Read csv data
        with open(file_name, "r") as csv_file:
            data_reader = csv.reader(csv_file)
            # skip the header
            next(data_reader, None)
            for row in data_reader:
                loss_data.append(float(row[0]))
                acc_data.append(float(row[1]))

        # Instantiate the subplots
        fig, ax1 = plt.subplots()

        # Set window title
        fig.canvas.set_window_title('Loss and Accuracy Plot')

        # Setup the first axis
        color = 'tab:red'
        ax1.set_xlabel('step number')
        ax1.set_ylabel('loss', color=color)
        ax1.plot(loss_data, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        # Instantiate a second axes that shares the x-axis
        ax2 = ax1.twinx()

        color = 'tab:blue'
        # X label not necessary due to sharing it with ax1
        ax2.set_ylabel('accuracy',
                       color=color)
        ax2.plot(acc_data, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Tight layout so ax2 label isn't clipped
        fig.tight_layout()

        # Show it
        plt.show()


if __name__ == "__main__":
    print("hi im nina")
    arguments = parse_args()
    PlotIt(arguments.path)
