"""Training GUI.

This module allows training to be displayed through a GUI. This makes it easier
to see what's going on and is just nicer overall compared to training a
segmentation network over the command line. The purpose of the module is to
show the ground truth and the generated segmentation, the status of the training
session, and a plot of the current loss vs accuracy of the network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
import os


class TrainingGUI:
    def __init__(self, total_epoch):
        """Creates a GUI to show training status.

        Args:
            total_epoch (int): How many epochs are going to be done.
        """
        # Assign variables
        self.total_steps = 0
        self.total_epochs = total_epoch

        # Start by making the tk components
        self.root = tk.Tk()
        self.root.title("CurbNet Training")
        self.root.geometry("714x630")

        # Configure the grid and geometry
        # ---------------------------------------------------------
        # |                                | step     | epoch     |
        # |       Target segmentation      | loss     | accuracy  |
        # |                                | rate     | time left |
        # |                                | status message       |
        # |-------------------------------------------------------|
        # |             Output             | loss/acc plot        |
        # ---------------------------------------------------------
        self.root.columnconfigure(0, minsize=414)
        self.root.columnconfigure(1, minsize=150)
        self.root.columnconfigure(2, minsize=150)

        # Prepare tk variables with default values
        self.step_var = tk.StringVar(master=self.root, value="Step: 0/0")
        self.epoch_var = tk.StringVar(master=self.root, value="Epoch: 0/{}"
                                      .format(total_epoch))
        self.rate_var = tk.StringVar(master=self.root, value="Rate: 0 steps/s")

        self.accuracy_var = tk.StringVar(master=self.root, value="0.00%")
        self.loss_var = tk.StringVar(master=self.root, value="Loss: 0.00")
        self.time_var = tk.StringVar(master=self.root, value="Time left: 0:00")
        self.status = tk.StringVar(master=self.root, value="Preparing dataset")

        # Prepare image canvases
        black_array = np.zeros((306, 408), float)
        black_image = ImageTk.PhotoImage(image=Image.fromarray(black_array))

        self.target_canvas = tk.Canvas(self.root, width=408, height=306)
        target_canvas_img = self.target_canvas.create_image(0, 0, anchor="nw",
                                                            image=black_image)
        self.target_canvas.grid(row=0, column=0, rowspan=4)

        self.seg_canvas = tk.Canvas(self.root, width=408, height=306)
        seg_canvas_img = self.seg_canvas.create_image(0, 0, anchor="nw",
                                                      image=black_image)
        self.seg_canvas.grid(row=4, column=0)

        # Prepare tk labels to be put on the grid
        # Row 0 labels
        tk.Label(self.root, textvariable=self.step_var).grid(row=0, column=1,
                                                             sticky="W",
                                                             padx=5, pady=5)

        tk.Label(self.root, textvariable=self.epoch_var).grid(row=0, column=2,
                                                              sticky="W",
                                                              padx=5, pady=5)

        # Row 1 labels
        tk.Label(self.root, textvariable=self.accuracy_var).grid(row=1,
                                                                 column=1,
                                                                 sticky="W",
                                                                 padx=5,
                                                                 pady=5)
        tk.Label(self.root, textvariable=self.loss_var).grid(row=1, column=2,
                                                             sticky="W", padx=5,
                                                             pady=5)

        # Row 2 labels
        tk.Label(self.root, textvariable=self.rate_var).grid(row=2, column=1,
                                                             sticky="W", padx=5,
                                                             pady=5)
        tk.Label(self.root, textvariable=self.time_var).grid(row=2, column=2,
                                                             sticky="W", padx=5,
                                                             pady=5)
        # Row 3 labels
        tk.Label(self.root, textvariable=self.status).grid(row=3, column=1,
                                                           columnspan=3,
                                                           sticky="SW", padx=5,
                                                           pady=5)

        # Update root so it actually shows something
        self._update()

        # Tell the user the window is loaded
        self.status.set("Window Loaded.")

        # Bring the window to the front
        self._lift()

        self.root.mainloop()  # FIXME Remove this later

    def update_data(self, target, generated, step, epoch, accuracy, loss, rate):
        """Updates the information within the GUI.

        The information displayed by the GUI should be updated after every step
        done by the trainer.

        Args:
            target (array-like): An array-like object that represents the target
                                 segmentation.
            generated (array-like): An array-like object that represents the
                                    segmentation generated by the network.
            step (int): The current step of the training process.
            epoch (int): The current epoch of the training process.
            accuracy (float): The accuracy of the network at the current step.
            loss (float): The loss of the network at the current step.
            rate (float): The rate the network is running at in steps per
                          second.
        """

        self._update()

    def update_status(self, message):
        """Updates the status message within the GUI

        Args:
            message (str): The new message that should displayed.
        """
        self.status.set(message)
        self._update()

    def _update(self):
        """Updates the root."""
        self.root.update()
        self.root.update_idletasks()

    def _lift(self):
        """Brings the tkinter window to the front.

        Note:
            This is required, and not simply root.lift, on macOS.
        """
        self.root.call('wm', 'attributes', '.', '-topmost', '1')
        self.root.call('wm', 'attributes', '.', '-topmost', '0')

    def set_max_step(self, total_steps):
        """Sets the number of steps that the training session will have.

        Args:
            total_steps (int): The total number of steps.
        """
        self.total_steps = total_steps
