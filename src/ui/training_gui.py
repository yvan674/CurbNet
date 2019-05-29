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
import datetime
from torch import argmax

# matplotlib imports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

from ui.training_ui import TrainingUI


class Status(tk.Frame):
    def __init__(self, master=None):
        """Creates a frame that contains all the string-based statuses."""
        super().__init__(master=master, bg="#282c34", width=400, height=300)
        self.columnconfigure(0, minsize=150)
        self.columnconfigure(1, minsize=150)

        # Store max values for epoch and step
        self.max_step = 0
        self.max_epoch = 0

        # Prepare tk variables with default values
        self.step_var = tk.StringVar(master, value="Step: 0/0")
        self.epoch_var = tk.StringVar(master, value="Epoch: 0/0")

        self.rate_var = tk.StringVar(master, value="Rate: 0 steps/s")
        self.time_var = tk.StringVar(master, value="Time left: 0 seconds")

        self.loss_var = tk.StringVar(master, value="Loss: 0.000")
        self.accuracy_var = tk.StringVar(master, value="Accuracy: 0.000%")

        self.status = tk.StringVar(master, value="")

        self.labels = [
            # Row 0 Labels
            tk.Label(self, textvariable=self.step_var),
            tk.Label(self, textvariable=self.epoch_var),
            # Row 1 Labels
            tk.Label(self, textvariable=self.loss_var),
            tk.Label(self, textvariable=self.accuracy_var),
            # Row 2 Labels
            tk.Label(self, textvariable=self.rate_var),
            tk.Label(self, textvariable=self.time_var),
            # Row 3 Labels
            tk.Label(self, textvariable=self.status)
        ]

        # Configure all the labels and put them on the grid
        counter = 0
        for label in self.labels:
            label["bg"] = "#282c34"
            label["fg"] = "#a8afb8"
            if counter > 7:
                label.grid(row=int(counter / 2), column=counter % 2, sticky="W",
                           padx=5, pady=5)
            else:
                label.grid(row=int(counter / 2), column=counter % 2, sticky="W",
                           columnspan=2, padx=5, pady=5)
            counter += 1

    def update_data(self, step, epoch, accuracy, loss, rate, status_file_path):
        """Updates the string-based information within the GUI.

        The information displayed by the GUI should be updated after every step
        done by the trainer.

        Args:
            step (int): The current step of the training process.
            epoch (int): The current epoch of the training process.
            accuracy (float): The accuracy of the network at the current step.
            loss (float): The loss of the network at the current step.
            rate (float): The rate the network is running at in steps per
                          second.
            status_file_path (str): The path to save the status file to.
        """
        # Row 0 labels
        self.step_var.set("Step: {}/{}".format(step, self.max_step))
        self.epoch_var.set("Epoch: {}/{}".format(epoch, self.max_epoch))

        # Row 1 labels
        self.loss_var.set("Loss: {:.3f}".format(loss))
        self.accuracy_var.set("Accuracy: {:.3f}%".format(accuracy * 100))

        # Row 2 labels
        self.rate_var.set("Rate: {:.3f} steps/sec".format(rate))

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

        self.time_var.set("Time left: {}".format(time_left))

        # Now write the status file
        if step % 10 == 0:
            with open(status_file_path, 'w') as status_file:
                lines = ["Step: {}\n".format(step),
                         "Epoch: {}\n".format(epoch),
                         "Accuracy: {}\n".format(accuracy),
                         "Loss: {:.3f}\n".format(loss),
                         "Rate: {:.3f} steps/s\n".format(rate),
                         "Time left: {}\n".format(time_left)]

                if step == self.max_step:
                    lines[5] = "Time left: -\n"
                    lines.append("Finished training.\n")

                status_file.writelines(lines)

    def update_status(self, message):
        """Updates the status message within the GUI.

        Args:
            message (str): The new message that should be displayed.
        """
        self.status.set(message)
        print(message)

    def set_max(self, max_step, max_epoch):
        """Sets the maximum values for step and epoch.

        Args:
            max_step (int): The maximum number of steps.
            max_epoch (int): The maximum number of epochs.
        """
        self.max_step = max_step
        self.max_epoch = max_epoch

        self.step_var.set("Step: 0/{}".format(max_step))
        self.epoch_var.set("Epoch: 0/{}".format(max_epoch))


class Plots(tk.Frame):
    def __init__(self, master=None):
        """Creates a plotting frame that can be used as a tk module."""
        super().__init__(master)
        f = Figure(figsize=(4, 3), dpi=100)
        f.set_facecolor("#282c34")
        self.loss_values = []
        self.accuracy_values = []

        # Axes
        ax1 = f.add_subplot(111)
        ax2 = ax1.twinx()
        self.axes = [ax1, ax2]

        # Axis label and color
        self.axes[0].set_ylabel("Loss")
        self.axes[0].yaxis.label.set_color("#a8afb8")
        self.axes[0].tick_params(axis='y', colors="tab:red")
        self.axes[1].set_ylabel("Accuracy")
        self.axes[1].tick_params(axis='y', colors="tab:blue")
        self.axes[1].yaxis.label.set_color("#a8afb8")

        # Colors for the axes
        for axis in self.axes:
            axis.set_facecolor("#282c34")
            axis.spines["bottom"].set_color("#a8afb8")
            axis.spines["top"].set_color("#a8afb8")
            axis.spines["left"].set_color("#a8afb8")
            axis.spines["right"].set_color("#a8afb8")
            axis.tick_params(axis='x', colors='#a8afb8')
            axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        f.tight_layout()

        self.canvas = FigureCanvasTkAgg(f, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH,
                                         expand=True)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_data(self, loss, accuracy):
        """Updates the data in the graph

        Args:
            loss (float): New loss value to be appended to the plot.
            accuracy (float): New accuracy value to be appended to the plot
        """
        self.loss_values.append(loss)
        self.accuracy_values.append(accuracy)

        # Clear in preparation of new updates
        for axis in self.axes:
            axis.clear()
            axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Plot the new values
        self.axes[0].plot(self.loss_values, color='tab:red')
        self.axes[1].plot(self.accuracy_values, color='tab:blue')

        # Axis label and color
        self.axes[0].set_ylabel("Loss")
        self.axes[0].yaxis.label.set_color("#a8afb8")
        self.axes[0].tick_params(axis='y', colors="tab:red")
        self.axes[1].set_ylabel("Accuracy")
        self.axes[1].tick_params(axis='y', colors="tab:blue")
        self.axes[1].yaxis.label.set_color("#a8afb8")

        # Redraw canvas
        self.canvas.draw()


class ImageFrame(tk.Frame):
    def __init__(self, master=None):
        """Super class for frames that can contain images."""
        super().__init__(master=master, bg="#282c34",
                         width=400, height=300,
                         borderwidth=0)
        super().configure(background="#282c34")

        # Create a black image to initialize the canvas with
        black_image = np.zeros((300, 400))
        black_image = ImageTk.PhotoImage(image=Image.fromarray(black_image))

        # Set up the canvas
        self.canvas = tk.Canvas(self, bg="#282c34",
                                width=400, height=300)
        self.img = black_image
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw",
                                                   image=self.img)
        self.canvas.pack()

    def update_image(self, segmentation, input_image):
        """Updates the image that is to be displayed."""
        img_array = self._class_to_image_array(segmentation)
        img_array = self._overlay_image(input_image, img_array)

        img_array = Image.fromarray(img_array)
        img_array = img_array.resize((400, 300), Image.NEAREST)
        self.img = ImageTk.PhotoImage(image=img_array)
        self.canvas.itemconfig(self.canvas_img, image=self.img)

    @staticmethod
    def _class_to_image_array(image):
        """Converts the ground truth segmentation to an image array.

        Args:
            image (torch.Tensor): tensor array of classes in each pixel in the
                                  shape [H, W]

        Returns:
            (np.array) in the form [H, W, Color]
        """
        image = image.numpy()
        out = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Color code the output
        out[image == 1] = np.array([255, 0, 0])
        out[image == 2] = np.array([0, 255, 0])

        return out

    @staticmethod
    def _overlay_image(input_image, segmentation):
        """Overlays the segmentation on top of the input image.

        Args:
            input_image (torch.Tensor): The input image as a tensor.
            segmentation (numpy.array): Segmentation as a numpy array.

        Returns:
            numpy.array: The overlaid image as a numpy array.
        """
        input_image = input_image.numpy()
        output = np.transpose(input_image, (1, 2, 0))
        output = np.maximum(output, segmentation)
        return output.astype('uint8')


class TrainingGUI(TrainingUI):
    def __init__(self):
        """Creates a GUI to show training status using tkinter."""
        # Configure root
        self.root = tk.Tk()
        self.root.title("CurbNet Training")
        self.root.configure(background="#282c34")
        self.root.resizable(False, False)

        # Configure the grid and geometry
        # ----------------------------------------------------------
        # |                            |                           |
        # |     Target segmentation    |  loss and accuracy plot   |
        # |                            |                           |
        # |                            |                           |
        # |--------------------------------------------------------|
        # |                            | step        | epoch       |
        # |     Output segmentation    | loss        | accuracy    |
        # |                            | rate        | time left   |
        # |                            | status message            |
        # ----------------------------------------------------------
        self.root.geometry("800x600")
        self.root.columnconfigure(0, minsize=400)
        self.root.columnconfigure(1, minsize=400)
        self.root.rowconfigure(0, minsize=300)
        self.root.rowconfigure(1, minsize=300)

        # Setup the widgets
        self.widgets = [
            ImageFrame(self.root),
            Plots(self.root),
            ImageFrame(self.root),
            Status(self.root)
        ]

        # Place the widgets in the grid
        for i in range(4):
            self.widgets[i].grid(row=int(i / 2), column=(i % 2))

        # Finally, lift the window to the top
        self._lift()

    def update_data(self, step, epoch, accuracy, loss, rate, status_file_path):
        """Updates the string-based data in the GUI."""
        self.widgets[3].update_data(step, epoch, accuracy, loss, rate,
                                    status_file_path)
        self.widgets[1].update_data(loss, accuracy)
        self._update()

    def update_status(self, message):
        """Updates the status message in the GUI."""
        self.widgets[3].update_status(message)
        self._update()

    def set_max_values(self, total_steps, total_epochs):
        """Sets the max value for steps and epochs."""
        self.widgets[3].set_max(total_steps, total_epochs)

    def update_image(self, target, generated, input_image):
        """Updates the image in the GUI."""
        self.widgets[0].update_image(target, input_image)
        self.widgets[2].update_image(generated, input_image)

    def _update(self):
        """Internal update call that shortens 2 lines to one."""
        self.root.update()
        self.root.update_idletasks()

    def _lift(self):
        """Brings the tkinter window to the front.

        Note:
            This is required, and not simply root.lift, on macOS.
        """
        self.root.call('wm', 'attributes', '.', '-topmost', '1')
        self.root.call('wm', 'attributes', '.', '-topmost', '0')

    def mainloop(self):
        """Called at the end of training to keep the window active.
        """
        self.root.mainloop()
