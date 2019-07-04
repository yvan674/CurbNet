"""Training GUI.

This module allows training to be displayed through a GUI. This makes it easier
to see what's going on and is just nicer overall compared to training a
segmentation network over the command line. The purpose of the module is to
show the ground truth and the generated segmentation, the status of the training
session, and a plot of the current loss vs accuracy of the network.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    Blending modes implementation:
        <https://github.com/flrs/blend_modes/>
"""
import tkinter as tk
import numpy as np
from PIL import ImageTk, Image
import datetime

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

    def update_data(self, step, epoch, accuracy, loss, rate, status_file_path,
                    validation=False):
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
            validation (bool): The state of the training, if it is in validation
                or in training where False means training. Defaults to False.
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
            # Add the validation steps
            steps_total += float(10 * self.max_epoch)

            steps_done_this_epoch = float(step + 1
                                          + (validation * self.max_step))

            steps_times_epochs_done = float(self.max_step * (epoch - 1))

            steps_left = (steps_total - steps_done_this_epoch
                          - steps_times_epochs_done)

            time_left = int(steps_left / rate)
            time_left = str(datetime.timedelta(seconds=time_left))

        self.time_var.set("Time left: {}".format(time_left))

        # Now write the status file
        if step % 10 == 0 or (step == self.max_step
                              and epoch == self.max_epoch):
            with open(status_file_path, 'w') as status_file:
                lines = ["Step: {}/{}\n".format(step, self.max_step),
                         "Epoch: {}/{}\n".format(epoch, self.max_epoch),
                         "Accuracy: {:.2f}%\n".format(accuracy * 100),
                         "Loss: {:.3f}\n".format(loss),
                         "Rate: {:.3f} steps/s\n".format(rate),
                         "Time left: {}\n".format(time_left)]

                if step == self.max_step and epoch == self.max_epoch:
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
        """Updates the image that is to be displayed.

        Args:
            segmentation (torch.Tensor): The segmentation in one-hot encoding.
            input_image (torch.Tensor): The original image as a torch tensor
                straight from the dataloader.
        """
        img_array = self._screen_image(input_image, segmentation.numpy())

        img_array = Image.fromarray(img_array)
        img_array = img_array.resize((400, 300), Image.NEAREST)
        self.img = ImageTk.PhotoImage(image=img_array)
        self.canvas.itemconfig(self.canvas_img, image=self.img)

    def _screen_image(self, input_image, segmentation):
        """Overlays the segmentation on top of the input image.

        This uses the screen blend mode
        <https://en.wikipedia.org/wiki/Blend_modes#Screen>. The python
        implementation is from <https://github.com/flrs/blend_modes/>

        Args:
            input_image (torch.Tensor): The input image as a tensor. This is
                expected to be in the range [0, 1].
            segmentation (numpy.array): Segmentation as a numpy array. This is
                expected to be in the range [0, 1].

        Returns:
            numpy.array: The overlaid image as a numpy array.
        """
        # First prepare base layer by turning it into numpy and rescaling it to
        # the range [0, 1].
        input_image = input_image.numpy().astype(float)

        # Then give it an alpha layer. Hacky but I'm too tired to think.
        base = np.full((4, input_image.shape[1], input_image.shape[2]), 1.)
        base[0:3] = input_image / 255

        # Finally tranpose it to [H x W x C]
        base = np.transpose(base, (1, 2, 0))

        # Rename the segmentation var and transpose if necessary to [C x H x W]
        segmentation = segmentation.astype(float)
        if segmentation.shape[0] != 3:
            segmentation = np.transpose(segmentation, (2, 0, 1))

        if segmentation.max() == 255.:
            segmentation = segmentation / 255.

        # Process segmentation
        # Make shape in the form [C x H x W] where C is RGBA and a filled array
        rgba_shape = (4, segmentation.shape[1], segmentation.shape[2])
        filled_array = np.full((rgba_shape[1], rgba_shape[2]), 1.)

        # First make the red layer, aka curbs
        red = np.zeros(rgba_shape)
        red[0] = filled_array
        red[3] = segmentation[1]
        # And transpose it
        red = np.transpose(red, (1, 2, 0))

        # Then make the green layer, aka curb cuts
        green = np.zeros(rgba_shape)
        green[1] = filled_array
        green[3] = segmentation[2]
        # And transpose it
        green = np.transpose(green, (1, 2, 0))

        # Now we have base, red, red alpha, green, green alpha. Time to blend.
        out = self._screen(base, red, 1.)
        out = self._screen(out, green, .75)  # 1. is too bright so we use .75
        return (out * 255).astype('uint8')

    @staticmethod
    def _screen(base, top, alpha):
        """Apply screen blending mode of a layer on an image.

        Args:
          base(3-dimensional numpy array of floats (r/g/b/a) in range
          0-255.0): Image to be blended upon
          top(3-dimensional numpy array of floats (r/g/b/a) in range
          0.0-255.0): Layer to be blended with image
          alpha(float): Desired opacity of layer for blending

        Returns:
          3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0:
          Blended image
        """
        def _compose_alpha(img_in, img_layer, opacity):
            """Calculate alpha composition ratio between two images."""
            comp_alpha = np.minimum(img_in[:, :, 3],
                                    img_layer[:, :, 3]) * opacity
            new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
            np.seterr(divide='ignore', invalid='ignore')
            alpha_ratio = comp_alpha / new_alpha
            alpha_ratio[alpha_ratio == np.NAN] = 0.0

            return alpha_ratio

        ratio = _compose_alpha(base, top, alpha)

        comp = 1.0 - (1.0 - base[:, :, :3]) * (1.0 - top[:, :, :3])

        ratio_rs = np.reshape(np.repeat(ratio, 3),
                              [comp.shape[0], comp.shape[1], comp.shape[2]])
        img_out = comp * ratio_rs + base[:, :, :3] * (1.0 - ratio_rs)
        img_out = np.nan_to_num(np.dstack(
            (img_out, base[:, :, 3])))  # add alpha channel and replace nans

        return img_out


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

    def update_data(self, step, epoch, accuracy, loss, rate, status_file_path,
                    validation=False):
        """Updates the string-based data in the GUI."""
        self.widgets[3].update_data(step, epoch, accuracy, loss, rate,
                                    status_file_path, validation)
        if not validation:
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
