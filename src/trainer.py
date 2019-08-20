# -*- coding: UTF-8 -*-
"""Trainer.

This class implements the necessary functions to train the network.

Also creates a current status file. This file is to be able to check the status
of the program from the command line without needing any sort of communication
protocol. It works by just simply being a file that has the current state
written to every 10 steps. It's located as a file that always has the same name
within the plotting path.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
# Torch imports
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss

# Python built in imports
import time
from time import strftime, localtime
from os import path
import datetime
import warnings
import numpy as np

# Custom classes imports
from ui.training_gui import TrainingGUI
from ui.training_cmd import TrainingCmd
from utils.mapillarydataset import MapillaryDataset
from utils.plotcsv import PlotCSV
from network.mce_loss import MCELoss
from constants import VALIDATION_STEPS
from utils import calculate_accuracy

# network imports
from network.parallelizer import Parallelizer as Network
from network.curbnet_d import CurbNetD
from network.curbnet_e import CurbNetE
from network.curbnet_f import CurbNetF
from network.curbnet_g import CurbNetG


class Trainer:
    def __init__(self, iaa, lr: float = 0.01, optimizer: str = "sgd",
                 loss_weights: list = None, cmd_line=None,
                 validation: bool = False,
                 loss_criterion: str = 'mce') -> None:
        """Training class used to train the CurbNetG network.

        Args:
            lr: Learning rate of the network
            optimizer: The optimizer to be used. Defaults to sgd.
            Possible optimizers are:
                - sgd: Stochastic Gradient Descent.
                - Adam: Adaptive moment estimation.
            loss_weights: A list of floats with length 3 that are the
                weight values for each class used by the loss function.
            cmd_line: Whether or not to use the command line
                interface. Defaults to None.
            silent: Whether or not to immediately quit upon completion. Defaults
                to false.
            validation: Whether or not to learn during this session
            loss_criterion: Which loss criterion to use. Options are 'mce' or
                'ce'. Defaults to 'mce'.
        """
        self.validation = validation
        self.network = None
        self.iaa = iaa

        if loss_weights is None:
            # To avoid mutable default values
            loss_weights = [0.005, 0.3000, 0.695]
            # Now normalize loss weights to account for 3x multiplication of
            # penalized weights
            # This was removed because I think it may have caused nan errors
            # loss_weights = [weight / 4 for weight in loss_weights]

        if torch.cuda.is_available():
            # Check if cuda is available and use it if it is
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            warnings.warn("CUDA compatible GPU not found. Running on CPU",
                          ResourceWarning)

        # Parameters
        self.lr = lr
        self.optimizer = optimizer

        # Set the loss criterion according to the recommended for pixel-wise
        # classification. We use weights so that missing curbs
        # will be more heavily penalized
        loss_weights = torch.tensor(loss_weights,
                                    dtype=torch.float).to(device=self.device)

        if loss_criterion == 'mce':
            self.criterion = MCELoss(weight_normal=loss_weights,
                                     weight_penalized=3 * loss_weights)
        if loss_criterion == 'ce':
            self.criterion = CrossEntropyLoss(weight=loss_weights)

        # Create UI
        if cmd_line:
            self.ui = TrainingCmd(cmd_line)
        else:
            self.ui = TrainingGUI()

        self.cmd_line = cmd_line

        # Set the status file path variable
        self.status_file_path = None

        # Creates logging tracker
        self.tracker = PlotCSV()

    def set_network(self, network: str = "d", pretrained=False,
                    px_coordinates=True):
        """Sets the network to be trained.

        Args:
            network: The network to be used. Defaults to DeepLab v3+
                Possible options are:
                - d: DeepLab v3+ based network.
                - e: ENET based network.
                - f: FCN based network.
                - g: GoogLeNet based encoder network.
            pretrained: Whether the network should use pretrained weights, if
                available.
            px_coordinates: Whether or not to include the pixel coordinates in
                the input for the network.
        """
        # Set up the different networks
        if network == "d":
            network = CurbNetD(pretrained=pretrained,
                               px_coordinates=px_coordinates)
        elif network == "e":
            network = CurbNetE()
        elif network == "f":
            network = CurbNetF()
        elif network == "g":
            network = CurbNetG()

        # Initialize the network as a parallelized network
        self.network = Network(network)

        self.network = self.network.to(device=self.device)

        # Set the network to train or to validation
        self.network.train(not self.validation)

        if not self.validation:
            # Set the optimizer according to the arguments if not validating
            if self.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.network.parameters(),
                                                  lr=self.lr, eps=0.1)
            elif self.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.network.parameters(),
                                                 lr=self.lr)
            else:
                raise ValueError("Illegal optimizer value: only SGD and Adam "
                                 "optimizers are currently supported.")

    def train(self, data_path, batch_size, num_epochs, plot_path, weights_path,
              augmentation = True, silent = False):
        """Start training the network.

        Args:
            data_path (str): Path to the dataset directory. This should be, for
                example, the mapillary directory, not the training or validation
                directory.
            batch_size (int): Number of images to run per batch.
            num_epochs (int): Number of epochs to run the trainer for.
            plot_path (str): Path to save the loss plot. This should be a
                directory.
            weights_path (str): The path to the weights.
            augmentation (bool): Whether or not to use augmentation. Defaults to
                True.
            silent (bool): Whether or not to immediately quit after finishing.
                Immediately quits if True. Defaults to False.

        Returns:
            None
        """
        if self.network is None:
            raise RuntimeError("No network has been set.")
        # Stat variables
        counter = 0

        self.tracker.configure({
            "plot path": plot_path,
            "weights path": weights_path,
            "training data path": data_path,
            "optimizer": self.optimizer,
            "batch size": batch_size,
            "epochs": num_epochs,
            "augmentation": augmentation,
            "network": self.network
        })

        # Set up the status file
        self.status_file_path = path.join(plot_path, "status.txt")

        # Load the dataset
        training_loader = self._load_dataset(path.join(data_path, "training"),
                                             augmentation,
                                             batch_size)
        validation_loader = self._load_dataset(path.join(data_path,
                                                         "validation"),
                                               augmentation,
                                               batch_size)

        # Load the state dictionary
        start_time = time.time()
        if path.isfile(weights_path):
            checkpoint = torch.load(weights_path)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._update_status("Loaded model and optimizer weights. ({} ms)"
                                .format(int((time.time() - start_time) * 1000)))
            del checkpoint  # Free up memory
        else:
            self._update_status("Warning: Weights do not exist yet.")

        # Start training
        start_time = time.time()
        absolute_start_time = time.time()
        self._update_status("Starting training on {} GPU(s)."
                            .format(torch.cuda.device_count()))
        for epoch in range(num_epochs):
            # Figure out number of max steps for info displays
            max_steps = len(training_loader)
            self.ui.set_max_values(max_steps, num_epochs)

            csv_data = []  # Reset the csv data list at every epoch start

            # Step iterations for training
            for data in enumerate(training_loader):
                # Grab the raw and target images
                raw_image = data[1]["raw"]
                target_image = data[1]["segmented"]

                # Run the network, but make sure the tensor is in the right
                # format
                out = self.network(raw_image.to(self.device, non_blocking=True))

                # Make sure it's been run properly
                if out is None:
                    raise ValueError("forward() has not been run properly.")

                # Since out is used multiple times, we detach it once
                detached_out = out.cpu().detach()

                # Start post processing
                # Create an argmax version here to avoid making it in the GUI
                # class and since we need it for multiple things
                out_argmax = torch.argmax(detached_out, dim=1)

                target_image[target_image == 3] = 0

                # Calculate loss, converting the tensor if necessary
                loss = self.criterion(out, target_image.to(self.device,
                                                           dtype=torch.long,
                                                           non_blocking=True))

                # Things to do if we're training and not validating
                if not self.validation:
                    # Zero out the optimizer
                    self.optimizer.zero_grad()

                    # Backprop and perform optimization
                    loss.backward()
                    self.optimizer.step()

                counter += 1

                # Calculate accuracy
                accuracy = self._calculate_batch_accuracy(target_image,
                                                          out_argmax,
                                                          batch_size)

                rate = float(counter) / (time.time() - start_time)

                loss_value = loss.item()

                self.ui.update_data(step=data[0] + 1,
                                    epoch=epoch + 1,
                                    accuracy=accuracy,
                                    loss=loss_value,
                                    rate=rate,
                                    status_file_path=self.status_file_path)



                show_image = (raw_image[0]+0.5)*255.
                show_image = show_image.type(torch.LongTensor)
                self._update_gui_image(detached_out[0], out_argmax[0],
                                       target_image[0], show_image)


                # Write to the plot file every step
                csv_data.append({"loss": loss_value,
                                 "accuracy other": accuracy[0],
                                 "accuracy curb": accuracy[1],
                                 "accuracy curb cut": accuracy[2],
                                 "validation loss": ""})

                # Remove unnecessary tensors to save memory
                del out
                del detached_out
                del out_argmax
                del target_image
                del raw_image
                del data

            # Empty cache first
            torch.cuda.empty_cache()

            # Do validation
            self._update_status("Started validation for epoch {}"
                                .format(epoch + 1))
            # Set to evaluation mode
            self.network.eval()
            sum_validation_loss = 0
            num_validation_steps = 0

            # Actually do the validation
            for data in enumerate(validation_loader):
                out = self.network(data[1]["raw"].to(self.device,
                                                     non_blocking=True))

                if out is None:
                    raise ValueError("forward() has not been run properly.")

                target_image = data[1]["segmented"]
                target_image[target_image == 3] = 0  # Get rid of roads

                loss = self.criterion(out,
                                      target_image.to(self.device,
                                                      dtype=torch.long,
                                                      non_blocking=True))

                sum_validation_loss += loss.item()

                counter += 1
                num_validation_steps += 1

                accuracy = self._calculate_batch_accuracy(target_image,
                                                          out.cpu().detach(),
                                                          batch_size)

                rate = float(counter) / (time.time() - start_time)

                self.ui.update_data(step=data[0] + 1,
                                    epoch=epoch + 1,
                                    accuracy=accuracy,
                                    loss=loss,
                                    rate=rate,
                                    status_file_path=self.status_file_path,
                                    validation=True)

                # self._update_gui_image(detached_out[0],
                #                        torch.argmax(detached_out[0], dim=1),
                #                        target_image[0],
                #                        data[1]["raw"][0])

                if data[0] + 1 == VALIDATION_STEPS:
                    # Only do 10 steps for validation
                    # This is at the end to prevent any performance loss from
                    # loading unnecessary data
                    del out
                    del target_image
                    del data
                    del loss
                    break

                # Delete stuff to save memory
                del out
                del target_image
                del data
                del loss
                torch.cuda.empty_cache()

            # Write validation loss
            validation_loss = sum_validation_loss / float(num_validation_steps)
            csv_data[-1]["validation loss"] = validation_loss

            self._update_status("Finished validation with {:.3f} "
                                "loss.".format(validation_loss))

            # Put network back to training mode
            self.network.train(not self.validation)
            # Write csv data
            self.tracker.write_data(csv_data)

            # Save the weights every epoch
            torch.save({
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()},
                weights_path)

        self._update_status(
            "Finished training in {}.".format(datetime.timedelta(
                seconds=int(time.time() - absolute_start_time))))

        # Now save the loss and accuracy file
        self.tracker.close()

        if not silent:
            if self.cmd_line:
                # Keep command line window open
                key = self.cmd_line.getch()
                if key == ord('q'):
                    self.cmd_line.clear()
                    return

            else:
                # Keep TK window open
                self.ui.mainloop()

    def _calculate_batch_accuracy(self, ground_truth, predicted, batch_size):
        """Calculates accuracy of the batch using intersection over union.

        We use a separate function to make it easier to follow and for the
        actual training code easier to maintain.

        Args:
            ground_truth (torch.*Tensor): The batch of the ground truth data.
            predicted (torch.*Tensor): The batch of the predicted segmentation
                                       generated by the network.
            batch_size (int): The batch size used.

        returns:
            float: Average accuracy of the batch.
        """
        accuracy = np.array([0., 0., 0.])
        for idx, item in enumerate(ground_truth):
            accuracy += calculate_accuracy(item, predicted[idx])

        return accuracy / batch_size

    def _update_status(self, message):
        """Updates the status of the program in the GUI, console, and log.

        Args:
            message (str): The message to be written.
            ui (ui.TrainingUI): The UI object used.
            plot_csv (utils.plotcsv.PlotCSV): The PlotCSV object used
        """
        message = "[{}] {}".format(strftime("%H:%M:%S", localtime()), message)
        self.tracker.write_log(message)
        self.ui.update_status(message)

    @staticmethod
    def _target_to_one_hot(target_tensor):
        """Converts target to a one_hot tensor, also removing road class.

        Args:
            target_tensor (torch.Tensor*): The tensor to be changed into one-hot
                encoding. If taken from a batch, this should be a single image
                and not the whole batch.
        """
        target_tensor = target_tensor.unsqueeze(2)
        one_hot = torch.zeros(target_tensor.shape[0], target_tensor.shape[1], 3)
        one_hot = one_hot.scatter_(2, target_tensor.to(dtype=torch.long), 1)
        return one_hot

    def _load_dataset(self, data_path, augmentation, batch_size):
        """Loads a dataset and creates the data loader.

        Args:
            data_path (str): The path of the mapillary folder.
            augmentation (bool): Augment the images or not.
            batch_size (int): Size of each batch

        Returns:
            torch.utils.data.DataLoader: The data loader for the dataset
        """
        if path.split(data_path)[1] == "":
            # Deal with edge case where there's a "/" at the end of the path.
            data_path = path.split(data_path)[0]

        if path.split(data_path)[1].endswith("training"):
            dataset_name = "training dataset"
        else:
            dataset_name = "validation dataset"

        start_time = time.time()
        self._update_status("Loading {}.".format(dataset_name))


        dataset = MapillaryDataset(data_path, augmentation, self.iaa)
        data_loader = DataLoader(dataset,
                                 batch_size,
                                 shuffle=True)

        self._update_status("{} loaded. ({} ms)".format(
            dataset_name.capitalize(),
            int((time.time() - start_time) * 1000)))

        return data_loader

    def _update_gui_image(self, predicted, predicted_argmax, target, raw):
        """Updates the GUI with the new set of images."""
        if not self.cmd_line:
            # Do processing to make the target_image into one-hot
            # encoding
            target = self._target_to_one_hot(target)
            predicted = self._process_out_for_gui(predicted, predicted_argmax)

            self.ui.update_image(
                target=target,
                generated=predicted,
                input_image=raw
            )

    @staticmethod
    def _process_out_for_gui(prediction, argmax):
        """Processes out for GUI by selecting only max values.

        Since the GUI should only show colors for max values, if we feed the
        entire processed image to the GUI, there will be overlapping red and
        green classifications. We therefore choose to only show the max prob for
        each pixel. For example, if there is a pixel with classification:
        [0.2, 0.5, 0.1], then we turn that into [0., 0.5, 0.]

        Args:
             prediction (torch.Tensor*): The output tensor from the network
                in the form [C x H x W].
             argmax (torch.Tensor*): The indices of the maximum values.

        Returns:
            torch.Tensor*: The tensor, processed as explained above.
        """
        out = torch.zeros_like(prediction)
        # Create a curbs mask
        mask = argmax == 1
        # Fill seg 1 (curbs) with those in the mask
        out[1] = torch.where(mask, prediction[1], out[2])

        # Create a curb cuts mask
        mask = argmax == 2
        # Fill seg 2 (curb cuts) with those in the mask
        out[2] = torch.where(mask, prediction[2], out[2])

        # Memory saving
        del mask

        return out
