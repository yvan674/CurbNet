#!/usr/bin/env python
"""Main.

The main script that is called to run everything else.


positional arguments:
    W                   path to the file that contains the weights. If the
                        file doesn't exist, one will be created.

optional arguments:
    -h, --help          show this help message and exit
    -c, --cmd-line      runs the program in command line mode using curses. Used
                        for remote training.

mode:
    mode to run the network in

    -t TRAIN, --train TRAIN
                        sets to training mode and gives the path to the
                        training data directory
    -v VALIDATE, --validate VALIDATE
                        sets to validation mode and gives the path to the
                        validation data directory
    -i, --infer         runs the network for inference

training arguments:
    -r [LEARNING_RATE], --learning-rate [LEARNING_RATE]
                        sets the learning rate for the optimizer
    -o [OPTIMIZER], --optimizer [OPTIMIZER]
                        sets the optimizer. Currently supported optimizers are
                        "adam" and "sgd"
    -l LOSS_WEIGHTS [LOSS_WEIGHTS ...], --loss-weights LOSS_WEIGHTS
                        custom per class loss weights as a set of 3 floats
    --pretrained        uses a pretrained decoder network, if available
    -x, --px-coordinates
                        adds pixel coordinates to the network input
    -p [PLOT], --plot [PLOT]
                        sets the path for the loss and accuracy csv file. If
                        none is given, set to the current working directory

training and validation arguments:
    -b [BATCH_SIZE], --batch-size [BATCH_SIZE]
                        sets the batch size for the session
    -e [EPOCHS], --epochs [EPOCHS]
                        sets the number of epochs for the session
    -a, --augment       activates image augmentation for the session
    -l LOSS_WEIGHTS [LOSS_WEIGHTS ...], --loss-weights LOSS_WEIGHTS
                        custom per class loss weights as a set of 3 floats


network arguments:
    -n, --network       choose the network. Options are "d" for DeepLab,"e" for
                        ENet, "f" for FCN, and "g" for GoogLeNet. Defaults to "d"
    --pretrained        uses a pretrained decoder network, if available
    -x, --px-coordinates
                        adds pixel coordinates to the network input
    -p [PLOT], --plot [PLOT]
                        sets the path for the loss and accuracy csv file. If
                        none is given, set to the current working directory

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import argparse
from trainer import Trainer
from os import getcwd
from os.path import join
import sys
from platform import system
import warnings
import atexit
try:
    import curses
except ImportError:
    if system() == "Windows":
        pass
    else:
        warnings.warn("Running on a non-Windows OS without curses. Command line"
                      " usage will not be possible.",
                      ImportWarning)


def parse_arguments():
    """Parses arguments."""
    description = "trains or validates CurbNet on a dataset, " \
                  "or uses it for inference."
    parser = argparse.ArgumentParser(description=description)

    # Start adding arguments
    parser.add_argument('weights', metavar='W', type=str, nargs=1,
                        help="path to the file that contains the weights. "
                             "If the file doesn't exist, one will be created.")

    # Mutex args
    mode = parser.add_argument_group("mode", "mode to run the network in")
    mutex = mode.add_mutually_exclusive_group(required=True)
    mutex.add_argument('-t', '--train', type=str, nargs=1,
                       help="sets to training mode and gives the path to "
                            "the training data directory")
    mutex.add_argument('-v', '--validate', type=str, nargs=1,
                       help="sets to validation mode and gives the path to "
                            "the validation data directory")

    mutex.add_argument('-i', '--infer', action='store_true',
                       help="runs the network for inference")

    # Training arguments
    training = parser.add_argument_group("training arguments")
    training.add_argument('-r', '--learning-rate', type=float, nargs='?',
                          default=0.002,
                          help="sets the learning rate for the optimizer")
    training.add_argument('-o', '--optimizer', type=str, nargs='?',
                          default="adam",
                          help="sets the optimizer. Currently supported"
                               "optimizers are \"adam\" and \"sgd\"")

    # Validation and training arguments
    vt = parser.add_argument_group("validation and training arguments")
    vt.add_argument('-b', '--batch-size', type=int, nargs='?', default=5,
                          help="sets the batch size for the session")
    vt.add_argument('-e', '--epochs', type=int, nargs='?', default=1,
                          help="sets the number of epochs for the session")
    vt.add_argument('-a', '--augment', action='store_true',
                          help="activates image augmentation for the session")
    vt.add_argument('-l', '--loss-weights', type=float, nargs='+',
                    help='custom per class loss weights as a set of 3 floats')

    # Network arguments
    network = parser.add_argument_group("network arguments")
    network.add_argument('-n', '--network', type=str, nargs='?', default="d",
                         help="sets the network to be used. Supported options "
                              "are \"e\", \"f\", and \"g\". Defaults to \"d\"")
    network.add_argument('--pretrained', action='store_true',
                         help="uses a pretrained decoder network, if "
                              "available")
    network.add_argument('-x', '--px-coordinates', action='store_true',
                         help="adds pixel coordinates to the network input")
    network.add_argument('-p', '--plot', type=str, nargs='?',
                         help="sets the path for the loss and accuracy csv "
                              "file. If none is given, set to the current "
                              "working directory")

    # General arguments
    parser.add_argument('-c', '--cmd-line', action='store_true',
                        help="runs the program in command line mode using "
                             "curses. Used for remote training.")

    arguments = parser.parse_args()

    if arguments.infer and (arguments.plot or arguments.augment
                            or arguments.epochs or arguments.batch_size
                            or arguments.optimizer or arguments.learning_rate
                            or arguments.validate or arguments.train):
        warnings.warn("Inference only requires weights to be given. All "
                      "other arguments will be ignored.", UserWarning)

    if (arguments.train or arguments.validate) and not arguments.plot:
        warnings.warn("Plot save path not given. Setting path to the current "
                      "working directory.", UserWarning)
        arguments.plot = join(getcwd(), "plot")

    return arguments


def main(arguments):
    """Main function that runs everything.

    Args:
        arguments (argparse.Namespace): The arguments given by the user.
    """
    # Get the trainer object ready
    if arguments.train:
        # Run in training mode
        trainer = Trainer(arguments.learning_rate, arguments.optimizer,
                          arguments.loss_weights, arguments.cmd_line)

        trainer.set_network(arguments.network, arguments.pretrained,
                            arguments.px_coordinates)
        data = arguments.train[0]

    elif arguments.validate:
        # Run for validation
        trainer = Trainer(cmd_line=arguments.cmd_line,
                          validation=arguments.validate)
        trainer.set_network(arguments.network, arguments.pretrained,
                            arguments.px_coordinates)
        data = arguments.validate[0]

    # Run training or validation
    if arguments.train or arguments.validate:
        if arguments.cmd_line:
            curses.wrapper(trainer.train(data,
                                         arguments.batch_size, arguments.epochs,
                                         arguments.plot, arguments.weights[0],
                                         arguments.augment))

        else:
            trainer.train(data, arguments.batch_size,
                          arguments.epochs, arguments.plot,
                          arguments.weights[0], arguments.augment)

    elif arguments.infer:
        raise NotImplementedError("Inference is not yet implemented.")

    # Clean exit on completion
    sys.exit()


def closing_functions():
    try:
        curses.echo()
        curses.endwin()
    except:
        pass
    sys.exit()

atexit.register(closing_functions)

if __name__ == "__main__":
    arguments = parse_arguments()

    if arguments.cmd_line:
        main(arguments)

    else:
        try:
            main(arguments)
        except KeyboardInterrupt:
            print("User exited program. Killing process.")
