"""Main.

The main script that is called to run everything else.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import argparse
from trainer import Trainer
import sys

def parse_arguments():
    """Parses arguments.

    Positional arguments:
        W               path to the file that contains the weights. If the
                        file doesn't exist, one will be created.

    Optional arguments:
        -h, --help          show this help message and exit
        -t, --train         sets to training mode and gives the path to the data
                            directory
        -v, --validate      sets to validation mode and gives the path to the
                            data directory for validation
        -r --learning-rate  sets the learning rate for the optimizer
        -o, --optimizer     sets the optimizer. Currently supported optimizers
                            are "adam" and "sgd"
        -b, --batch-size    sets the batch size for the session
        -e, --epochs        sets the number of epochs for the session
        -a, --augment       activates image augmentation for the session
        -p, --plot          sets the path for the loss and accuracy csv file
        -i, --infer         runs the network for inference
        --profile           profiles the network when used in conjunction with
                            either training, validation or inference mode
    """
    description = "Trains or validates CurbNet on a dataset, " \
                  "or uses it for inference."
    parser = argparse.ArgumentParser(description=description)

    # Start adding arguments
    parser.add_argument('weights', metavar='W', type=str, nargs=1,
                        help="path to the file that contains the weights. "
                             "If the file doesn't exist, one will be created.")
    parser.add_argument('-t', '--train', type=str, nargs='?',
                        help="sets to training mode and gives the path to the "
                             "data directory")
    parser.add_argument('-v', '--validate', type=str, nargs='?',
                        help="sets to validation mode and gives the path to "
                             "the data directory for validation")
    parser.add_argument('-r', '--learning-rate', type=float, nargs='?',
                        default=0.002,
                        help="sets the learning rate for the optimizer")
    parser.add_argument('-o', '--optimizer', type=str, nargs='?',
                        default="adam",
                        help="sets the optimizer. Currently supported"
                             "optimizers are \"adam\" and \"sgd\"")
    parser.add_argument('-b', '--batch-size', type=int, nargs='?', default=5,
                        help="sets the batch size for the session")
    parser.add_argument('-e', '--epochs', type=int, nargs='?', default=1,
                        help="sets the number of epochs for the session")
    parser.add_argument('-a', '--augment', action='store_true',
                        help="activates image augmentation for the session")
    parser.add_argument('-p', '--plot', type=str, nargs='?',
                        help="sets the path for the loss and accuracy csv file")
    parser.add_argument('-i', '--infer', action='store_true',
                        help="runs the network for inference")
    parser.add_argument('--profile', action='store_true',
                        help="profiles the network when used in conjunction "
                             "with either training, validation or "
                             "inference mode")

    arguments = parser.parse_args()

    if arguments.infer and (arguments.plot or arguments.augment
                            or arguments.epochs or arguments.batch_size
                            or arguments.optimizer or arguments.learning_rate
                            or arguments.validate or arguments.train):
        parser.error("Inference only requires weights to be given")

    if arguments.profile and not (arguments.infer or arguments.train
                                  or arguments.validate):
        parser.error("Must profile a specific mode")

    return arguments

def main(arguments):
    """Main function that runs everything.

    Args:
        arguments (argparse.Namespace): The arguments given by the user.
    """
    if arguments.train:
        # Run in training mode
        trainer = Trainer(arguments.learning_rate, arguments.optimizer)
        trainer.train(arguments.train, arguments.batch_size, arguments.epochs,
                      arguments.plot, arguments.W, arguments.augment)

        # Clean exit on completion
        sys.exit()


if __name__ == "__main__":
    try:
        main(parse_arguments())
    except KeyboardInterrupt:
        print("User exited program. Killing process.")

    # t = Trainer()
    # t.train("", 5, 5, "", "", True)
