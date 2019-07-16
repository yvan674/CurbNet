"""Main.

The main script that is called to run everything else.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from platform import system
from trainer import Trainer
from utils.slacker import Slacker

from os import getcwd
from os.path import join
import warnings
import traceback

try:
    import curses
except ImportError:
    if system() == "Windows":
        pass
    else:
        warnings.warn("Running on a non-Windows OS without curses. Command line"
                      " usage will not be possible.",
                      ImportWarning)


def run_training(arguments):
    """Main function that runs everything.

    Args:
        arguments (dict): The arguments given by the user.
    """
    # Get the curses window ready by setting it to None
    stdscr = None

    try:
        if arguments["cmd_line"]:
            stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            try:
                curses.curs_set(0)
            except:
                pass

        # Get the trainer object ready
        if arguments["train"]:
            # Run in training mode
            trainer = Trainer(arguments["learning_rate"],
                              arguments['optimizer'],
                              arguments['loss_weights'], stdscr)

            trainer.set_network(arguments['network'], arguments['pretrained'],
                                arguments['px_coordinates'])
            data = arguments['train']

        elif arguments['validate']:
            # Run for validation
            trainer = Trainer(cmd_line=stdscr,
                              validation=arguments['validate'])
            trainer.set_network(arguments['network'], arguments['pretrained'],
                                arguments['px_coordinates'])
            data = arguments['validate']

        elif arguments['infer']:
            raise NotImplementedError("Inference is not yet implemented.")

        else:
            raise ValueError("Must run in one of the possible modes.")

        # Run training or validation
        if arguments['train'] or arguments['validate']:
            trainer.train(data, arguments['batch_size'],
                          arguments['epochs'], arguments['plot'],
                          arguments['weights'], arguments['augment'])
    finally:
        if stdscr is not None:
            stdscr.clear()
            curses.echo()
            curses.nocbreak()
            try:
                curses.curs_set(1)
            except:
                pass
            curses.endwin()

        exception_encountered = traceback.format_exc(0)
        if "SystemExit" in exception_encountered:
             return

        else:
            print("I Died")
            with open(join(getcwd(), "traceback.txt"), mode="w") as file:
                traceback.print_exc(file=file)

            Slacker.send_message("Exception encountered", exception_encountered)
