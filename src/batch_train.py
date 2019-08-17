#!/usr/bin/env python3
"""Batch train.

Trains a batch of training sessions in series. Reads the configurations from a
json file as interpreted by main_json.py.

Usage:
    batch_train.py [path to json file]

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import json
import main_json
import argparse
from os import getcwd, remove
from os.path import join
from utils.run_training import run_training
import re
from imgaug import augmenters as iaa


def parse_arguments():
    """Parses command line arguments."""
    description = "trains or validates a series of configurations for CurbNet" \
                  " based on a JSON file"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('parameters', metavar='P', type=str, nargs="?",
                        help = "path to the JSON file that contains the"
                               "configuration. defaults to train.json in the "
                               "current working directory")

    return parser.parse_args()


def parse_json_batch(file_path):
    """Turns the json batch configurations into a list of json strings."""
    with open(file_path, mode='r') as json_file:
        batch = json.load(json_file)

    json_batch = []  # This is the list of json strings
    for session in batch["sessions"]:
        json_batch.append(json.dumps(session))

    return json_batch

def watchdog(configuration_list):
    """Watches over the training sessions and makes sure nothing goes wrong.

    This function watches over the training sessions and if something goes
    wrong, attempts to restart/continue the session without requiring input from
    the user.

    Args:
        configuration_list (list): A list of strings, where each string is the
            json encoded configuration for a single training session.
    """
    i = 0  # counter variable
    current_config = None

    while i < len(configuration_list):
        if not current_config:  # Checks to see if current config exists
            current_config = main_json.parse_json(configuration_list[i])

        run_training(current_config, iaa, silence=True)
        print(current_config['plot'])

        try:
            # Next check the status file to see if it has finished training
            with open(join(current_config['plot'], "status.txt")) as status:
                lines = status.readlines()

            if "Finished Training." in lines[-1]:
                # Means that training completed properly.
                i += 1
                current_config = None
            else:
                epoch = int(re.findall(r"\d+", lines[1])[0]) - 1
                current_config['epochs'] -= epoch
        except FileNotFoundError or IndexError or KeyboardInterrupt as error:
            # Means that something's wrong with the status file, most likely
            # meaning that something is completely wrong with the training
            # session. Delete the weight associated and restart.
            if error == FileNotFoundError or IndexError:
                remove(current_config['weights'])
            else:
                return  # Exit if KeyboardInterrupt


if __name__ == "__main__":
    arguments = parse_arguments()

    fp = arguments.parameters if arguments.parameters else join(getcwd(),
                                                                "train.json")

    config_list = parse_json_batch(fp)

    try:
        watchdog(config_list)

    except KeyboardInterrupt:
        print("User exited program. Killing process.")
