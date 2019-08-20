#!/usr/bin/env python3
"""Main JSON.

Runs training based on parameters from a JSON file.
"""
import json
import argparse
import warnings
from os import getcwd
from os.path import join
from utils.run_training import run_training
from imgaug import augmenters as iaa



def parse_arguments():
    """Parses command line arguments."""
    description = "trains or validates CurbNet based on a JSON file."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('parameters', metavar='P', type=str, nargs="?",
                        help="path to the JSON file that contains the"
                             "configuration. defaults to train.json in the "
                             "current working directory")

    return parser.parse_args()


def parse_json(json_string, silence=False):
    """Parses the JSON configuration string into a dictionary.

    The JSON file _MUST_ contain the following values:
    - "weights"      : [path to the network and optimizer weights].
    - "mode"         : ["train" | "validate"]. Validate means no backpropagation
                       is done.
    - "data path"    : [path to the data folder].

    The JSON file _CAN_ contain the following values:
    - "batch size"   : int. Defaults to 4.
    - "epochs"       : int. Defaults to 1.
    - "learning rate": float. Defaults to 0.002
    - "optimizer"    : ["sgd" | "adam"]. Defaults to "adam"
    - "augmentation" : bool. True means that the images will be augmented.
                       Defaults to true.
    - "loss weights" : [float, float, float]. Defaults to [0.005, 0.3, 0.695]
    - "plot"         : [path to the plot location]. Defaults to the current
                       working directory.
    - "command line" : bool. True means that the command line interface will be
                       used. Defaults to false.
    - "plot location": [path to the plot file]. This is the directory where the
                       loss, accuracy, and log will be recorded to. Defaults to
                       the current working directory.
    - "px data"      : bool. True means that px coordinates will be added to the
                       4th and 5th dimensions of the input image.
    - "network"      : str. Defaults to "d", can be "e", "f", or "g". Selects
                       the corresponding network.
    Args:
        json_string (str): The json configuration file as a string.
    """
    json_config = json.loads(json_string)

    # Check the JSON file
    if "weights" not in json_config or "mode" not in json_config \
            or "data path" not in json_config:
        raise ValueError("JSON configuration file incomplete.")

    # noinspection PyDictCreation
    out = {"weights": json_config["weights"],
           'train': None,
           'validate': None,
           'infer': None,
           'batch_size': 4,
           'epochs': 1,
           'learning_rate': 0.002,
           'optimizer': 'adam',
           'loss_weights': None,
           'network': 'd',
           'pretrained': False,
           'px_coordinates': True,
           'augment': True,
           'cmd_line': False,
           'criterion': 'mce',
           'plot': join(getcwd(), "plot")}

    out[json_config['mode']] = json_config['data path']

    # Deal with all optionals
    if "batch size" in json_config:
        out['batch_size'] = json_config["batch size"]

    if "epochs" in json_config:
        out['epochs'] = json_config["epochs"]

    if "learning rate" in json_config:
        out['learning_rate'] = json_config["learning rate"]

    if "optimizer" in json_config:
        out['optimizer'] = json_config["optimizer"]

    if "augmentation" in json_config:
        out['augment'] = json_config["augmentation"]

    if "loss weights" in json_config:
        out['loss_weights'] = json_config["loss weights"]

    if "command line" in json_config:
        out['cmd_line'] = json_config['command line']

    if "plot" not in json_config:
        warnings.warn("Plot save path not given. Setting path to the current "
                      "working directory.", UserWarning)
    else:
        out['plot'] = json_config['plot']

    if "pretrained" in json_config:
        out['pretrained'] = json_config['pretrained']

    if "px data" in json_config:
        out['px_coordinates'] = json_config['px data']
    else:
        out['plot'] = json_config['plot']

    if 'criterion' in json_config:
        out['criterion'] = json_config['criterion']

    if 'network' in json_config:
        out['network'] = json_config['network']

    return out


if __name__ == '__main__':
    arguments = parse_arguments()

    fp = arguments.parameters if arguments.parameters else join(getcwd(),
                                                                "train.json")

    with open(fp, mode='r') as file:
        json_str = file.read()

    configuration = parse_json(json_str)

    try:
        run_training(configuration, iaa)
    except KeyboardInterrupt:
        print("User exited program. Killing process.")
