"""Selector.

The purpose of this script it to produce a list of all viable files from the
dataset and output the list into a text file that is put within the mapillary
subfolder (e.g. mapillary/training/viable.txt).

Viable is defined as having at least one pixel classified as a curb.
"""
from PIL import Image
import numpy as np
import os
import argparse
from time import time
from datetime import timedelta


def process_folder(path):
    """Processes a subfolder.

    Args:
        path (str): The path to the subfolder.
    """
    start_time = time()  # Timer for time left measurement

    labels_dir = os.path.join(path, "labels")
    images = os.listdir(labels_dir)
    output_file_path = os.path.join(path, "viable.txt")
    pruned_list = []

    total = len(images)

    for idx, image in enumerate(images):
        # Open the image
        img = Image.open(os.path.join(labels_dir, image))

        # Turn it into a numpy array for calculations
        img = np.array(img)

        if np.count_nonzero(img == 2):
            pruned_list.append(os.path.splitext(image)[0])

        if idx % 10 == 0 and idx != 0:
            # Calculate time left
            rate = float(idx) / (time() - start_time)
            files_left = total - idx
            time_left = int(files_left / rate)
            time_left = timedelta(seconds=time_left)
            print("Processing image {} of {}. Time left: {}".format(idx, total,
                                                                    time_left))

    # Check for correctness
    # if not len(pruned_list) == 15159:
    #     raise ValueError("Wrong number of curb photos. Calculated {} photos."
    #                      .format(len(pruned_list)))

    with open(output_file_path, 'w+') as output_file:
        for line in pruned_list:
            output_file.write(line + "\n")

    print("Number of viable images: {}".format(len(pruned_list)))
    print("Process finished. Wrote list of viable files to:\n{}"
          .format(output_file_path))


def parse_arguments():
    """Parses arguments."""
    description = "selects only viable images from the dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('path', metavar='P', type=str, nargs=1,
                        help="path to the mapillary subfolder")
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_arguments()
    process_folder(args.path[0])
