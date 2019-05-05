"""Stats.

This module is intended to calculate some statistics of the mapillary dataset.
Specifically, it calculates the number of images, the min dimensions, the max
dimensions, the number of photos of each dimension, the percentage of images
with both curbs and curb cuts, percentage of images with curbs, and the
percentage of images with curb cuts. It also gives the average amount of curbs,
the average amount of curbs in images where a curb actually exists, the average
amount of curb cuts, and the average amount in images where curb cuts actually
exist.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import os
import argparse
from PIL import Image
import numpy as np

# Import sound only on windows
try:
    import winsound
except ImportError:
    winsound = None

CURB_LABEL = 2
CURB_CUT_LABEL = 9


def parse_arguments():
    """Parses arguments."""
    description = "gathers statistics on the mapillary dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('path', metavar='P', type=str, nargs=1,
                        help="path to the mapillary dataset")
    arguments = parser.parse_args()
    return arguments


def main(path):
    folders = ["training"]
    dimensions = {}
    curbs = []
    curb_cuts = []
    min_dim = [200000, 200000]
    max_dim = [0, 0]

    for folder in folders:
        folder_stats = {"total": None,
                        "max dim": None,
                        "min dim": None,
                        "dims": None,
                        "percent with curbs": None,
                        "percent with cuts": None,
                        "avg curbs total": None,
                        "avg curbs exists": None,
                        "avg cuts total": None,
                        "avg cuts exists": None}

        current_dir = os.path.join(path, folder)
        images_dir = os.path.join(current_dir, "images")
        labels_dir = os.path.join(current_dir, "labels")

        images = os.listdir(images_dir)

        # count number
        folder_stats["total"] = len(images)

        # Process each photo
        for idx, image in enumerate(images):
            # First split the file extension
            image = os.path.splitext(image)[0]

            image_stats = process_image(image, labels_dir)

            # Deal with dimensions
            if image_stats["dim"] in dimensions:
                dimensions[image_stats["dim"]] += 1
            else:
                dimensions[image_stats["dim"]] = 1

            curbs.append(image_stats["curb"])
            curb_cuts.append(image_stats["curb cut"])
            if idx % 10 == 0:
                print("Processing images {} of {}"
                      .format(idx, folder_stats["total"]))

        # get min and max dimensions
        for key in dimensions:
            # Max first
            if key[0] > max_dim[0]:
                max_dim[0] = key[0]
            if key[1] > max_dim[1]:
                max_dim[1] = key[1]

            # Min now
            if key[0] < min_dim[0]:
                min_dim[0] = key[0]
            if key[1] < min_dim[1]:
                min_dim[1] = key[1]

        folder_stats["max dim"] = (max_dim[0], max_dim[1])
        folder_stats["min dim"] = (min_dim[0], min_dim[1])

        folder_stats["dims"] = dimensions

        # Use numpy to process curbs
        curbs = np.array(curbs)
        curb_cuts = np.array(curb_cuts)

        total = float(folder_stats["total"])
        num_curbs = float(np.count_nonzero(curbs))
        num_cuts = float(np.count_nonzero(curb_cuts))

        # Get percentages
        folder_stats["percent with curbs"] = num_curbs / total
        folder_stats["percent with cuts"] = num_cuts / total

        # Get averages for curbs
        folder_stats["avg curbs total"] = np.average(curbs)
        folder_stats["avg curbs exists"] = float(np.sum(curbs)) / num_curbs

        # Get averages for curb cuts
        folder_stats["avg cuts total"] = np.average(curb_cuts)
        folder_stats["avg cuts exists"] = \
            float(np.sum(curb_cuts)) / num_cuts

        # Calculate if every image with curbs have curb cuts
        all_cuts_have_curbs = True
        cuts_without_curbs = 0
        for i in range(len(curbs)):
            if curb_cuts[i]:
                if not curbs[i]:
                    all_cuts_have_curbs = False
                    cuts_without_curbs += 1

        # Calculate percentage of cuts that are a part of curbs

        # Create output strings
        l1 = ["Total images:\t{}".format(folder_stats["total"]),
              "Max dim:\t{}".format(folder_stats["max dim"]),
              "Min dim:\t{}\n".format(folder_stats["min dim"]),
              "List of dims:"]
        for key in folder_stats["dims"]:
            l1.append("\t{}:\t{}".format(key, folder_stats["dims"][key]))

        l2 = ["",
              "Percent curbs:\t{}".format(folder_stats["percent with curbs"]),
              "Percent cuts:\t{}".format(folder_stats["percent with cuts"]),
              "Average curbs:\t{}".format(folder_stats["avg curbs total"]),
              "  where exists:\t{}".format(folder_stats["avg curbs exists"]),
              "Average cuts:\t{}".format(folder_stats["avg cuts total"]),
              "  where exists:\t{}".format(folder_stats["avg cuts exists"]),
              "Cuts in curbs:\t{}".format(all_cuts_have_curbs),
              "Cuts w/o curbs:\t{}".format(cuts_without_curbs)
              ]
        lines = l1 + l2

        stat_path = os.path.join(path, "stats.txt")
        stat_file = open(stat_path, mode='w')

        print("")
        print("==========================")
        print("Writing stats to: {}".format(stat_path))

        for line in lines:
            print(line)
            stat_file.write(line + '\n')

        stat_file.close()

        # Beep to acknowledge program complete
        if os.name == "nt":
            winsound.Beep(440, 500)
            winsound.Beep(784, 500)


def process_image(image_name, labels_dir):
    """Processed the image

    Args:
        image_name (str): Image file name without extension.
        labels_dir (str): The directory of the labelled images.

    Returns:
        dict: A dictionary containing the dimensions, proportion of the image
        that is a curb, and proportion of the image that is a curb cut.
    """
    out = {"dim": (0, 0),
           "curb": 0.,
           "curb cut": 0.}

    labeled = Image.open(os.path.join(labels_dir, image_name + ".png"))

    out["dim"] = labeled.size
    total_pixels = labeled.size[0] * labeled.size[1]

    # Use a numpy array to calculate curb and curb cut proportions
    labeled = np.array(labeled)

    # Note: We use np.count_nonzero instead of np.sum because it is
    # approximately 139% faster than np.sum()
    total_curbs = np.count_nonzero(labeled == CURB_LABEL)
    total_curb_cuts = np.count_nonzero(labeled == CURB_CUT_LABEL)

    out["curb"] = float(total_curbs) / float(total_pixels)
    out["curb cut"] = float(total_curb_cuts) / float(total_pixels)

    return out


if __name__ == '__main__':
    args = parse_arguments()
    main(args.path[0])
