"""Normalization Constants.

Computes the mean and standard deviation of a dataset.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>

References:
    Computing the mean and std of dataset.
    <https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949>
"""
import argparse
from os.path import join
from utils.mapillarydataset import MapillaryDataset
from torch.utils.data.dataloader import DataLoader

def process_folder(path, batch_size=32):
    """Computes the mean and standard deviation for images in a folder.

    Computes the mean and standard deviation, averaged by batch, and returns
    the results in a file named "normalization_constants.txt" in the same
    directory as was given for the data path.
    """
    print("Loading dataset.")
    dataset = MapillaryDataset(path, True)
    loader = DataLoader(dataset, batch_size)
    print("Dataset loaded.")

    mean = 0.
    std = 0.
    total = float(len(dataset))

    print("Computing mean and standard deviation.")
    for data in enumerate(loader):
        batch_samples = data[1]["raw"].size(
            0)  # batch size (the last batch can have smaller size!)
        images = data[1]["raw"].view(batch_samples, data[1]["raw"].size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= total
    std /= total
    print("Done computing.")
    print("    mean: {}".format(mean))
    print("    std:  {}".format(std))

    # Now output the constants
    fp = join(path, "normalization_constants.txt")
    with open(fp, "w") as file:
        file.writelines(["{}\n".format(mean),
                         "{}\n".format(std)])

    print("Results written to {}".format(fp))

def parse_arguments():
    """Parses arguments."""
    description = "resizes and crops images from the dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('path', metavar='P', type=str,
                        help="path to the mapillary subfolder")
    parser.add_argument('batch_size', metavar='B', type=int, nargs='?',
                        help="size of the batch to run")
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_arguments()
    process_folder(args.path)