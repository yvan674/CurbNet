"""Copier.

Copies n files from one directory to another directory.

Authors:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import shutil
import os
import argparse
import random
import warnings


def parse_args():
    description = "copies n files from one directory to another directory"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('src', type=str, help="source directory")
    parser.add_argument('dest', type=str, help="destination directory")
    parser.add_argument('n', type=int, help='number of files to copy')
    parser.add_argument('-r', '--randomize', action='store_true')

    return parser.parse_args()

def copy_n_files(source, dest, n, randomize):
    """Copies the first n files in source to dest.

    Args:
        source (str): The source directory path.
        dest (str): The destination directory path.
        n (int): The number of files to copy.
        randomize (bool): Whether or not to randomize which files are chosen.
    """
    if not os.path.isdir(source):
        raise ValueError("Source is not a directory")

    file_list = os.listdir(source)

    # check if n is a legal value
    if n > len(file_list):
        raise ValueError("n is larger than the number of files in source")

    # Check destination directory
    if not os.path.isdir(dest):
        if os.path.isfile(dest):
            raise ValueError("Destination is not a directory.")
        warnings.warn("Destination directory does not exist. Will create it.",
                      UserWarning)
        os.mkdir(dest)

    if not len(os.listdir(dest)) == 0:
        raise ValueError("Destination directory is not empty.")

    if randomize:
        copy_list = random.sample(file_list, n)  # randomly select n files
    else:
        copy_list = file_list[:n]  # Select first n files

    for file in copy_list:
        shutil.copyfile(os.path.join(source, file), dest)


if __name__ == '__main__':
    args = parse_args()
    copy_n_files(args.src, args.dest, args.n, args.randomize)
