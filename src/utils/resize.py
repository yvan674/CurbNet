"""Resize.

Resizes all images in a folder to the specified value.
Here we resize all images to 360px, since that seems to be a good size that fits
a batch of 32 in the GPU memory.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
from .. import constants
import PIL.Image as Image
from os.path import join, exists
from os import mkdir
import argparse
from time import time
from datetime import timedelta


WIDTH = constants.DIMENSIONS[0]
HEIGHT = constants.DIMENSIONS[1]


def crop_and_resize(image, sampling=Image.BICUBIC):
    """Resize and crops image to defined values.

    First crops the image to 4:3 width to height ratio around the center, then
    resizes it to a predetermined width and height.

    Args:
        image (PIL.Image.Image): The image object to be manipulated.
        sampling (PIL.Image.Filters*): The sampling method. Defaults to bicubic.

    Returns:
        PIL.Image.Image): The resized and cropped image.
    """
    if image.width == WIDTH \
            and image.height == HEIGHT:
        return image

    # Calculate what the cropped size should be based on dimensions
    if 4 * image.width >= 3 * image.height:
        # the height is the smaller element
        cropped_height = image.height
        cropped_width = int(image.height * 1.33333333333333)
    else:
        cropped_height = int(image.width * .75)
        cropped_width = image.width

    # Calculate new cropped dimensions from center of image
    box = ((image.width - cropped_width) / 2,
           (image.height - cropped_height) / 2,
           (image.width + cropped_width) / 2,
           (image.height + cropped_height) / 2)

    image_cropped = image.crop(box)
    image_resized = image_cropped.resize((WIDTH, HEIGHT), sampling)

    return image_resized


def process_folder(folder_path):
    """Processes a mapillary folder (e.g. training) for resizing and cropping.

    Creates a new folder adjacent to the folder path given with the resized
    images. Only processes images which are in the viable.txt file.

    Args:
        folder_path (str): The path to the mapillary folder.

    Returns:
        none
    """
    total_files = 0
    files_processed = 0
    start_time = time()

    resized_folder_path = folder_path + "_resized"

    # Make new folders
    if not exists(resized_folder_path):
        mkdir(resized_folder_path)
    if not exists(join(resized_folder_path, "images")):
        mkdir(join(resized_folder_path, "images"))
    if not exists(join(resized_folder_path, "labels")):
        mkdir(join(resized_folder_path, "labels"))

    # First open the file to count the number of lines so we can see our
    # progress
    with open(join(folder_path, "viable.txt"), "r") as viable_file:
        for _ in viable_file:
            total_files += 1

    with open(join(folder_path, "viable.txt"), "r") as viable_file:
        for line in viable_file:
            line = line.rstrip()
            # Get image file names
            image_path = join(folder_path, "images", line + ".jpg")
            label_path = join(folder_path, "labels", line + ".png")

            image = Image.open(image_path)
            label = Image.open(label_path)
            # Resize the image
            resized_image = crop_and_resize(image)
            # Resize the label
            resized_label = crop_and_resize(label, Image.NEAREST)
            # Save them
            resized_image_path = join(resized_folder_path, "images",
                                      line + ".jpg")
            resized_label_path = join(resized_folder_path, "labels",
                                      line + ".png")
            resized_image.save(resized_image_path, image.format)
            resized_label.save(resized_label_path, label.format)

            files_processed += 1
            if files_processed % 10 == 0:
                # Calculate time taken for per file for the last 10 files
                rate = 10 / time() - start_time

                # Calculate time left
                files_left = total_files - files_processed
                time_left = int(files_left / rate)
                time_left = timedelta(seconds=time_left)

                # Print progress every 10 files
                print("{} of {} files processed. Time left: {}".format(
                    files_processed,
                    total_files,
                    time_left)
                )
                start_time = time()  # Reset timer

    print("{} of {} files processed. Program completed.".format(files_processed,
                                                                total_files))


def parse_arguments():
    """Parses arguments."""
    description = "resizes and crops images from the dataset"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('path', metavar='P', type=str,
                        help="path to the mapillary subfolder")
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_arguments()
    process_folder(args.path)
