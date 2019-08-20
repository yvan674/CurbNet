"""Mask visualization.

A tool that shows the masked area as Masked Cross Entropy Loss would produce.
This is meant as a tool that only produces the mask. The actual overlay and
visualization should be done with an external program.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import argparse
from PIL import Image
from os.path import join, split, splitext

def parse_args():
    """Parses command line arguments."""
    description = "visualizes the masked area used by Maksed Cross Entropy Loss"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('path', metavar='P', type=str,
                        help='path to the image whose mask is to be visualized.'
                             ' There must be a corresponding image in the label'
                             'folder.')

    return parser.parse_args()


def create_mask(path):
    """Creates the mask.

    Args:
        path (str): The path to the image. A label must be in the corresponding
            label folder.
    """
    image = np.array(Image.open(path))

    label_path = join(split(split(path)[0])[0], 'labels', split(path)[1])
    label_path = splitext(label_path)[0] + '.png'
    label = parse_target(label_path)

    # Extract the road mask from the target
    road = np.zeros((image.shape[0], image.shape[1]), dtype=int)
    road[label == 3] = 1

    # Create b
    b_dim = int(image.shape[0] * 0.03)
    b = np.ones((b_dim, b_dim), dtype=bool)

    # Calculate the road perimeter mask
    mask = np.zeros(label.shape)

    print("Road shape: {}".format(road.shape))
    print("mask.shape: {}".format(mask.shape))
    print("b shape: {}".format(b.shape))

    mask = binary_dilation(road, b)

    # Remove the road itself from the mask
    mask[road == 1] = 0

    # Create output image, and make it blue
    output = np.zeros_like(image)
    output[mask == 1] = np.array([0, 0, 255])

    mask = Image.fromarray(np.uint8(output))

    Image.open(path).show()

def parse_target(path):
    """Parses the target segmentation image to a numpy array.

    Returns:
        np.array: The parsed target segmentation
    """
    image = np.array(Image.open(path))
    out_array = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    out_array[image == 2] = 1  # Label for curb
    out_array[image == 9] = 2  # Label for curb cut

    # All road labels, including any markings
    # 10: parking
    # 13: road
    # 14: service lane
    # 23: Crosswalk
    # 24: Marking
    # 41: Manhole
    # 43: Pothole
    labels = [10, 13, 14, 23, 24, 41, 43]
    for i in labels:
        out_array[image == i] = 3

    return out_array

if __name__ == '__main__':
    args = parse_args()
    create_mask(args.path)
