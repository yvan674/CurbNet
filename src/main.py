"""Main.

The main script that is called to run everything else.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import argparse
from trainer import Trainer

def parse_arguments():
    """Parses arguments.

    The arguments are:
        - Training or Validate and Data path
        - Learning rate
        - Optimizer (Adam or SGD)
        - Batch size
        - Number of epochs
        - Plot path (optional)
        - Inference mode
        - Path to weights
        - Image augmentation
    """

if __name__ == "__main__":
    t = Trainer()
    t.train("", 5, 5, "", "", True)
