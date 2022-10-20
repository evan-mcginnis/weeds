#
# Visualize the depth data from the realsense camera
#
# The data must be in numpy format
#

import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser("Visualize Depth Data")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
arguments = parser.parse_args()

depth = None

try:
    depth = np.load(arguments.input)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))


if depth is not None:
    plt.imshow(depth, interpolation='none')
    plt.show()


