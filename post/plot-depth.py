#! /bin/env python
#
# Visualize the depth data from the realsense camera
#
# The data must be in numpy format
#

import argparse

import PIL
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser("Visualize Depth Data")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
arguments = parser.parse_args()

depth = None

try:
    depth = np.load(arguments.input)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))


if depth is not None:
    print("Min/Max: {}/{}".format(depth.min(), depth.max()))
    print("Shape is: {}".format(np.shape(depth)))
    plt.imshow(depth, interpolation='none', vmin=250, vmax=340)
    plt.show()
    matplotlib.image.imsave("depth.png", depth)

# if depth is not None:
#     im = Image.fromarray(depth)
#     im.save("depth.jpg")


