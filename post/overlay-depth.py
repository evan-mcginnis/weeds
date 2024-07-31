
import sys
import argparse

import PIL
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from PIL import Image
import cv2 as cv

parser = argparse.ArgumentParser("Overlay Depth Data on image")
parser.add_argument("-d", "--depth", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-i", "--image", action="store", required=True, help="RGB image in JPG format")
parser.add_argument("-n", "--vmin", action="store", required=False, default=250, type=int, help="vmin for plot")
parser.add_argument("-x", "--vmax", action="store", required=False, default=340, type=int, help="vmax for plot")

parser.add_argument("-o", "--output", action="store", required=False, help="Image to be saved")
arguments = parser.parse_args()

depth = None

try:
    depth = np.load(arguments.depth)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))
    sys.exit(-1)

if arguments.vmin is not None:
    minval = np.min(depth[np.nonzero(depth)])
else:
    minval = arguments.vmin

if arguments.vmax is not None:
    maxval = np.max(depth[np.nonzero(depth)])
else:
    maxval = arguments.vmax


print("Min/Max: {}/{}".format(minval, maxval))
print("Shape is: {}".format(np.shape(depth)))
if arguments.output is not None:
    plt.imsave(arguments.output, depth, vmin=minval, vmax=maxval)
else:
    plt.imshow(depth, interpolation='none', vmin=minval, vmax=maxval)
    plt.show()
