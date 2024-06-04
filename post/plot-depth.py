#!/bin/env python
#
# Visualize the depth data from the realsense camera
#
# The data must be in numpy format
#

import sys
import argparse

import PIL
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser("Visualize Depth Data")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-o", "--output", action="store", required=False, help="Image to be saved")
parser.add_argument("-n", "--vmin", action="store", required=False, type=int, help="vmin for plot")
parser.add_argument("-x", "--vmax", action="store", required=False, type=int, help="vmax for plot")
parser.add_argument("-t", "--threshold", action="store", required=False, type=int, help="Ground threshold")
parser.add_argument("-c", "--correct", action="store_true", required=False, default=False, help="Correct depth")
parser.add_argument("-g", "--grayscale", action="store_true", required=False, default=False, help="Covert to greyscale")
parser.add_argument("-no", "--normalize", action="store_true", required=False, default=False, help="Normalize between 0 and 1")
parser.add_argument("-hg", "--histogram", action="store_true", required=False, default=False, help="Plot histogram")
arguments = parser.parse_args()

depth = None

try:
    depth = np.load(arguments.input)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))
    sys.exit(-1)

# Clean up the invalid values by setting them to zero
depth[depth == 65535] = 0

if arguments.threshold is not None:
    depth[depth > arguments.threshold] = 0



if arguments.vmin is None:
    minval = np.min(depth[np.nonzero(depth)])
else:
    minval = arguments.vmin

if arguments.vmax is None:
    maxval = np.max(depth[np.nonzero(depth)])
else:
    maxval = arguments.vmax

print(f"There are {len(depth[depth == 0])} readings for 0")
depth[depth == 0] = maxval

if arguments.correct:
    height, width = depth.shape
    standardeviation = np.std(depth, where=depth > 0)

    for h in range(height - 1):
        zeroSeen = False
        for w in range(width - 2):
            if abs(depth[h, w] - depth[h, w + 1]) < standardeviation:
                depth[h, w] = 0

# bins = np.linspace(np.min(depth[np.nonzero(depth)]), np.max(depth), 1000, dtype=float)
# print(bins)
# hist, bins = np.histogram(depth, bins=bins)
# print(hist)

if depth is not None:
    print("Min/Max: {}/{}".format(minval, maxval))
    print(f"Min {minval} Max {maxval} Avg {np.average(depth)} Std Deviation {np.std(depth, where=depth>0)}")
    print("Shape is: {}".format(np.shape(depth)))

    if arguments.normalize:
        normalizedData = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    if arguments.histogram:
        bins = np.linspace(0, 1, 1000, dtype=float)
        hist, bins = np.histogram(normalizedData, bins=bins)
        plt.plot(hist)
        plt.show()
    if arguments.grayscale:
        # Normalize the depth data between 0 and 1
        normalizedData = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        plt.imshow(normalizedData, cmap='gray', vmin=0, vmax=1)
        plt.show()
        # Normalize the depth data between 0 and 255
        # normalizedData = np.zeros_like(depth, dtype="float64")
        # np.copyto(normalizedData, depth)
        # normalizedData *= (255.0 / depth.max())
        # plt.imshow(normalizedData, cmap='gray', vmin=0, vmax=255)
        # plt.show()
    else:
        if arguments.output is not None:
            plt.imsave(arguments.output, depth, vmin=250, vmax=340)
        else:
            plt.imshow(depth, interpolation='none', vmin=minval, vmax=maxval)
            plt.show()

# if depth is not None:
#     im = Image.fromarray(depth)
#     im.save("depth.jpg")


