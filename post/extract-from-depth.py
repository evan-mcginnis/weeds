import argparse
import sys
import time

import numpy as np

from typing import Tuple
import scipy.ndimage
from PIL import Image

import constants
from OptionsFile import OptionsFile
from DepthImage import DepthImage

parser = argparse.ArgumentParser("Extract from Depth Data")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-ini", "--ini", action="store", required=True, help="Options INI")
parser.add_argument("-o", "--output", action="store", required=True, help="Depth data file in npy format")
arguments = parser.parse_args()

depth = None

optionsLoaded = False
try:
    options = OptionsFile(arguments.ini)
    optionsLoaded = options.load()
except Exception as e:
    print("Unable to load {}".format(arguments.ini))
    sys.exit(-1)
if not optionsLoaded:
    print("Unable to load {}".format(arguments.ini))
    sys.exit(-1)

# Get the locations of the legos in the RGB and depth images
try:
    legoARGB = options.tuple(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_LEGO_A)
    legoBRGB = options.tuple(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_LEGO_B)
except KeyError:
    print("Expected to find lego locations in [{}] {}/{}".format(constants.PROPERTY_SECTION_CAMERA,
                                                                 constants.PROPERTY_LEGO_A,
                                                                 constants.PROPERTY_LEGO_B))
    sys.exit(-1)

print("RGB Lego A: ({}/{})".format(legoARGB[0], legoARGB[1]))
print("RGB Lego B: ({}/{})".format(legoBRGB[0], legoBRGB[1]))

try:
    legoADepth = options.tuple(constants.PROPERTY_SECTION_DEPTH, constants.PROPERTY_LEGO_A)
    legoBDepth = options.tuple(constants.PROPERTY_SECTION_DEPTH, constants.PROPERTY_LEGO_B)
except KeyError:
    print("Expected to find lego locations in [{}] {}/{}".format(constants.PROPERTY_SECTION_DEPTH,
                                                                 constants.PROPERTY_LEGO_A,
                                                                 constants.PROPERTY_LEGO_B))
    sys.exit(-1)

print("Depth Lego A: ({}/{})".format(legoADepth[0], legoADepth[1]))
print("Depth Lego B: ({}/{})".format(legoBDepth[0], legoBDepth[1]))

try:
    rgbResolution = options.tuple(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_RESOLUTION)
except KeyError:
    print("Expected to find camera resolution in [{}] {}".format(constants.PROPERTY_SECTION_CAMERA,
                                                                 constants.PROPERTY_RESOLUTION))
    sys.exit(-1)
# Load the depth data

try:
    depth = np.load(arguments.input)
    depthImage = DepthImage(depth)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))
    sys.exit(-1)

if depth is not None:
    print("Max X is: {}".format(depth.max()))
    print("Shape is: {}".format(np.shape(depth)))


subset = depthImage.extractScaledSubset(legoARGB, legoBRGB, legoADepth, legoBDepth, rgbResolution)
depthImage.save(arguments.output)


