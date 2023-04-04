#
# Visualize the transect of the laser level
#
#
# This will plot the values of the red band of an image, identifying the maximum value found there.
# The image tested is black with the exception of the red laser beam shown horizontally across the image.

import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import sys
import os

parser = argparse.ArgumentParser("Visualize transect")
parser.add_argument("-i", "--input", action="store", required=True, help="Image to process")
parser.add_argument("-p", "--position", action="store", required=False, default=1000, type=int, help="Position along x axis")
parser.add_argument("-b", "--band", action="store", required=False, default="red", choices=["red", "blue", "green"], type=str, help="BGR band")
arguments = parser.parse_args()

image = None
BLUE = 0
GREEN = 1
RED = 2

if not os.path.exists(arguments.input):
    print("Unable to find file: {}".format(arguments.input))
    sys.exit(-1)

try:
    image = cv.imread(arguments.input, cv.IMREAD_COLOR)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))
    sys.exit(-1)

if image is None:
    print("Unable to read image: {}".format(arguments.input))
    sys.exit(-1)

# Extract the transect
if arguments.band == "red":
    band = RED
elif arguments.band == "blue":
    band = BLUE
elif arguments.band == "green":
    band = GREEN

transect = image[:, arguments.position, band]

plt.plot(transect)
plt.xlabel('Y Position')
plt.ylabel('Pixel Value')
plt.title('{} values of transect through X={}'.format(arguments.band, arguments.position))
plt.text(30, 62, "Max value = {} at ({},{})".format(np.max(transect), arguments.position, np.argmax(transect)))
plt.grid(True)
plt.show()

sys.exit(0)



