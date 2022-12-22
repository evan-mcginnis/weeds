#
# Visualize the transect of the laser level
#
#

import argparse

import PIL
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import cv2 as cv
from PIL import Image

parser = argparse.ArgumentParser("Visualize transect")
parser.add_argument("-i", "--input", action="store", required=True, help="Image to process")
parser.add_argument("-t", "--threshold", action="store", required=False, default=50, help="Threshold of spike")
parser.add_argument("-p", "--position", action="store", required=False, default=1000, type=int, help="Position along x axis")
arguments = parser.parse_args()

image = None
RED = 2

try:
    image = cv.imread(arguments.input, cv.IMREAD_COLOR)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))

# Extract the transect
transect = image[:, arguments.position, RED]

plt.plot(transect)
plt.xlabel('Y Position')
plt.ylabel('Pixel Value')
plt.title('Red values of transect through X={}'.format(arguments.position))
plt.text(30, 62, "Max value = {} at ({},{})".format(np.max(transect), arguments.position, np.argmax(transect)))
plt.grid(True)
plt.show()

# if image is not None:
#     plt.imshow(image, interpolation='none', vmin=430, vmax=470)
#     plt.show()

