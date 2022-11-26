#
# P L O T
#
# This produces a line plot of the green band of a specific transect point of an image.
# Created to produce a figure for an ENVS508 assignment
#

import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import constants
from ImageManipulation import ImageManipulation
from CameraFile import CameraFile

parser = argparse.ArgumentParser("Image Plot")

parser.add_argument('-i', '--input', action="store", required=True, help="Image")
parser.add_argument('-o', '--output', action="store", required=False, default=".", help="Output directory")
parser.add_argument('-t','--type', action="store", required=True, help="Type of plot")
parser.add_argument('-b', '--band', action="store", required=False, default="red", help="Band to plot")
arguments = parser.parse_args()

# Get the image from disk
image = cv.imread(arguments.input, cv.IMREAD_COLOR)

manipulated = ImageManipulation(image,0)

band = image[:,:,2]

TRANSECT_LOCATION = 1500

transect= band[TRANSECT_LOCATION,:]

x = np.arange(len(transect))
plt.title("Red band transect of image at location 2000")
plt.xlabel("Pixel Position")
plt.ylabel("Pixel Value")
plt.plot(x,transect)
plt.savefig(arguments.output + "/" + "transect-plot.png")
plt.show()

cv.line(image, (0, TRANSECT_LOCATION), (len(transect), TRANSECT_LOCATION), (0, 0, 255), 3, cv.LINE_AA)
cv.imwrite(arguments.output + "/" + "transect-image.jpg", image)








