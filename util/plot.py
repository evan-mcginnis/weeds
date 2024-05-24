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
import logging.config

from ImageLogger import ImageLogger
import constants
from ImageManipulation import ImageManipulation
from CameraFile import CameraFile

parser = argparse.ArgumentParser("Image Plot")

colorSpaces = ["rgb", "yiq"]
plotTypes = ["transect", "band", "histogram"]

parser.add_argument('-i', '--input', action="store", required=True, help="Image")
parser.add_argument('-o', '--output', action="store", required=False, default=".", help="Output directory")
parser.add_argument('-t', '--type', action="store", required=False, default="transect", choices=plotTypes, help="Type of plot")
parser.add_argument('-y', '--yloc', action="store", required=False, default=1200, type=int, help="Y coordinate for transect")
parser.add_argument('-b', '--band', action="store", required=False, default=0, type=int, help="Band to plot")
parser.add_argument('-hb', '--bins', action="store", required=False, default=25, type=int, help="Number of bins")
parser.add_argument('-c', '--color', action="store", required=False, default="rgb", choices=colorSpaces, help="Color space")
parser.add_argument('-l', '--logging', action="store", required=True, help="Logging configuration")
arguments = parser.parse_args()

# Initialize logging
logging.config.fileConfig(arguments.logging)
log = logging.getLogger("jetson")

logger = ImageLogger()
# Get the image from disk
rgb = cv.imread(arguments.input, cv.IMREAD_COLOR)

if rgb is None:
    print(f"Unable to access: {arguments.input}")
    exit(-1)

sizeY, sizeX, _ = rgb.shape
manipulated = ImageManipulation(rgb, 0, logger)

if arguments.color == "rgb":
    image = rgb
elif arguments.color == "yiq":
    image = manipulated.toYIQ()
else:
    print(f"Unknown color space: {arguments.color}")

band = image[:, :, arguments.band]

TRANSECT_LOCATION = arguments.yloc

transect = band[TRANSECT_LOCATION, :]

if arguments.type == "band":
    x = np.arange(len(transect))
    plt.figure()
    plt.title(f"Band {arguments.band} transect of image at location {arguments.yloc}")
    plt.xlabel("Pixel Position")
    plt.ylabel("Pixel Value")
    #plt.plot(x, transect, marker='.', markersize=5)
    plt.scatter(x, transect, s=1)
    plt.savefig(arguments.output + "/" + "transect-plot.png")
    plt.show()

    log.debug(f"Draw line from (0,{TRANSECT_LOCATION}) to ({sizeX},{TRANSECT_LOCATION})")
    cv.line(rgb, (0, TRANSECT_LOCATION), (sizeX, TRANSECT_LOCATION), (0, 0, 255), 3, cv.LINE_AA)
    cv.imwrite(arguments.output + "/" + "transect-image.jpg", rgb)

elif arguments.type == "transect":
    plt.figure()
    plt.title(f"Band {arguments.band} transect of image at location {arguments.yloc}")
    plt.xlabel("Pixel Position")
    plt.ylabel("Pixel Value")
    fig, ax = plt.subplots()
    ax.imshow(image)
    x = np.arange(0, sizeY)
    y = np.repeat(sizeX, TRANSECT_LOCATION)
    ax.plot(y, x, '--', linewidth=5, color='firebrick')
    plt.show()

elif arguments.type == "histogram":
    # Compute the histogram
    counts, bins = np.histogram(image, arguments.bins)
    allBins = np.arange(0, len(bins))
    plt.hist(counts, bins=allBins)
    plt.show()









