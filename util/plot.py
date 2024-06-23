#
# P L O T
#
# This produces a line plot of the green band of a specific transect point of an image.
# Created to produce a figure for an ENVS508 assignment
#
import math

import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import logging.config

import pandas as pd

from ImageLogger import ImageLogger
import constants
from ImageManipulation import ImageManipulation
from CameraFile import CameraFile

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale

from skimage.color import rgb2yiq, yiq2rgb

parser = argparse.ArgumentParser("Image Plot")

colorSpaces = ["rgb", "yiq", "hsi", "hsv", "cielab", "ycbcr"]
plotTypes = ["transect", "band", "histogram"]

bandNames = {
    "RGB": ["Red", "Green", "Blue"],
    "BGR": ["Blue", "Green", "Red"],
    "HSI": ["Hue", "Saturation", "Intensity"],
    "YIQ": ["Luma", "In-Phase", "Quadrature"],
    "HSV": ["Hue", "Saturation", "Value"],
    "CIELAB": ["Lightness", "A", "B"],
    "YCBCR": ["Luma", "Blue-difference", "Red-difference"],
    "YUV": ["Luma", "Croma-U", "Chroma-V"]
}
def magnitude(x):
    #print(f"X={x}")
    return int(math.floor(math.log10(abs(x))))

parser.add_argument('-i', '--input', action="store", required=True, help="Image")
parser.add_argument('-o', '--output', action="store", required=False, default=".", help="Output directory")
parser.add_argument('-t', '--type', action="store", required=False, default="transect", choices=plotTypes, help="Type of plot")
parser.add_argument('-y', '--y', action="store", required=False, default=1200, type=int, help="Y coordinate for transect")
parser.add_argument('-x', '--x', action="store", required=False, default=1200, type=int, help="X coordinate for transect")
parser.add_argument('-len', '--length', action="store", required=False, default=-1, type=int, help="Transect length")
parser.add_argument('-b', '--band', action="append", required=True, choices=[0, 1, 2], type=int, nargs='+', help="Band to plot")
parser.add_argument('-hb', '--bins', action="store", required=False, default=25, type=int, help="Number of bins")
parser.add_argument('-c', '--color', action="store", required=False, default="rgb", choices=colorSpaces, help="Color space")
parser.add_argument('-l', '--logging', action="store", required=True, help="Logging configuration")
parser.add_argument('-n', '--normalize', action="store_true", required=False, default=False, help="Normalize band to range 0..1")
parser.add_argument('-r', '--rgb', action="store_true", required=False, default=False, help="Show the corresponding RGB band (band plot only")
parser.add_argument('-s', '--subject', action="store", required=True, help="Subject of graph")
arguments = parser.parse_args()

if arguments.rgb and arguments.type != "band":
    print(f"-rgb option applies only to band plots")
    exit(-1)

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
    #image[:, :, 0] = cv.equalizeHist(image[:, :, 0])
    #rgb = yiq2rgb(image)
elif arguments.color == "hsi":
    image = manipulated.toHSI()
elif arguments.color == "cielab":
    image = manipulated.toCIELAB()
elif arguments.color == "ycbcr":
    image = manipulated.toYCBCR()
    image[:, :, 0] = cv.equalizeHist(image[:, :, 0])
elif arguments.color == "hsv":
    image = manipulated.toHSV().astype(np.uint8)
    image[:, :, 2] = cv.equalizeHist(image[:, :, 2])
    #rgb = cv.cvtColor(image,cv.COLOR_HSV2BGR)
else:
    print(f"Unknown color space: {arguments.color}")
    exit(-1)

TRANSECT_LOCATION = arguments.y

band = image[:, :, arguments.band[0]]

# Copy begin
height, width, _ = image.shape

transects = []
axisRequired = 0
maxMagnitude = 0
minMagnitude = 0

if arguments.length == -1:
    for bandNumber in range(len(arguments.band)):
        transect = image[:, arguments.y, arguments.band[bandNumber]]
        print(f"Band {bandNumber} range: {transect.min()} to {transect.max()}")
        transects.append(transect)
else:
    # The transect is from the x specified to the width
    if arguments.x not in range(width):
        print(f"Specified x outside width of image {width}")
        exit(-1)
    if arguments.x + arguments.length not in range(width):
        print(f"Specified x + length exceeds width of image {width}")
        exit(-1)
    stop = arguments.x + arguments.length
    #transect = image[arguments.x:stop, arguments.y, arguments.band[0]]
    for bandNumber in range(len(arguments.band[0])):
        start = arguments.x
        stop = arguments.x + arguments.length
        transect = image[arguments.y, start:stop, arguments.band[0][bandNumber]]
        print(f"Band {bandNumber} range: {transect.min()} to {transect.max()} Magnitude {magnitude(transect.max())}")

        # Determine how many axis are required for the plot.
        if magnitude(transect.max()) > maxMagnitude:
            axisRequired += 1
            maxMagnitude = magnitude(transect.max())
        elif magnitude(transect.max()) < minMagnitude:
            axisRequired += 1
            minMagnitude = magnitude(transect.max())

        transects.append(transect)

    print(f"Plot requires {axisRequired} axis")
    if axisRequired > 2:
        print("The number of axis required is > 2. Unable to graph data.")
        exit(-1)

# Difference between two bands
if arguments.type == "band":
    difference = [0.0] * len(transects[0])
    for i in range(len(transects[0])):
        difference[i] = max(abs(transects[2][i]), abs(transects[1][i])) - min(abs(transects[2][i]), abs(transects[1][i]))
        #difference[i] = math.log2(abs(float(transects[2][i]) - float(transects[1][i])))
        #difference[i] = math.log2(abs(float(transects[1][i]))) * transects[0][i]

    print(f"Difference range: {min(difference)}-{max(difference)}")

# Copy end

if arguments.normalize:
    scaler = MinMaxScaler(feature_range=(0, 1))
    bandDF = pd.DataFrame(band)
    band = scaler.fit_transform(bandDF)



if arguments.rgb:
    rgbBand = rgb[:, :, arguments.band]
    rgbTransect = rgbBand[TRANSECT_LOCATION, :]

    # Normalize the data to match th min/max of the other color space

    scaler = MinMaxScaler(feature_range=(band.min(), band.max()))
    rgbDF = pd.DataFrame(rgbTransect)
    scaled = scaler.fit_transform(rgbDF)
    scaledRGB = minmax_scale(rgbTransect)
    print(f"{scaledRGB}")



#transect = band[TRANSECT_LOCATION, :]
colors = ["red", "green", "blue"]
if arguments.type == "band":

    # Lines in the plot, irrespective of axis
    lines = []
    x = np.arange(len(transects[0]))
    #plt.figure()
    fig, ax1 = plt.subplots(layout='constrained')
    # Make a long plot
    fig.set_figwidth(15)
    plt.title(f"{arguments.color.upper()} transect of image at location ({arguments.x},{arguments.y}) - {arguments.subject}")
    ax1.set_xlabel("Pixel Offset")
    ax1.set_ylabel("Pixel Value")

    # If more than one axis is required, twin the x axis
    if axisRequired > 1:
        ax2 = ax1.twinx()
    else:
        ax2 = ax1

    #plt.ylim(-0.2, 1.0)
    ax1Ylabel = ""
    ax2Ylabel = ""
    for i in range(len(transects)):
        # Determine which axis to use.  Axis 1 is the bigger magnitude
        if magnitude(transects[i].max()) == maxMagnitude:
            axis = ax1
            ax1Ylabel += bandNames[arguments.color.upper()][int(arguments.band[0][i])] + " "
            ax1.set_ylabel(ax1Ylabel, color=colors[i])
            ax1.tick_params(axis='y', labelcolor=colors[i])
        else:
            axis = ax2
            ax2Ylabel += bandNames[arguments.color.upper()][int(arguments.band[0][i])] + " "
            ax2.set_ylabel(ax2Ylabel, color=colors[i])
            ax2.tick_params(axis='y', labelcolor=colors[i])

        lines.extend(axis.plot(x, transects[i], marker='.', markersize=5, color=colors[i]))

    axis.plot(x, difference, marker='.', markersize=3, color='black')
    #plt.scatter(x, transect, s=1)
    if arguments.rgb:
        plt.scatter(x, scaled, s=1, color='red')
        plt.legend([arguments.color.upper(), "RGB"], loc="lower right")
    else:
        #plt.legend([arguments.color.upper()], loc="lower right")
        legendItems = []
        for selectedBand in arguments.band[0]:
            legendItems.append(bandNames[arguments.color.upper()][selectedBand])
        print(f"Legend: {legendItems}")

        ax1.legend(lines, legendItems)
    plt.savefig(arguments.output + "/" + "transect-plot.png")
    plt.show()

    log.debug(f"Draw line from ({arguments.x},{arguments.y}) to ({arguments.x + arguments.length},{arguments.y})")
    cv.line(rgb, (arguments.x, arguments.y), (arguments.x + arguments.length, arguments.y), (0, 0, 255), 3, cv.LINE_AA)
    #cv.line(rgb, (0, TRANSECT_LOCATION), (sizeX, TRANSECT_LOCATION), (0, 0, 255), 3, cv.LINE_AA)
    cv.imwrite(arguments.output + "/" + "transect-image.jpg", rgb)

elif arguments.type == "transect":
    plt.figure()
    plt.title(f"{arguments.color.upper()} Band {arguments.band} transect of image at location {arguments.y}")
    plt.xlabel("Pixel Position")
    plt.ylabel("Pixel Value")
    fig, ax = plt.subplots()
    ax.imshow(image)
    x = np.arange(0, sizeY)
    y = np.repeat(sizeX, TRANSECT_LOCATION)
    ax.plot(y, x, '--', linewidth=5, color='firebrick')
    plt.show()

elif arguments.type == "histogram":
    plt.figure()
    plt.title(f"{arguments.subject}")
    imgGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Compute the histogram
    allBins = np.arange(0, arguments.bins)
    counts, bins = np.histogram(imgGray, arguments.bins)
    plt.xlabel("Greyscale Pixel Value")
    plt.ylabel("Count")
    plt.hist(imgGray.ravel(), bins=128, color='skyblue', histtype='step', fill=True)
    plt.savefig(arguments.output + "/" + "histogram.jpg")
    plt.show()









