#
#
import sys
import os

import argparse
import cv2 as cv
import numpy as np
import logging.config
import shutil

import plotly as plot
import plotly.graph_objects as go

from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger

def startupLogger(outputDirectory: str) -> ():
    """
    Initializes two logging systems: the image logger and python centric logging.
    :param outputDirectory: The output directory for the images
    :return: The image logger instance
    """

    # The command line argument contains the name of the YAML configuration file.

    # Confirm the INI exists
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("jetson")

    logger = ImageLogger()
    if not logger.connect(outputDirectory):
        print("Unable to connect to logging. ./output does not exist.")
        sys.exit(1)
    return logger, log

def plot3D(image: np.ndarray, title: str):
    # I can get plotly to work only with square arrays, not rectangular, so just take a subset
    subset = image[0:1500, 0:1500]
    log.debug("Index is {}".format(image.shape))
    log.debug("Subset is {}".format(subset.shape))
    xi = np.linspace(0, subset.shape[0], num=subset.shape[0])
    yi = np.linspace(0, subset.shape[1], num=subset.shape[1])
    zi = subset[:, :, 2]
    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=zi)])

    # The plane represents the threshold value
    # x1 = np.linspace(0, 1500, 1500)
    # y1 = np.linspace(0, 1500, 1500)
    # z1 = np.ones(1500) * planeLocation
    # plane = go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)).T, opacity=0.5, showscale=False, showlegend=False)

    # fig.add_traces([plane])


    # Can't get these to work
    # fig = go.Figure(data=[go.Mesh3d(x=xi, y=yi, z=subset, color='lightpink', opacity=0.50)])
    # fig = go.Figure(data=go.Isosurface(x=xi, y=yi,z=subset, isomin=-1, isomax=1))

    fig.update_layout(title=title, autosize=False,
                      width=1000, height=1000,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()

parser = argparse.ArgumentParser("Plot image values in colorspace")

parser.add_argument("-i", '--image', action="store", required=True, help="Target image")
parser.add_argument("-c", '--color', action="store", required=True, choices=["rgb", "yiq"], help="Color space")
parser.add_argument("-l", '--logging', action="store", required=True, help="Color space")

arguments = parser.parse_args()

# The raw image read from disk
rawImage = cv.imread(arguments.image, cv.IMREAD_COLOR)

(logger, log) = startupLogger("./")

manipulated = ImageManipulation(rawImage, 0, logger)

if arguments.color == "rgb":
    manipulated.toRGB()
    plot3D(manipulated.rgb, "RGB")
