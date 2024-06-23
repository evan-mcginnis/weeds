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

def substitute(image: np.ndarray) -> np.ndarray:
    blueBand = image[:, :, 0]
    greenBand = image[:, :, 1]
    redBand = image[:, :, 2]

    # There os probably a super-elegant way to do this in numpy, but just iterate over the image.
    rows, columns, channels = image.shape
    result = np.zeros_like(image)
    for row in range(rows):
        for column in range(columns):
            if image[row, column, 0] > 0:
                result[row, column] = [0, 0, 255]
            if image[row, column, 1] > 0:
                result[row, column] = [0, 0, 255]
            if image[row, column, 2] > 0:
                result[row, column] = [0, 0, 255]

    return result

parser = argparse.ArgumentParser("Overlay two images")

parser.add_argument('-b', '--base', action="store", required=True, help="Base Image")
parser.add_argument('-t', '--top', action="store", required=True, help="Top (overlay) Image")
parser.add_argument('-o', '--output', action="store", required=False, default="overlay.jpg", help="Output file")
parser.add_argument('-a', '--alpha', action="store", required=False, default=False, type=float, help="Alpha weight")
parser.add_argument('-s', '--subject', action="store", required=True, help="Subject of graph")
arguments = parser.parse_args()

baseImage = cv.imread(arguments.base)
overlayImage = cv.imread(arguments.top)

overlay = substitute(overlayImage).astype(np.uint8)

beta = (1.0 - arguments.alpha)
dst = cv.addWeighted(baseImage, arguments.alpha, overlay, beta, 0.0)

cv.imwrite(arguments.output, dst)

