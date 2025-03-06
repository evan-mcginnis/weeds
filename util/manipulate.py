import argparse
import numpy as np

import cv2 as cv
import logging.config

from ImageLogger import ImageLogger
from ImageManipulation import ImageManipulation
parser = argparse.ArgumentParser("Image Manipulation")

colorSpaces = ["rgb", "yiq", "hsi"]

parser.add_argument('-i', '--input', action="store", required=True, help="Input Image")
parser.add_argument('-o', '--output', action="store", required=True, help="Output Image")
parser.add_argument('-c', '--color', action="store", required=False, help="Target color space")
parser.add_argument('-e', '--equalize', action="store_true", required=False, default=False, help="Histogram equalization")
parser.add_argument('-l', '--logging', action="store", required=True, help="Logging configuration")
parser.add_argument('-a', '--average', action="store_true", required=False, help="Apply averaging kernel")
parser.add_argument('-lbp', '--lbp', action="store_true", required=False, default=False, help="Transform with Local Binary Pattern")

arguments = parser.parse_args()

# Initialize logging
logging.config.fileConfig(arguments.logging)
log = logging.getLogger("jetson")

logger = ImageLogger()
# Get the image from disk
bgr = cv.imread(arguments.input, cv.IMREAD_COLOR)

if bgr is None:
    print(f"Unable to access: {arguments.input}")
    exit(-1)

sizeY, sizeX, _ = bgr.shape
manipulated = ImageManipulation(bgr, 0, logger)

# Equalize the histograms to improve the contrast
if arguments.equalize:
    manipulated.equalizeContrast()
    dst = manipulated.image

if arguments.lbp:
    print(f"Not yet supported")
    exit(-1)

if arguments.color is not None:
    if arguments.color == "yiq":
        yiq = manipulated.toYIQ().astype(np.int)
        if arguments.equalize:
            yiq[:, :, 0] = cv.equalizeHist(yiq[:, :, 0])
        dst = yiq

    if arguments.color == "hsv":
        hsv = manipulated.toHSV()
        if arguments.equalize:
            hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
        dst = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

elif arguments.average is not None and arguments.average:
    #dst = averageBlur = cv.blur(bgr, (5, 5))
    dst = cv.medianBlur(bgr, 5)

#src = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

#dst = cv.equalizeHist(src)

cv.imwrite(arguments.output, dst)


