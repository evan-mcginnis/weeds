import cv2 as cv
import time
import argparse
import numpy as np
import sys

from CameraFile import CameraPhysical, CameraFile
from ImageManipulation import ImageManipulation
import constants

parser = argparse.ArgumentParser("Calibrate camera & pixel dimensions")

parser.add_argument('-i', '--input', action="store", required=True, help="Calibration photo")
parser.add_argument('-o', '--output', action="store", required=True, help="Output for calibration data")
parser.add_argument('-d', '--distance', action="store", default=76.2, type=float, help="Distance in MM between two dots")
parser.add_argument("-v", "--verbose", action="store_true", default=False)

results = parser.parse_args()

if results.input is not None:
    image = cv.imread(results.input)

manipulation = ImageManipulation(image)

hsv = manipulation.toHSV()

# define range of green color in HSV
# Not the best choice -- we need a color that is not likely to appear in the field

lower_green = np.array([60 - 20,100,100])
upper_green = np.array([60 + 20,255,255])

# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_green, upper_green)
# Bitwise-AND mask and original image
res = cv.bitwise_and(image,image, mask= mask)

if results.verbose:
    cv.imwrite("mask.jpg", mask)
    cv.imwrite("result.jpg", res)

manipulation = ImageManipulation(res)
(contours, hierarchy, blobs, largestName) = manipulation.findBlobs(2)
manipulation.drawBoxes(blobs)

if results.verbose:
    cv.imwrite("calibration-results.jpg", manipulation.image)

# We should have only 2 blobs detected in the carlibration
if len(blobs) != 2:
    print("Incorrect number of blobs detect in image for calibration (" + str(len(blobs)) + ")")
    sys.exit(1)

blob0 = blobs["blob0"]
blob1 = blobs["blob1"]

(x1, y1) = blob0.get(constants.NAME_CENTER)
(x2, y2) = blob1.get(constants.NAME_CENTER)

# Take the absolute value, as the blobs can be in any order
pixelDistance = np.abs(x2 - x1)
pixelsPerMM = pixelDistance / results.distance

if results.verbose:
    print("Pixel distance " + str(pixelDistance))
    print("Pixels per mm " + str(pixelsPerMM))

calibrationFile = open(results.output, "w")
calibrationFile.write(constants.PROPERTY_PIXELS_PER_MM + " " + str(pixelsPerMM))
calibrationFile.close()

sys.exit(0)


