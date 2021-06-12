#
# C A L I B R A T E
#
# This program will produce a calibration text file that can subsequently be used by the main program
# Here, the primary data item of interest is the pixels/mm value
#
# This can be performed live using a camera or from a static image.
# in either case, the target image should have two colored dots a specific distance apart
#
# Exit status 1 == failure
# Exit status 0 == OK
#


import cv2 as cv
import time
import argparse
import numpy as np
import sys
import atexit

from CameraFile import CameraPhysical, CameraFile
from ImageManipulation import ImageManipulation
import constants

parser = argparse.ArgumentParser("Calibrate camera & pixel dimensions")

parser.add_argument('-i', '--input', action="store", help="Calibration photo")
parser.add_argument('-o', '--output', action="store", required=True, help="Output for calibration data")
parser.add_argument('-d', '--distance', action="store", default=76.2, type=float, help="Distance in MM between two dots")
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-c", "--color", action="store", help="Color of the calibration dots", default="green")

results = parser.parse_args()

# Clean up by disconnecting from the camera. Bad things seem to happen if the program
# crashes without doing this.
def cleanup():
    if camera is not None:
        camera.disconnect()

if results.input is not None:
    image = cv.imread(results.input)
    if image is None:
        print("Unable to read calibration image " + results.input)
        sys.exit(1)
else:
    atexit.register(cleanup)
    camera = CameraPhysical("")

    if not camera.connect():
        print("Unable to connnect to camera")
        sys.exit(1)

    # Read the image from the camera
    image = camera.capture()

    if results.verbose:
        cv.imwrite("camera-image.jpg", image)

# The HSV values for the colors we expect
COLOR_RANGE = 20

if results.color == "green":
    green = np.uint8([[[0,255,0 ]]])
    hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
    lower = np.array([int(hsv_green[0,0,0]) - COLOR_RANGE,100,100])
    upper = np.array([int(hsv_green[0,0,0]) + COLOR_RANGE, 255,255])

manipulation = ImageManipulation(image)

hsv = manipulation.toHSV()

# define range of green color in HSV
# Not the best choice -- we need a color that is not likely to appear in the field

lower_green = np.array([60 - 20,100,100])
upper_green = np.array([60 + 20,255,255])

# Threshold the HSV image to get only blue colors
#mask = cv.inRange(hsv, lower_green, upper_green)
mask = cv.inRange(hsv, lower, upper)
# Bitwise-AND mask and original image
res = cv.bitwise_and(image,image, mask= mask)

if results.verbose:
    cv.imwrite("mask.jpg", mask)
    cv.imwrite("result.jpg", res)

manipulation = ImageManipulation(res)

# Find the two blobs in the image -- those should be the two colored dots

(contours, hierarchy, blobs, largestName) = manipulation.findBlobs(2)

# Not really required, just for debugging

manipulation.drawBoxes(blobs)

if results.verbose:
    cv.imwrite("calibration-results.jpg", manipulation.image)

# We should have only 2 blobs detected in the calibration
if len(blobs) != 2:
    print("Incorrect number of blobs detect in image for calibration (" + str(len(blobs)) + ")")
    sys.exit(1)

blob0 = blobs["blob0"]
blob1 = blobs["blob1"]

# We should already have the centers of the two blobs
(x1, y1) = blob0.get(constants.NAME_CENTER)
(x2, y2) = blob1.get(constants.NAME_CENTER)

# Take the absolute value, as the blobs can be in any order
pixelDistance = np.abs(x2 - x1)
pixelsPerMM = pixelDistance / results.distance

if results.verbose:
    print("Pixel distance " + str(pixelDistance))
    print("Pixels per mm " + str(pixelsPerMM))

# Write out the calculated values to the configuration file

calibrationFile = open(results.output, "w")
calibrationFile.write(constants.PROPERTY_PIXELS_PER_MM + " " + str(pixelsPerMM))
calibrationFile.close()

sys.exit(0)


