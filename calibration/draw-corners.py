#
# D R A W C O R N E R S
# Draw corners on the image specified
#
# The image must include a checkerboard image
#

import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library

import argparse
import sys

parser = argparse.ArgumentParser("Draw corners on checkerboard image")

parser.add_argument('-i', '--input', action="store", required=True, help="Input Image")
parser.add_argument('-o', '--output', action="store", required=True, help="Output Image")
parser.add_argument('-x', '--xsquares', action="store", default=10, required=False, help="Number of X squares")
parser.add_argument('-y', '--ysquares', action="store", default=6, required=False, help="Number of Y squares")
arguments = parser.parse_args()

# Original code before modifications
# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Detect corners on a chessboard

filename = arguments.input

# Chessboard dimensions
number_of_squares_X = arguments.xsquares  # Number of chessboard squares along the x-axis
number_of_squares_Y = arguments.ysquares  # Number of chessboard squares along the y-axis
nX = number_of_squares_X - 1  # Number of interior corners along x-axis
nY = number_of_squares_Y - 1  # Number of interior corners along y-axis


def main():
    # Load an image
    image = cv2.imread(filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the corners on the chessboard
    success, corners = cv2.findChessboardCorners(gray, (nY, nX), None)

    # If the corners are found by the algorithm, draw them
    if success:
        # Draw the corners
        cv2.drawChessboardCorners(image, (nY, nX), corners, success)

        # Save the new image in the working directory
        cv2.imwrite(arguments.output, image)

        # Display the image
        cv2.imshow("Image", image)

        # Display the window until any key is pressed
        cv2.waitKey(0)

        # Close all windows
        cv2.destroyAllWindows()
    else:
        print("Unable to locate corners in image")

main()