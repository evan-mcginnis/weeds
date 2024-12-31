#
# P L O T - T H R E S H O L D
#
# Calculate and display the histogram and threshold using the triangle method

import sys

import argparse
import cv2 as cv
import numpy as np

#import plotly.express as plot
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Plot threshold of image")

parser.add_argument("-i", '--image', action="store", required=True, help="Target image")
parser.add_argument("-b", '--bins', action="store", required=False, default=256, help="Target image")

arguments = parser.parse_args()

# Read the image as greyscale
img = cv.imread(arguments.image, 0)

# Compute the histogram
counts, bins = np.histogram(img, arguments.bins)
plt.stairs(counts, bins, fill=True)
plt.title("Intensity Values of Image")
plt.ylabel("Count")
plt.xlabel("Pixel Value")
plt.tight_layout()
plt.show()

# hist, binEdges = np.histogram(img, arguments.bins)
# #plt.hist(img, bins=20)
# plt.show()
# fig = plot.histogram(hist, x="Points")
# fig.show()

img = cv.imread(arguments.image)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (7, 7), 0)

histogram_gray, bin_edges_gray = np.histogram(gray, bins=256, range=(0, 255))
histogram_blurred, bin_edges_blurred = np.histogram(blurred, bins=256, range=(0, 255))

#fig, ax = plt.subplots(1, 2, figsize=(12, 4))

#ax[0].plot(bin_edges_gray[0:-1], histogram_gray)
#ax[1].plot(bin_edges_blurred[0:-1], histogram_blurred)
#fig.show()

plt.plot(bin_edges_gray[0:-1], histogram_gray)
plt.ylabel("Count")
plt.xlabel("Pixel Value")
plt.show()


# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
#cv.imshow("image", img)

cv.waitKey(0)

sys.exit(0)

