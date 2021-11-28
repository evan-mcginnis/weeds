#
# R N R  5 2 2
# F I N A L  P R O J E C T
#


import argparse
import sys
from typing import Callable

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import logging
import logging.config
import yaml
import os
import constants

from ImageManipulation import ImageManipulation


parser = argparse.ArgumentParser("Process orthomosaic")

parser.add_argument('-c', '--contours', action="store_true", default=False, help="Draw contours")
parser.add_argument('-i', '--input', action="store", required=True, help="Image to process")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-m", '--mm', action="store", required=True, type=float, help="mm per pixel")
parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)

arguments = parser.parse_args()

def plot3D(index, title):
    yLen,xLen = index.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,10))
    axes = fig.gca(projection ='3d')
    plt.title(title)
    axes.scatter(x, y, index, c=index, cmap='BrBG', s=0.25)
    plt.show()
    cv.waitKey()

def displayStats(blobs: {}):
    totalArea = 0

    for blobName, blobAttributes in blobs.items():
        totalArea += blobAttributes[constants.NAME_AREA]

    print("Total blobs: {}".format(len(blobs)))
    print("Total area: {} pixels * {} mm^2 per pixel = {} mm^2".format(totalArea, arguments.mm, arguments.mm * totalArea))
# if arguments.logistic:
#     try:
#         classifier = LogisticRegressionClassifier()
#         classifier.load(arguments.data, stratify=False)
#         classifier.createModel(arguments.score)
#         #classifier.scatterPlotDataset()
#     except FileNotFoundError:
#         print("Regression data file %s not found\n" % arguments.regression)
#         sys.exit(1)
# elif arguments.knn:
#    classifier = KNNClassifier()
#    classifier.load(arguments.data, stratify=False)
#    classifier.createModel(arguments.score)
# elif arguments.tree:
#    classifier = DecisionTree()
#    classifier.load(arguments.data, stratify=False)
#    classifier.createModel(arguments.score)
#    classifier.visualize()
# elif arguments.forest:
#    classifier = RandomForest()
#    classifier.load(arguments.data, stratify=True)
#    classifier.createModel(arguments.score)
#    classifier.visualize()
# elif arguments.gradient:
#    classifier = GradientBoosting()
#    classifier.load(arguments.data, stratify=False)
#    classifier.createModel(arguments.score)
#    classifier.visualize()
# elif arguments.support:
#     classifier = SVM()
#     classifier.load(arguments.data, stratify=False)
#     classifier.createModel(arguments.score)
#     classifier.visualize()
# else:
#     # TODO: This should be HeuristicClassifier
#     classifier = Classifier()
#     print("Classify using heuristics\n")

log = logging.getLogger(__name__)

# Not really needed -- just so we can keep the constructor the same.
imageNumber = 0
try:
    rawImage = cv.imread(arguments.input,cv.IMREAD_COLOR)

    plt.imshow(rawImage)
    plt.show()

    manipulated = ImageManipulation(rawImage, imageNumber)
    manipulated.mmPerPixel = arguments.mm
    #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

    #if arguments.plot:
    #    plot3D(index, arguments.algorithm)

    # TODO: Conversion to HSV should be done automatically
    manipulated.toYCBCR()
    manipulated.toHSV()
    manipulated.toHSI()
    manipulated.toYIQ()

    # Find the plants in the image
    contours, hierarchy, blobs, largest = manipulated.findBlobs(arguments.minarea)

    displayStats(blobs)

    # The test should probably be if we did not find any blobs
    # if largest == "unknown":
    #     logger.logImage("error", manipulated.image)
    #     sys.exit(1)
    #
    # # Set the classifier blob set to be the set just identified
    # classifier.blobs = blobs
    #
    #
    # performance.start()
    # manipulated.computeShapeIndices()
    # performance.stopAndRecord(constants.PERF_SHAPES)
    #
    # performance.start()
    # manipulated.computeLengthWidthRatios()
    # performance.stopAndRecord(constants.PERF_LW_RATIO)
    #
    # # Classify items by where they are in image
    # # This only marks items that can't be fully seen (at edges) of image
    # classifier.classifyByPosition(size=manipulated.image.shape)



    # classifiedBlobs = classifier.blobs
    #
    #
    #
    # # Compute the mean of the hue across the plant
    # manipulated.extractImagesFrom(manipulated.hsi,0, constants.NAME_HUE, np.nanmean)
    # performance.stopAndRecord(constants.PERF_MEAN)
    # manipulated.extractImagesFrom(manipulated.hsv,1, constants.NAME_SATURATION, np.nanmean)
    #
    # # Discussion of YIQ can be found here
    # # Sabzi, Sajad, Yousef Abbaspour-Gilandeh, and Juan Ignacio Arribas. 2020.
    # # “An Automatic Visible-Range Video Weed Detection, Segmentation and Classification Prototype in Potato Field.”
    # # Heliyon 6 (5): e03685.
    # # The article refers to the I component as in-phase, but its orange-blue in the wikipedia description
    # # of YIQ.  Not sure which is correct.
    #
    # # Compute the standard deviation of the I portion of the YIQ color space
    # performance.start()
    # manipulated.extractImagesFrom(manipulated.yiq,1, constants.NAME_I_YIQ, np.nanstd)
    # performance.stopAndRecord(constants.PERF_STDDEV)
    #
    # # Compute the mean of the blue difference in the ycc color space
    # performance.start()
    # manipulated.extractImagesFrom(manipulated.ycbcr,1, constants.NAME_BLUE_DIFFERENCE, np.nanmean)
    # performance.stopAndRecord(constants.PERF_MEAN)
    #
    # #Use either heuristics or logistic regression
    # if arguments.logistic or arguments.knn or arguments.tree or arguments.forest or arguments.gradient:
    #     performance.start()
    #     classifier.classify()
    #     performance.stopAndRecord(constants.PERF_CLASSIFY)
    #     classifiedBlobs = classifier.blobs
    # else:
    #     performance.start()
    #     classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=arguments.minratio)
    #     performance.stopAndRecord(constants.PERF_CLASSIFY)

    # Draw boxes around the images we found with decorations for attributes selected
    #manipulated.drawBoundingBoxes(contours)
    #manipulated.drawBoxes(manipulated.name, classifiedBlobs, featuresToShow)

    #manipulated.drawCropline()
    #logger.logImage("crop-line", manipulated.croplineImage)
    if arguments.contours:
        manipulated.drawContours()

    manipulated.show("Final", manipulated.image)
    cv.imwrite("final.jpg", manipulated.image)


except IOError as e:
    print("There was a problem communicating with the camera")
    print(e)
    sys.exit(1)

# if arguments.histograms:
#     reporting.showHistogram("Areas", 20, constants.NAME_AREA)
#     reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)
