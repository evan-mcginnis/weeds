#
# W E E D S
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

from CameraFile import CameraFile, CameraPhysical
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SVM
from Odometer import Odometer, VirtualOdometer
from Performance import Performance
from Reporting import Reporting
from Treatment import Treatment
#from Selection import Selection

import constants

# Used in command line processing so we can accept thresholds that are tuples
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

# This is here so we can extract the supported algorithms

veg = VegetationIndex()

parser = argparse.ArgumentParser("Weed recognition system")

parser.add_argument('-i', '--input', action="store", required=True, help="Images directory")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms(),
                    default="ngrdi")
parser.add_argument("-c", "--contours", action="store_true", default=False, help="Show contours on images")
parser.add_argument("-d", "--decorations", action="store", type=str, default="all", help="Decorations on output images (all and none are shortcuts)")
parser.add_argument("-df", "--data", action="store", help="Name of the data in CSV for use in logistic regression or KNN")
parser.add_argument("-hg", "--histograms", action="store_true", default=False, help="Show histograms")
parser.add_argument("-he", "--height", action="store_true", default=False, help="Consider height in scoring")
group = parser.add_mutually_exclusive_group()
group.add_argument("-k", "--knn", action="store_true", default=False, help="Predict using KNN. Requires data file to be specified")
group.add_argument("-l", "--logistic", action="store_true", default=False, help="Predict using logistic regression. Requires data file to be specified")
group.add_argument("-dt", "--tree", action="store_true", default=False, help="Predict using decision tree. Requires data file to be specified")
group.add_argument("-f", "--forest", action="store_true", default=False, help="Predict using random forest. Requires data file to be specified")
group.add_argument("-g", "--gradient", action="store_true", default=False, help="Predict using gradient boosting. Requires data file to be specified")
group.add_argument("-svm", "--support", action="store_true", default=False, help="Predict using support vector machine. Requires data file to be specified")
parser.add_argument("-im", "--image", action="store", default=200, type=int, help="Horizontal length of image")
parser.add_argument("-lg", "--logging", action="store", default="info-logging.yaml", help="Logging configuration file")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
parser.add_argument("-mr", "--minratio", action="store", default=5, type=int, help="Minimum size ratio for classifier")
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
parser.add_argument("-r", "--results", action="store", default="results.csv", help="Name of results file")
parser.add_argument("-s", "--stitch", action="store_true", help="Stitch adjacent images together")
parser.add_argument("-sc", "--score", action="store_true", help="Score the prediction method")
parser.add_argument("-se", "--selection", action="store", default="all-parameters.csv", help="Parameter selection file")
parser.add_argument("-sp", "--spray", action="store_true", help="Generate spray treatment grid")
parser.add_argument("-spe", "--speed", action="store", default=1, type=int, help="Speed in meters per second")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Generate debugging data and text")
parser.add_argument("-x", "--xtract", action="store_true", default=False, help="Extract each crop plant into images")


arguments = parser.parse_args()

# The list of decorations on the output.
# index
# classifier
# ratio
# center
# area
# distance
decorations = [item for item in arguments.decorations.split(',')]

if (arguments.logistic or arguments.knn or arguments.tree or arguments.forest) and arguments.data is None:
    print("Data file is not specified.")
    sys.exit(1)

#
# C A M E R A
#

def startupCamera():
    if arguments.input is not None:
        # Get the images from a directory
        camera = CameraFile(arguments.input)
    else:
        # Get the images from an actual camera
        camera = CameraPhysical("")
    if not camera.connect():
        print("Unable to connect to camera.")
        sys.exit(1)

    (w, h) = camera.getResolution()

    # Test the camera
    diagnosticResult, diagnosticText = camera.diagnostics()
    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)
    return camera

#
# O D O M E T E R
#

def startupOdometer(imageProcessor: Callable) -> Odometer:
    """
    Start up the odometry subsystem and run diagnostics
    :param imageProcessor: The image processor executed at each interval
    :return:
    """
    # For now, we have only a virtual odometer
    odometer = VirtualOdometer(arguments.speed, arguments.image, imageProcessor)
    if not odometer.connect():
        print("Unable to connect to odometer.")
        sys.exit(1)

    # Run diagnostics on the odometer before we begin.
    diagnosticResult, diagnosticText = odometer.diagnostics()

    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)

    return odometer

def startupPerformance() -> Performance:
    """
    Start up the performance subsystem.
    :return:
    """
    performance = Performance(arguments.performance)
    (performanceOK, performanceDiagnostics) = performance.initialize()
    if not performanceOK:
        print(performanceDiagnostics)
        sys.exit(1)
    return performance

#
# L O G G E R
#

def startupLogger(outputDirectory: str) -> Logger:
    """
    Initializes two logging systems: the image logger and python centric logging.
    :param outputDirectory: The output directory for the images
    :return: The image logger instance
    """

    # The command line argument contains the name of the YAML configuration file.

    # Confirm the YAML file exists
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    # Initialize logging
    with open(arguments.logging, "rt") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        #logger = logging.getLogger(__name__)

    logger = Logger()
    if not logger.connect(outputDirectory):
        print("Unable to connect to logging. ./output does not exist.")
        sys.exit(1)
    return logger

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


# Keep track of attributes in processing

reporting = Reporting(arguments.results)

(reportingOK, reportingReason) = reporting.initialize()
if not reportingOK:
    print(reportingReason)
    sys.exit(1)

# Used in stitching
previousImage = None
sequence = 0

# The factors considered in classification
#
# factors = [constants.NAME_RATIO,
#            constants.NAME_DISTANCE_NORMALIZED,
#            constants.NAME_SHAPE_INDEX]
#
# if arguments.height:
#     factors.append(constants.NAME_HEIGHT)

# Initialize logistic regression only if the user specified a data file

if arguments.logistic:
    try:
        classifier = LogisticRegressionClassifier()
        classifier.loadSelections(arguments.selection)
        classifier.load(arguments.data, stratify=False)
        classifier.createModel(arguments.score)
        #classifier.scatterPlotDataset()
    except FileNotFoundError:
        print("Regression data file %s not found\n" % arguments.regression)
        sys.exit(1)
elif arguments.knn:
   classifier = KNNClassifier()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=False)
   classifier.createModel(arguments.score)
elif arguments.tree:
   classifier = DecisionTree()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=False)
   classifier.createModel(arguments.score)
   classifier.visualize()
elif arguments.forest:
   classifier = RandomForest()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=True)
   classifier.createModel(arguments.score)
   classifier.visualize()
elif arguments.gradient:
   classifier = GradientBoosting()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=False)
   classifier.createModel(arguments.score)
   classifier.visualize()
elif arguments.support:
    classifier = SVM()
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    classifier.visualize()
else:
    # TODO: This should be HeuristicClassifier
    classifier = Classifier()
    print("Classify using heuristics\n")

# These are the attributes that will decorate objects in the images
if constants.NAME_ALL in arguments.decorations:
    featuresToShow = [constants.NAME_AREA,
                      constants.NAME_TYPE,
                      constants.NAME_LOCATION,
                      constants.NAME_CENTER,
                      constants.NAME_SHAPE_INDEX,
                      constants.NAME_RATIO,
                      constants.NAME_REASON,
                      constants.NAME_DISTANCE_NORMALIZED,
                      constants.NAME_NAME,
                      constants.NAME_HUE,
                      constants.NAME_TYPE,
                      constants.NAME_SOLIDITY,
                      constants.NAME_ROUNDNESS,
                      constants.NAME_CONVEXITY,
                      constants.NAME_ECCENTRICITY,
                      constants.NAME_I_YIQ]
elif constants.NAME_NONE in arguments.decorations:
    featuresToShow = []
else:
    featuresToShow = [arguments.decorations]



# The contours are a bit distracting
if arguments.contours:
    featuresToShow.append(constants.NAME_CONTOUR)

imageNumber = 0

def processImage() -> bool:
    global imageNumber
    global sequence
    global previousImage

    try:

        if arguments.verbose:
            print("Processing image " + str(imageNumber))
        log.info("Processing image " + str(imageNumber))
        performance.start()
        rawImage = camera.capture()
        performance.stopAndRecord(constants.PERF_ACQUIRE)

        #ImageManipulation.show("Source",image)
        veg.SetImage(rawImage)

        manipulated = ImageManipulation(rawImage, imageNumber)
        manipulated.mmPerPixel = mmPerPixel
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        performance.start()
        index = veg.Index(arguments.algorithm)
        performance.stopAndRecord(constants.PERF_INDEX)

        #ImageManipulation.show("index", index)
        #cv.imwrite("index.jpg", index)
        if arguments.plot:
            plot3D(index, arguments.algorithm)

        # Get the mask
        #mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)
        mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, 1, arguments.threshold)

        veg.applyMask()
        # This is the slow call
        #image = veg.GetMaskedImage()
        image = veg.GetImage()
        normalized = np.zeros_like(image)
        finalImage = cv.normalize(image,  normalized, 0, 255, cv.NORM_MINMAX)
        if arguments.mask:
            filledMask = mask.copy().astype(np.uint8)
            cv.floodFill(filledMask, None, (0,0),255)
            filledMaskInverted = cv.bitwise_not(filledMask)
            manipulated.toGreyscale()
            threshold, imageThresholded = cv.threshold(manipulated.greyscale, 0,255, cv.THRESH_BINARY_INV)
            finalMask = cv.bitwise_not(filledMaskInverted)
            logger.logImage("processed", finalImage)
            veg.ShowImage("Thresholded", imageThresholded)
            logger.logImage("inverted", filledMaskInverted)
            veg.ShowImage("Filled", filledMask)
            veg.ShowImage("Inverted", filledMaskInverted)
            veg.ShowImage("Final", finalMask)
            logger.logImage("final", finalMask)
            #plt.imshow(veg.imageMask, cmap='gray', vmin=0, vmax=1)
            plt.imshow(finalImage)
            plt.show()
            #logger.logImage("mask", veg.imageMask)
        #ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(finalImage, imageNumber)
        manipulated.mmPerPixel = mmPerPixel

        # TODO: Conversion to HSV should be done automatically
        performance.start()
        manipulated.toYCBCR()
        performance.stopAndRecord(constants.PERF_YCC)
        performance.start()
        manipulated.toHSV()
        performance.stopAndRecord(constants.PERF_HSV)
        performance.start()
        manipulated.toHSI()
        performance.stopAndRecord(constants.PERF_HSI)
        performance.start()
        manipulated.toYIQ()
        performance.stopAndRecord(constants.PERF_YIQ)

        # Find the plants in the image
        performance.start()
        contours, hierarchy, blobs, largest = manipulated.findBlobs(arguments.minarea)
        performance.stopAndRecord(constants.PERF_CONTOURS)

        # The test should probably be if we did not find any blobs
        if largest == "unknown":
            logger.logImage("error", manipulated.image)
            return

        performance.start()
        manipulated.identifyOverlappingVegetation()
        performance.stopAndRecord(constants.PERF_OVERLAP)

        # Set the classifier blob set to be the set just identified
        classifier.blobs = blobs


        performance.start()
        manipulated.computeShapeIndices()
        performance.stopAndRecord(constants.PERF_SHAPES)

        performance.start()
        manipulated.computeLengthWidthRatios()
        performance.stopAndRecord(constants.PERF_LW_RATIO)

        # New image analysis based on readings here:
        # http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf

        performance.start()
        manipulated.computeCompactness()
        performance.stopAndRecord(constants.PERF_COMPACTNESS)

        manipulated.computeElogation()
        manipulated.computeEccentricity()
        manipulated.computeRoundness()
        manipulated.computeConvexity()
        manipulated.computeSolidity()
        # End image analysis

        # Classify items by where they are in image
        # This only marks items that can't be fully seen (at edges) of image
        classifier.classifyByPosition(size=manipulated.image.shape)



        classifiedBlobs = classifier.blobs


        performance.start()
        manipulated.findAngles()
        manipulated.findCropLine()
        performance.stopAndRecord(constants.PERF_ANGLES)

        # Crop row processing
        manipulated.identifyCropRowCandidates()

        # Extract various features
        performance.start()

        # Compute the mean of the hue across the plant
        manipulated.extractImagesFrom(manipulated.hsi,0, constants.NAME_HUE, np.nanmean)
        performance.stopAndRecord(constants.PERF_MEAN)
        manipulated.extractImagesFrom(manipulated.hsv,1, constants.NAME_SATURATION, np.nanmean)

        # Discussion of YIQ can be found here
        # Sabzi, Sajad, Yousef Abbaspour-Gilandeh, and Juan Ignacio Arribas. 2020.
        # “An Automatic Visible-Range Video Weed Detection, Segmentation and Classification Prototype in Potato Field.”
        # Heliyon 6 (5): e03685.
        # The article refers to the I component as in-phase, but its orange-blue in the wikipedia description
        # of YIQ.  Not sure which is correct.

        # Compute the standard deviation of the I portion of the YIQ color space
        performance.start()
        manipulated.extractImagesFrom(manipulated.yiq,1, constants.NAME_I_YIQ, np.nanstd)
        performance.stopAndRecord(constants.PERF_STDDEV)

        # Compute the mean of the blue difference in the ycc color space
        performance.start()
        manipulated.extractImagesFrom(manipulated.ycbcr,1, constants.NAME_BLUE_DIFFERENCE, np.nanmean)
        performance.stopAndRecord(constants.PERF_MEAN)

        #Use either heuristics or logistic regression
        if arguments.logistic or arguments.knn or arguments.tree or arguments.forest or arguments.gradient:
            performance.start()
            classifier.classify()
            performance.stopAndRecord(constants.PERF_CLASSIFY)
            classifiedBlobs = classifier.blobs
        else:
            performance.start()
            classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=arguments.minratio)
            performance.stopAndRecord(constants.PERF_CLASSIFY)

        # Draw boxes around the images we found with decorations for attributes selected
        #manipulated.drawBoundingBoxes(contours)
        manipulated.drawBoxes(manipulated.name, classifiedBlobs, featuresToShow)

        #logger.logImage("cropline", manipulated.croplineImage)
        # This is using the hough transform which we abandoned as a technique
        #manipulated.detectLines()
        #TODO: Draw crop line as part of image decoration
        manipulated.drawCropline()
        #logger.logImage("crop-line", manipulated.croplineImage)
        if arguments.contours:
            manipulated.drawContours()


        # Just a test of stitching. This needs some more thought
        # we can't stitch things where there is nothing in common between the two images
        # even if there is overlap. It may just be black background after the segmentation.
        # One possibility here is to put something in the known overlap area that can them be used
        # to align the images.
        # The alternative is to use the original images and use the soil as the element that is common between
        # the two.  The worry here is computational efficiency

        if arguments.stitch:
            if previousImage is not None:
                manipulated.stitchTo(previousImage)
            else:
                previousImage = image

        # Write out the processed image
        #cv.imwrite("processed.jpg", manipulated.image)
        logger.logImage("processed", manipulated.image)
        logger.logImage("original", manipulated.original)
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # Write out the crop images so we can use them later
        if arguments.xtract:
            manipulated.extractImages(classifiedAs=constants.TYPE_DESIRED)
            for blobName, blobAttributes in manipulated.blobs.items():
                if blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED:
                    logger.logImage("crop", blobAttributes[constants.NAME_IMAGE])

        reporting.addBlobs(sequence, blobs)
        sequence = sequence + 1

        if arguments.spray:
            if arguments.verbose:
                print("Forming treatment")
            performance.start()
            treatment = Treatment(manipulated.original, manipulated.binary)
            treatment.overlayTreatmentLanes()
            treatment.generatePlan(classifiedBlobs)
            #treatment.drawTreatmentLanes(classifiedBlobs)
            performance.stopAndRecord(constants.PERF_TREATMENT)
            logger.logImage("treatment", treatment.image)

        imageNumber = imageNumber + 1

    except IOError as e:
        print("There was a problem communicating with the camera")
        print(e)
        sys.exit(1)
    except EOFError:
        print("End of input")
        return False

    if arguments.histograms:
        reporting.showHistogram("Areas", 20, constants.NAME_AREA)
        reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)

    return True

#
# Start up various subsystems
#

camera = startupCamera()
odometer = startupOdometer(processImage)
logger = startupLogger(arguments.output)
log = logging.getLogger(__name__)
performance = startupPerformance()

mmPerPixel = camera.getMMPerPixel()

# The odometer drives everything
odometer.start()

performance.cleanup()

# Not quite right here to get the list of all blobs from the reporting module
#classifier.train(reporting.blobs)

result, reason = reporting.writeSummary()

if not result:
    print(reason)
    sys.exit(1)
else:
    sys.exit(0)