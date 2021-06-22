#
# W E E D S
#

import argparse
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from CameraFile import CameraFile, CameraPhysical
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier
from Odometer import VirtualOdometer
from Performance import Performance
from Reporting import Reporting
import constants

# Used in command line processing so we can accept thresholds that are tuples
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

# This is here so we can extract the supported algorithms

veg = VegetationIndex()

parser = argparse.ArgumentParser("Weed recognition system")

parser.add_argument('-i', '--input', action="store", help="Images directory")
parser.add_argument('-o', '--output', action="store", help="Output directory for processed images")
parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms(),
                    default="ngrdi")
parser.add_argument("-d", "--decorations", action="store", type=str, default="all", help="Decorations on output images")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")
parser.add_argument("-s", "--stitch", action="store_true", help="Stitch adjacent images together")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Generate debugging data and text")
parser.add_argument("-c", "--contours", action="store_true", default=False, help="Show contours on images")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-hg", "--histograms", action="store_true", default=False, help="Show histograms")
parser.add_argument("-r", "--results", action="store", default="results.csv", help="Name of results file")
parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
parser.add_argument("-mr", "--minratio", action="store", default=5, type=int, help="Minimum size ratio for classifier")
parser.add_argument("-l", "--lettuce", action="store_true", default=False, help="Isolate each crop plant into images")
parser.add_argument("-rf", "--regression", action="store", help="Name of the logistic regression data in CSV")

results = parser.parse_args()

# The list of decorations on the output.
# index
# classifier
# ratio
# center
# area
# distance
decorations = [item for item in results.decorations.split(',')]

#
# C A M E R A
#

def startupCamera():
    if results.input != None:
        # Get the images from a directory
        camera = CameraFile(results.input)
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

def startupOdometer():
    odometer = VirtualOdometer("")
    if not odometer.connect():
        print("Unable to connect to odometer")
        sys.exit(1)

    diagnosticResult, diagnosticText = odometer.diagnostics()

    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)

    return odometer

def startupPerformance() -> Performance:
    performance = Performance(results.performance)
    return performance

#
# L O G G E R
#

def startupLogger(outputDirectory: str) -> Logger:
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
#
# Start up various subsystems
#

camera = startupCamera()
odometer = startupOdometer()
logger = startupLogger(results.output)
performance = startupPerformance()

mmPerPixel = camera.getMMPerPixel()

# Keep track of everything seen in processing

reporting = Reporting()

# Used in stitching
previousImage = None
sequence = 0

# Initialize logistic regression
classifier = Classifier()

# Initialize logistic regression only if the user specified a data file

if results.regression is not None:
    try:
        classifier.loadLogisticRequestion(results.regression)
    except FileNotFoundError:
        print("Regression data file %s not found\n" % results.regression)
        sys.exit(1)
else:
    print("Classify using heuristics\n")

# These are the attributes that will decorate objects in the images
if constants.NAME_ALL in results.decorations:
    featuresToShow = [constants.NAME_AREA,\
                      constants.NAME_TYPE,
                      constants.NAME_LOCATION,
                      constants.NAME_CENTER,
                      constants.NAME_SHAPE_INDEX,
                      constants.NAME_RATIO,
                      constants.NAME_REASON,
                      constants.NAME_TYPE]
else:
    featuresToShow = results.decorations

# The contours are a bit distracting
if results.contours:
    featuresToShow.append(constants.NAME_CONTOUR)

#
# G R A N D  L O O P
#
try:
    # Loop and process images until requested to stop
    # TODO: Accept signal to stop processing
    while True:
        performance.start()
        rawImage = camera.capture()
        performance.stopAndRecord("aquire")

        #ImageManipulation.show("Source",image)
        veg.SetImage(rawImage)

        manipulated = ImageManipulation(rawImage)
        manipulated.mmPerPixel = mmPerPixel
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        performance.start()
        index = veg.Index(results.algorithm)
        performance.stopAndRecord("index")

        #ImageManipulation.show("index", index)
        #cv.imwrite("index.jpg", index)
        if results.plot:
            plot3D(index, results.algorithm)

        # Get the mask
        #mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)
        mask, threshold = veg.MaskFromIndex(index, not results.nonegate, 1, results.threshold)

        veg.applyMask()
        # This is the slow call
        #image = veg.GetMaskedImage()
        image = veg.GetImage()
        normalized = np.zeros_like(image)
        finalImage = cv.normalize(image,  normalized, 0, 255, cv.NORM_MINMAX)
        if results.mask:
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
            break
        #ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(finalImage)
        manipulated.mmPerPixel = mmPerPixel

        # Find the plants in the image
        performance.start()
        contours, hierarchy, blobs, largest = manipulated.findBlobs(results.minarea)
        performance.stopAndRecord("contours")

        # The test should probably be if we did not find any blobs
        if largest == "unknown":
            logger.logImage("error", manipulated.image)
            continue

        performance.start()
        manipulated.identifyOverlappingVegetation()
        performance.stopAndRecord("overlap")

        # Set the classifier blob set to be the set just identified
        classifier.blobs = blobs


        performance.start()
        manipulated.computeShapeIndices()
        performance.stopAndRecord("shapes")

        performance.start()
        manipulated.computeLengthWidthRatios()
        performance.stopAndRecord("LW Ratio")

        # Classify items by where they are in image
        classifier.classifyByPosition(size=manipulated.image.shape)



        classifiedBlobs = classifier.blobs


        performance.start()
        manipulated.findAngles()
        performance.stopAndRecord("angles")

        # Crop row processing
        manipulated.identifyCropRowCandidates()

        #Use either heuristics or logistic regression
        if results.regression is not None:
            performance.start()
            classifier.classifyByLogisticRegression()
            performance.stopAndRecord("regression")
            classifiedBlobs = classifier.blobs
        else:
            performance.start()
            classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=results.minratio)
            performance.stopAndRecord("classify")

        # Draw boxes around the images we found
        #manipulated.drawBoundingBoxes(contours)
        manipulated.drawBoxes(classifiedBlobs, featuresToShow)

        #logger.logImage("cropline", manipulated.croplineImage)
        # This is using the hough transform which we abandoned as a technique
        #manipulated.detectLines()
        manipulated.drawCropline()
        #logger.logImage("crop-line", manipulated.croplineImage)
        if results.contours:
            manipulated.drawContours()


        # Just a test of stitching. This needs some more thought
        # we can't stitch things where there is nothing in common between the two images
        # even if there is overlap. It may just be black background after the segmentation.
        # One possibility here is to put something in the known overlap area that can them be used
        # to align the images.
        # The alternative is to use the original images and use the soil as the element that is common between
        # the two.  The worry here is computational efficiency

        if results.stitch:
            if previousImage is not None:
                manipulated.stitchTo(previousImage)
            else:
                previousImage = image

        # Write out the processed image
        #cv.imwrite("processed.jpg", manipulated.image)
        logger.logImage("processed", manipulated.image)
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # Write out the crop images so we can use them later
        if results.lettuce:
            manipulated.extractImages(classifiedAs=constants.TYPE_DESIRED)
            for blobName, blobAttributes in manipulated.blobs.items():
                if blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED:
                    logger.logImage("crop", blobAttributes[constants.NAME_IMAGE])

        reporting.addBlobs(sequence, blobs)
        sequence = sequence + 1


except IOError:
    print("There was a problem communicating with the camera")
    sys.exit(1)
except EOFError:
    print("End of input")
    #sys.exit(0)

if results.histograms:
    reporting.showHistogram("Areas", 20, constants.NAME_AREA)
    reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)

# Not quite right here to get the list of all blobs from the reporting module
#classifier.train(reporting.blobs)

reporting.writeSummary(results.results)