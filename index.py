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

#from CameraFile import CameraFile, CameraPhysical
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SVM
from Odometer import Odometer, VirtualOdometer
from Performance import Performance
from Reporting import Reporting
from Treatment import Treatment
import constants

ALG_ALL = "all"
# Used in command line processing so we can accept thresholds that are tuples
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

# This is here so we can extract the supported algorithms

veg = VegetationIndex()

parser = argparse.ArgumentParser("Apply Index to given image")

parser.add_argument('-i', '--input', action="store", required=True, help="Image to process")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms().append(ALG_ALL),
                    default="ngrdi")
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-d", "--direction", action="store", default=1, type=int, help="Direction 1 or -1")
parser.add_argument("-g", "--gsd", action="store", default=2.74, type=float, help="Ground Sampling Distance")
# parser.add_argument("-c", "--contours", action="store_true", default=False, help="Show contours on images")
# parser.add_argument("-d", "--decorations", action="store", type=str, default="all", help="Decorations on output images (all and none are shortcuts)")
# parser.add_argument("-df", "--data", action="store", help="Name of the data in CSV for use in logistic regression or KNN")
# parser.add_argument("-hg", "--histograms", action="store_true", default=False, help="Show histograms")
# parser.add_argument("-he", "--height", action="store_true", default=False, help="Consider height in scoring")
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-k", "--knn", action="store_true", default=False, help="Predict using KNN. Requires data file to be specified")
# group.add_argument("-l", "--logistic", action="store_true", default=False, help="Predict using logistic regression. Requires data file to be specified")
# group.add_argument("-dt", "--tree", action="store_true", default=False, help="Predict using decision tree. Requires data file to be specified")
# group.add_argument("-f", "--forest", action="store_true", default=False, help="Predict using random forest. Requires data file to be specified")
# group.add_argument("-g", "--gradient", action="store_true", default=False, help="Predict using gradient boosting. Requires data file to be specified")
# group.add_argument("-svm", "--support", action="store_true", default=False, help="Predict using support vector machine. Requires data file to be specified")
# parser.add_argument("-im", "--image", action="store", default=200, type=int, help="Horizontal length of image")
# parser.add_argument("-lg", "--logging", action="store", default="info-logging.yaml", help="Logging configuration file")
# parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
# parser.add_argument("-mr", "--minratio", action="store", default=5, type=int, help="Minimum size ratio for classifier")
# parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
# parser.add_argument("-r", "--results", action="store", default="results.csv", help="Name of results file")
# parser.add_argument("-s", "--stitch", action="store_true", help="Stitch adjacent images together")
# parser.add_argument("-sc", "--score", action="store_true", help="Score the prediction method")
# parser.add_argument("-sp", "--spray", action="store_true", help="Generate spray treatment grid")
# parser.add_argument("-spe", "--speed", action="store", default=1, type=int, help="Speed in meters per second")
# parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Generate debugging data and text")
# parser.add_argument("-x", "--xtract", action="store_true", default=False, help="Extract each crop plant into images")


arguments = parser.parse_args()

ALG_NDI="ndi"
ALG_TGI="tgi"
ALG_EXG="exg"
ALG_EXR="exr"
ALG_EXGEXR="exgexr"
ALG_CIVE="cive"
ALG_NGRDI="ngrdi"
ALG_VEG="veg"
ALG_COM1="com1"
ALG_MEXG="mexg"
ALG_COM2="com2"
ALG_RGD="rgd"

thresholds = {ALG_NDI: (130,0),
              ALG_EXG: (18, 0),
              ALG_EXR: (35, 5),
              ALG_CIVE: (30,0),
              ALG_EXGEXR: (0,0),
              ALG_NGRDI: (10,0),
              ALG_VEG: (1.25,0),
              ALG_COM1: (-100,0),
              ALG_MEXG: (25,0),
              ALG_COM2: (15,0),
              ALG_TGI: (300,-15)} # Revisit

indices = {ALG_NDI: {"short": ALG_NDI, "create": veg.NDI, "negate": True, "threshold": thresholds[ALG_NDI], "direction": 1},
           ALG_EXG: {"short": ALG_EXG, "create": veg.ExG, "negate": True, "threshold": thresholds[ALG_EXG], "direction": 1},
           ALG_EXR: {"short": ALG_EXR, "create": veg.ExR, "negate": False, "threshold": thresholds[ALG_EXR], "direction": 1},
           ALG_CIVE: {"short": ALG_CIVE, "create": veg.CIVE, "negate": False, "threshold": thresholds[ALG_CIVE], "direction": 1},
           ALG_EXGEXR: {"short": ALG_EXGEXR, "create": veg.ExGR, "negate": True, "threshold": thresholds[ALG_EXGEXR], "direction": 1},
           ALG_NGRDI: {"short": ALG_NGRDI, "create": veg.NGRDI, "negate": True, "threshold": thresholds[ALG_NGRDI], "direction": 1},
           ALG_VEG: {"short": ALG_VEG, "create": veg.VEG, "negate": True, "threshold": thresholds[ALG_VEG], "direction": 1},
           ALG_COM1: {"short": ALG_COM1, "create": veg.COM1, "negate": False, "threshold": None, "direction": 1} ,
           ALG_MEXG: {"short": ALG_MEXG, "create": veg.MExG, "negate": True, "threshold": thresholds[ALG_MEXG], "direction": 1},
           ALG_COM2: {"short": ALG_COM2, "create": veg.COM2, "negate": False, "threshold": thresholds[ALG_COM2], "direction": 1},
           ALG_TGI: {"short": ALG_TGI, "create": veg.TGI, "negate": True, "threshold": thresholds[ALG_TGI], "direction": 1}}

_algorithms = [ALG_NDI,
              ALG_EXG,
              ALG_EXR,
              ALG_CIVE,
              ALG_EXGEXR,
              ALG_NGRDI,
              ALG_VEG,
              ALG_COM1,
              ALG_MEXG,
              ALG_COM2,
              ALG_TGI]

#_algorithms = [ALG_TGI]

#
# self.computations = {self.ALG_NDI     : {"description": "Normalized Difference", "create": self.NDI, "threshold": thresholds["NDI"]},
#                      self.ALG_EXG     : {"description": "Excess Green", "create": self.ExG, "threshold": thresholds["EXG"]},
#                      self.ALG_EXR     : {"description": "Excess Red", "create": self.ExR, "threshold": thresholds["EXR"]},
#                      self.ALG_CIVE    : {"description": "Color Index of Vegetation Extraction", "create": self.CIVE, "threshold": thresholds["CIVE"]},
#                      self.ALG_EXGEXR  : {"description": "Excess Green - Excess Red", "create": self.ExGR, "threshold": thresholds["EXGEXR"]},
#                      self.ALG_NGRDI   : {"description": "Normalized Green Red Difference", "create": self.NGRDI, "threshold": thresholds["NGRDI"]},
#                      self.ALG_VEG     : {"description": "Vegetative Index", "create": self.VEG, "threshold": thresholds["VEG"]},
#                      self.ALG_COM1    : {"description": "Combined Index 1", "create": self.COM1, "threshold": thresholds["COM1"]} ,
#                      self.ALG_MEXG    : {"description": "Modified Excess Green", "create": self.MExG, "threshold": thresholds["MEXG"]},
#                      self.ALG_COM2    : {"description": "Combined Index 2", "create": self.COM2, "threshold": thresholds["COM2"]},
#                      self.ALG_TGI     : {"description": "TGI", "create": self.TGI, "threshold": thresholds["TGI"]},
#                      self.ALG_RGD     : {"description": "Red Green Dots", "create": self.redGreenDots, "threshold": 0}}


# # The list of decorations on the output.
# # index
# # classifier
# # ratio
# # center
# # area
# # distance
# decorations = [item for item in arguments.decorations.split(',')]
#
# if (arguments.logistic or arguments.knn or arguments.tree or arguments.forest) and arguments.data is None:
#     print("Data file is not specified.")
#     sys.exit(1)

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
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Index Value')
    plt.show()
    cv.waitKey()

def plotTransect(image, title):
    yLen, xLen = image.shape
    x = np.arange(0, yLen, 1)
    #plt.style.use('ggplot')
    plt.figure(figsize=(20,10))
    plt.title("Transect across river")
    plt.xlabel('Distance')
    plt.ylabel('Pixel Value')
    plt.plot(x, image[:,0], color='red', label="Red")
    plt.plot(x, image[:,1], color='green', label="Green")
    plt.plot(x, image[:,2], color='blue', label="Blue")
    plt.legend()
    plt.show()
    cv.waitKey()



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
performance = startupPerformance()
imageNumber = 0
mmPerPixel = 0
try:
    performance.start()
    #rawImage = camera.capture()
    rawImage = cv.imread(arguments.input,cv.IMREAD_COLOR)

    #plt.imshow(rawImage)
    #plt.show()

    if arguments.algorithm == ALG_ALL:
        veg.SetImage(rawImage)
        results = [0] * (len(thresholds) + 1)
        #algorithms = [0] * (len(thresholds) + 1)
        algorithms = []
        i = 0
        # Debug line
        #for algorithm in _algorithms:
        for algorithm in veg.GetSupportedAlgorithms():
            # This is a workaround for what is probably a bug (or a something I don't understand)
            # A call to GetSupportedAlgorithms() returns the "all" appended to it above
            # This doesn't seem quite right...
            if algorithm != ALG_ALL and algorithm != ALG_RGD:
                print("Algorithm: {}".format(algorithm))
                veg = VegetationIndex()
                veg.SetImage(rawImage)
                index = veg.Index(algorithm)
                mask, threshold = veg.MaskFromIndex(index,
                                                    indices.get(algorithm).get("negate"),
                                                    indices.get(algorithm).get("direction"),
                                                    indices.get(algorithm).get("threshold"))
                #mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, 1, arguments.threshold)
                veg.applyMask()
                image = veg.GetImage()
                # Uncomment this line to show
                #ImageManipulation.show(algorithm, image)
                #ImageManipulation.save(image, algorithm + ".jpg")
                cv.imwrite(arguments.output + "/" + "processed-with-" + algorithm + ".jpg", image)
                veg.ShowStats(image)
                # The results hold the area calculation
                results[i] = veg.GetImageStats(image) * arguments.gsd
                i += 1
                algorithms.append(algorithm)

        xs = np.arange(0, len(results), 1)
        #x_pos = [i for i, _ in enumerate(algorithms)]
        plt.figure(figsize=(10,5))
        plt.title("Vegetated Area")
        plt.xlabel('Algorithm')
        plt.ylabel('Vegetated Area (CM)')
        plt.bar(algorithms, results)
        plt.savefig(arguments.output + "/" + "bar-counts.jpg")
        plt.show()

        sys.exit(0)


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



    # Get the mask
    #mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)
    mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, arguments.direction, arguments.threshold)

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

    if arguments.plot:
        plot3D(index[8000:8250,12000:12250], arguments.algorithm)
        plotTransect(image[9690,6170:8440], 'Transect')

    ImageManipulation.show("Masked", image)
    cv.imwrite(arguments.output + ".jpg", image)
    veg.ShowStats(image)

    sys.exit(0)

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
        sys.exit(1)

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

if arguments.histograms:
    reporting.showHistogram("Areas", 20, constants.NAME_AREA)
    reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)
