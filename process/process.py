#
# P R O C E S S
#
# Process an image to identify weeds
#
from typing import Callable
import sys
import os

import argparse
import logging.config
import numpy as np
import scipy.ndimage
import cv2 as cv

import constants
from VegetationIndex import VegetationIndex
from CameraFile import CameraFile
from Performance import Performance
from Logger import Logger

from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SuppportVectorMachineClassifier
from OptionsFile import OptionsFile
from Reporting import Reporting
from ProcessedImage import ProcessedImage, Images
from Diagnostics import Diagnostics
from CameraFile import CameraFile

try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    supportsPlotting = True
except ImportError:
    print("Unable to import plotting libraries.")
    supportsPlotting = False

class Processor:
    def __init__(self, arguments):
        self._original = None
        self._processed = None
        self._arguments = arguments

        self._reporting = Reporting(self._arguments.results)
        self._performance = Performance(arguments.performance)
        self._log = None
        self._logger = None
        self._processor = self._processImage
        self._camera = None
        self._veg = VegetationIndex()
        self._imageNumber = 0

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
                self._classifier = LogisticRegressionClassifier()
                self._classifier.loadSelections(arguments.selection)
                self._classifier.load(arguments.data, stratify=False)
                self._classifier.createModel(arguments.score)
                # classifier.scatterPlotDataset()
            except FileNotFoundError:
                print("Regression data file %s not found\n" % arguments.regression)
                sys.exit(0)
        elif arguments.knn:
            self._classifier = KNNClassifier()
            # Load selected parameters
            self._classifier.loadSelections(arguments.selection)
            self._classifier.load(arguments.data, stratify=False)
            self._classifier.createModel(arguments.score)
        elif arguments.tree:
            self._classifier = DecisionTree()
            # Load selected parameters
            self._classifier.loadSelections(arguments.selection)
            self._classifier.load(arguments.data, stratify=False)
            self._classifier.createModel(arguments.score)
            self._classifier.visualize()
        elif arguments.forest:
            self._classifier = RandomForest()
            # Load selected parameters
            self._classifier.loadSelections(arguments.selection)
            self._classifier.load(arguments.data, stratify=True)
            self._classifier.createModel(arguments.score)
            self._classifier.visualize()
        elif arguments.gradient:
            self._classifier = GradientBoosting()
            # Load selected parameters
            self._classifier.loadSelections(arguments.selection)
            self._classifier.load(arguments.data, stratify=False)
            self._classifier.createModel(arguments.score)
            self._classifier.visualize()
        elif arguments.support:
            self._classifier = SuppportVectorMachineClassifier()
            # Load selected parameters
            self._classifier.loadSelections(arguments.selection)
            self._classifier.load(arguments.data, stratify=False)
            self._classifier.createModel(arguments.score)
            self._classifier.visualize()
        else:
            # TODO: This should be HeuristicClassifier
            self._classifier = Classifier()
            print("Classify using heuristics\n")

        # These are the attributes that will decorate objects in the images
        if constants.NAME_ALL in arguments.decorations:
            self._featuresToShow = [constants.NAME_AREA,
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
                                    constants.NAME_I_YIQ,
                                    constants.NAME_DIST_TO_LEADING_EDGE]
        elif constants.NAME_NONE in arguments.decorations:
            self._featuresToShow = []
        else:
            self._featuresToShow = [arguments.decorations]

        # The contours are a bit distracting
        if arguments.contours:
            self._featuresToShow.append(constants.NAME_CONTOUR)

    def initialize(self) -> bool:
        (reportingOK, reportingReason) = self._reporting.initialize()
        self._startupLogger(self._arguments.output)
        self._startupPerformance()

        # Connect to camera
        if self._arguments.input is not None:
            self._camera = CameraFile(directory=arguments.input, TYPE=constants.ImageType.RGB.name)
            self._camera.connect()

        return reportingOK and (self._log is not None)
        # if not reportingOK:
        #     print(reportingReason)
        #     sys.exit(0)

    @property
    def original(self) -> np.ndarray:
        return self._original

    @original.setter
    def original(self, image: np.ndarray):
        self._original = image

    @property
    def processed(self) -> np.ndarray:
        return self._processed

    def _segment(self, segmenter: Callable):
        pass

    def _startupPerformance(self):
        (performanceOK, performanceDiagnostics) = self._performance.initialize()
        if not performanceOK:
            print(performanceDiagnostics)
            sys.exit(1)

    def _startupLogger(self, outputDirectory: str) -> ():
        """
        Initializes two logging systems: the image logger and python centric logging.
        :param outputDirectory: The output directory for the images
        :return: The image logger instance
        """

        # The command line argument contains the name of the YAML configuration file.

        # Confirm the INI exists
        if not os.path.isfile(arguments.logging):
            print("Unable to access logging configuration file {}".format(arguments.logging))
            sys.exit(1)

        # Initialize logging
        logging.config.fileConfig(arguments.logging)
        self._log = logging.getLogger("processor")

        self._logger = Logger()
        if not self._logger.connect(outputDirectory):
            print("Unable to connect to logging. ./output does not exist.")
            sys.exit(1)
        return self._logger, self._log

    def _processImage(self) -> constants.ProcessResult:
        self._performance.start()

        self._imageNumber += 1
        # Attempt to capture the image.
        try:
            processed = self._camera.capture()
            rawImage = processed.image
        except EOFError as eof:
            # This case is where we just hit the end of an image set from disk
            self._log.info("Encountered end of image set")
            return constants.ProcessResult.EOF
        except IOError as io:
            # This is the case where something went wrong with a grab from a camera
            self._log.error("Encountered I/O Error {}".format(io))
            return constants.ProcessResult.INTERNAL_ERROR

        self._performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

       # ImageManipulation.show("Source",image)
        self._veg.SetImage(rawImage)
        self._performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        manipulated = ImageManipulation(rawImage, self._imageNumber, self._logger)
        self._logger.logImage(constants.FILENAME_RAW, manipulated.image)

        # manipulated.mmPerPixel = mmPerPixel
        # ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        self._performance.start()
        index = self._veg.Index(arguments.algorithm)
        self._performance.stopAndRecord(constants.PERF_INDEX)

        # ImageManipulation.show("index", index)
        # cv.imwrite("index.jpg", index)
        if arguments.plot:
            plot3D(index, arguments.algorithm)

        # Get the mask
        # mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)
        mask, threshold = self._veg.MaskFromIndex(index, not arguments.nonegate, 1, arguments.threshold)

        self._veg.applyMask()
        # This is the slow call
        # image = veg.GetMaskedImage()
        image = self._veg.GetImage()
        normalized = np.zeros_like(image)
        finalImage = cv.normalize(image, normalized, 0, 255, cv.NORM_MINMAX)
        if arguments.mask:
            filledMask = mask.copy().astype(np.uint8)
            cv.floodFill(filledMask, None, (0, 0), 255)
            filledMaskInverted = cv.bitwise_not(filledMask)
            manipulated.toGreyscale()
            threshold, imageThresholded = cv.threshold(manipulated.greyscale, 0, 255, cv.THRESH_BINARY_INV)
            finalMask = cv.bitwise_not(filledMaskInverted)
            self._logger.logImage("processed", finalImage)
            # veg.ShowImage("Thresholded", imageThresholded)
            self._logger.logImage("inverted", filledMaskInverted)
            # veg.ShowImage("Filled", filledMask)
            # veg.ShowImage("Inverted", filledMaskInverted)
            # veg.ShowImage("Final", finalMask)
            self._logger.logImage("final", finalMask)
            # plt.imshow(veg.imageMask, cmap='gray', vmin=0, vmax=1)
            # plt.imshow(finalImage)

            # print("X={}".format(x))            #plt.show()
            # logger.logImage("mask", veg.imageMask)
            # ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(finalImage, self._imageNumber, self._logger)
        manipulated.mmPerPixel = 20

        # TODO: Conversion to HSV should be done automatically
        self._performance.start()
        manipulated.toYCBCR()
        self._performance.stopAndRecord(constants.PERF_YCC)
        self._performance.start()
        manipulated.toHSV()
        self._performance.stopAndRecord(constants.PERF_HSV)
        self._performance.start()
        manipulated.toHSI()
        self._performance.stopAndRecord(constants.PERF_HSI)
        self._performance.start()
        manipulated.toYIQ()
        self._performance.stopAndRecord(constants.PERF_YIQ)

        # Find the plants in the image
        self._performance.start()
        contours, hierarchy, blobs, largest = manipulated.findBlobs(arguments.minarea)
        self._performance.stopAndRecord(constants.PERF_CONTOURS)

        # The test should probably be if we did not find any blobs
        if largest == "unknown":
            self._logger.logImage("error", manipulated.image)
            return constants.ProcessResult.NOT_PROCESSED

        self._performance.start()
        manipulated.identifyOverlappingVegetation()
        self._performance.stopAndRecord(constants.PERF_OVERLAP)

        # Set the classifier blob set to be the set just identified
        self._classifier.blobs = blobs

        self._performance.start()
        manipulated.computeShapeIndices()
        self._performance.stopAndRecord(constants.PERF_SHAPES_IDX)

        self._performance.start()
        manipulated.computeLengthWidthRatios()
        self._performance.stopAndRecord(constants.PERF_LW_RATIO)

        # New image analysis based on readings here:
        # http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf

        self._performance.start()
        manipulated.computeCompactness()
        manipulated.computeElogation()
        manipulated.computeEccentricity()
        manipulated.computeRoundness()
        manipulated.computeConvexity()
        manipulated.computeSolidity()
        self._performance.stopAndRecord(constants.PERF_SHAPES)
        # End image analysis

        # Classify items by where they are in image
        # This only marks items that can't be fully seen (at edges) of image
        self._classifier.classifyByPosition(size=manipulated.image.shape)

        # Determine the distance from the object to the edge of the image given  the pixel size of the camera
        self._performance.start()
        manipulated.computeDistancesToImageEdge(self._camera.getMMPerPixel(), self._camera.getResolution())
        self._performance.stopAndRecord(constants.PERF_DISTANCE)

        classifiedBlobs = self._classifier.blobs

        self._performance.start()
        manipulated.findAngles()
        manipulated.findCropLine()
        self._performance.stopAndRecord(constants.PERF_ANGLES)

        # Crop row processing
        manipulated.identifyCropRowCandidates()

        # Extract various features
        self._performance.start()

        # Compute the mean of the hue across the plant
        manipulated.extractImagesFrom(manipulated.hsi, 0, constants.NAME_HUE, np.nanmean)
        self._performance.stopAndRecord(constants.PERF_MEAN)
        manipulated.extractImagesFrom(manipulated.hsv, 1, constants.NAME_SATURATION, np.nanmean)

        # Discussion of YIQ can be found here
        # Sabzi, Sajad, Yousef Abbaspour-Gilandeh, and Juan Ignacio Arribas. 2020.
        # “An Automatic Visible-Range Video Weed Detection, Segmentation and Classification Prototype in Potato Field.”
        # Heliyon 6 (5): e03685.
        # The article refers to the I component as in-phase, but its orange-blue in the wikipedia description
        # of YIQ.  Not sure which is correct.

        # Compute the standard deviation of the I portion of the YIQ color space
        self._performance.start()
        manipulated.extractImagesFrom(manipulated.yiq, 1, constants.NAME_I_YIQ, np.nanstd)
        self._performance.stopAndRecord(constants.PERF_STDDEV)

        # Compute the mean of the blue difference in the ycc color space
        self._performance.start()
        manipulated.extractImagesFrom(manipulated.ycbcr, 1, constants.NAME_BLUE_DIFFERENCE, np.nanmean)
        self._performance.stopAndRecord(constants.PERF_MEAN)

        # Use either heuristics or logistic regression
        if arguments.logistic or arguments.knn or arguments.tree or arguments.forest or arguments.gradient:
            self._performance.start()
            self._classifier.classify()
            self._performance.stopAndRecord(constants.PERF_CLASSIFY)
            classifiedBlobs = self._classifier.blobs
        else:
            self._performance.start()
            self._classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=arguments.minratio)
            self._performance.stopAndRecord(constants.PERF_CLASSIFY)

        # Draw boxes around the images we found with decorations for attributes selected
        # manipulated.drawBoundingBoxes(contours)
        manipulated.drawBoxes(manipulated.name, classifiedBlobs, self._featuresToShow)

        # logger.logImage("cropline", manipulated.croplineImage)
        # This is using the hough transform which we abandoned as a technique
        # manipulated.detectLines()
        # TODO: Draw crop line as part of image decoration
        manipulated.drawCropline()
        # logger.logImage("crop-line", manipulated.croplineImage)
        if arguments.contours:
            manipulated.drawContours()

        # Everything in the image is classified, so decorate the image with distances
        manipulated.drawDistances()



        # Write out the processed image
        # cv.imwrite("processed.jpg", manipulated.image)
        self._logger.logImage("processed", manipulated.image)
        self._logger.logImage("original", manipulated.original)
        # ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # Write out the crop images so we can use them later
        # if self._arguments.xtract:
        #     manipulated.extractImages(classifiedAs=constants.TYPE_DESIRED)
        #     for blobName, blobAttributes in manipulated.blobs.items():
        #         if blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED:
        #             self._logger.logImage("crop", blobAttributes[constants.NAME_IMAGE])

        self._reporting.addBlobs(self._imageNumber, blobs)

        # if self._arguments.histograms:
        #     self._reporting.showHistogram("Areas", 20, constants.NAME_AREA)
        #     self._reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)

        return constants.ProcessResult.OK

    def process(self):
        """
        Process images -- this method will not return until all images are processed
        """
        processing = constants.ProcessResult.OK

        while processing == constants.ProcessResult.OK:
            processing = self._processor()


def resample(index: np.ndarray, targetX: int, targetY: int) -> np.ndarray:
    # Hardcode this for now -- depth is 1280x720, and we want 1920x1080

    #z = (1920 / 1280, 1080 / 720)
    z = (targetY / 1920, targetX / 1080)

    transformed = scipy.ndimage.zoom(index, z, order=0)
    return transformed

# The plotly version
def plot3D(index: np.ndarray, title: str):

    if not supportsPlotting:
        print("Unable to produce plots on this platform")
        return

    # I can get plotly to work only with square arrays, not rectangular, so just take a subset
    #subset = index[-1:1500, 0:1500]
    subset = index[0:1500, 0:1500]
    print("Index is {}".format(index.shape))
    print("Subset is {}".format(subset.shape))
    xi = np.linspace(-1, subset.shape[0], num=subset.shape[0])
    yi = np.linspace(-1, subset.shape[1], num=subset.shape[1])

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=subset)])

    # Can't get these to work
    #fig = go.Figure(data=[go.Mesh2d(x=xi, y=yi, z=subset, color='lightpink', opacity=0.50)])
    #fig = go.Figure(data=go.Isosurface(x=xi, y=yi,z=subset, isomin=-2, isomax=1))

    fig.update_layout(title=title, autosize=False,
                      width=999, height=1000,
                      margin=dict(l=64, r=50, b=65, t=90))

    fig.show()

# The matplotlib version is very slow to visualize and then rotate.
def plot2Dmatplotlib(index, title):

    if not supportsPlotting:
        log.error("Unable to produce plots on this platform")
        return

    downsampled = resample(index, 719, 1280)

    yLen,xLen = downsampled.shape
    x = np.arange(-1, xLen, 1)
    y = np.arange(-1, yLen, 1)
    log.debug("2D plot x: {} y: {}".format(x,y))

    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(9,10))
    axes = fig.gca(projection ='2d')
    plt.title(title)
    axes.scatter(x, y, downsampled, c=downsampled.flatten(), cmap='BrBG', s=-1.25)
    plt.show()
    cv.waitKey()

def readINI() -> OptionsFile:
    options = OptionsFile(arguments.ini)
    options.load()
    return options

# Used in command line processing so we can accept thresholds that are tuples
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)

_veg = VegetationIndex()

parser = argparse.ArgumentParser("Weed processing")

parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=_veg.GetSupportedAlgorithms(),
                    default="ngrdi")
parser.add_argument("-c", "--contours", action="store_true", default=False, help="Show contours on images")
parser.add_argument("-d", "--decorations", action="store", type=str, default="all", help="Decorations on output images (all and none are shortcuts)")
parser.add_argument("-df", "--data", action="store", help="Name of the data in CSV for use in logistic regression or KNN")
parser.add_argument("-hg", "--histograms", action="store_true", default=False, help="Show histograms")

parser.add_argument('-i', '--input', action="store", required=False, help="Images directory")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
group = parser.add_mutually_exclusive_group()
group.add_argument("-k", "--knn", action="store_true", default=False, help="Predict using KNN. Requires data file to be specified")
group.add_argument("-l", "--logistic", action="store_true", default=False, help="Predict using logistic regression. Requires data file to be specified")
group.add_argument("-dt", "--tree", action="store_true", default=False, help="Predict using decision tree. Requires data file to be specified")
group.add_argument("-f", "--forest", action="store_true", default=False, help="Predict using random forest. Requires data file to be specified")
group.add_argument("-g", "--gradient", action="store_true", default=False, help="Predict using gradient boosting. Requires data file to be specified")
group.add_argument("-svm", "--support", action="store_true", default=False, help="Predict using support vector machine. Requires data file to be specified")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
parser.add_argument("-mr", "--minratio", action="store", default=5, type=int, help="Minimum size ratio for classifier")
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
parser.add_argument("-r", "--results", action="store", default="results.csv", help="Name of results file")
parser.add_argument("-s", "--stitch", action="store_true", help="Stitch adjacent images together")
parser.add_argument("-sc", "--score", action="store_true", help="Score the prediction method")
parser.add_argument("-se", "--selection", action="store", default="all-parameters.csv", help="Parameter selection file")
parser.add_argument("-sp", "--spray", action="store_true", help="Generate spray treatment grid")
parser.add_argument("-stand", "--standalone", action="store_true", default=False, help="Run standalone and just process the images")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")

arguments = parser.parse_args()

theProcessor = Processor(arguments)
theProcessor.initialize()
sys.exit(theProcessor.process())





