#
# I M A G E
#
# Image manipulation
import gc
import uuid

import numpy
from PIL import Image
from skimage import color
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import pandas as pd
import logging
import colorsys
from math import pi
# from GPSPhoto import gpsphoto

from collections import namedtuple
from operator import mul

from skimage.color import rgb2yiq

from ImageLogger import ImageLogger

import constants
from GLCM import GLCM
from HOG import HOG
from LBP import LBP
from Performance import Performance

from Factors import FactorKind
from hashlib import sha1
import psutil
import os
from memory_profiler import profile

# Colors for the bounding boxes
COLOR_WEED = (0, 0, 255)
COLOR_CROP = (0, 255, 0)
COLOR_UNKNOWN = (255, 0, 0)
COLOR_UNTREATED = (0, 127, 0)
COLOR_IGNORED = (255, 255, 255)

# How far outside the midline of the image vegetation should be considered the cropline
MIDDLE_THRESHOLD = 200

# The lines for the enclosing rectangle
BOUNDING_BOX_THICKNESS = 16


class ImageManipulation:
    def __init__(self, img: np.ndarray, sequenceNumber: int, logger: ImageLogger):
        self._image = img
        self._name = constants.NAME_IMAGE + "-" + str(sequenceNumber)
        self._rectangles = []
        self._largestName = ""
        self._largestArea = 0
        # Scalar attributes go here
        self._blobs = {}
        # Vector attributes like the LBP histogram goes here
        self._blobsWithVectors = {}
        self._cropRowCandidates = {}
        self._mmPerPixel = 0
        #self._stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
        self._centers = []
        self._angles = None
        self._shapeIndices = []
        self._original = img.copy()
        self._imageAsBinary = np.ndarray
        self._imageAsRGB = None
        self._imageAsYIQ = None
        self._imgAsYCBCR = None
        self._imageAsYUV = None
        self._imageAsCIELAB = None
        self._imgAsGreyscale = None
        self._imgAsHSI = None
        self._logger = logger

        self._performance = None

        # Contours of the blobs
        self._contours = None

        # Threshold used in blob identification
        self._threshold = -9999

        (self._maxY, self._maxX, self._depth) = img.shape
        self._centerLineY = int(self._maxY / 2)

        self.log = logging.getLogger(__name__)

        # The minimum distance to the edge of crop that we will tolerate
        self._minDistanceToContour = 100

        self._hash = sha1(numpy.ascontiguousarray(img)).hexdigest()

        # Look of things in decorated images
        self._fontScale = 2.0

        # The bucket lids
        self._lids = {}

    def unload(self):
        """
        Debug memory leaks
        """
        self._image = None
        self._imageAsRGB = None
        self._imageAsYIQ = None
        self._imgAsYCBCR = None
        self._imageAsYUV = None
        self._imageAsCIELAB = None
        self._imgAsHSI = None
        self._imgAsHSV = None
        self._imgAsYCBCR = None
        self._imgAsCIELAB = None
        self._imgAsGreyscale = None
        del self._image
        del self._imageAsRGB
        del self._imageAsYIQ
        del self._imageAsYUV
        del self._imgAsCIELAB
        del self._imgAsHSI
        del self._imgAsHSV
        del self._imgAsYCBCR
        del self._imgAsGreyscale
        gc.collect()

    @property
    def fontscale(self) -> float:
        return self._fontScale

    @fontscale.setter
    def fontscale(self, scale: float):
        self._fontScale = scale

    @property
    def lids(self) -> {}:
        """
        Bucket lids in image.  Call the
        :return: list of lids
        """
        return self._lids

    @property
    def performance(self) -> Performance:
        """
        The performance object used
        :return:
        """
        return self._performance

    @performance.setter
    def performance(self, thePerformance: Performance):
        self._performance = thePerformance

    @property
    def hash(self) -> str:
        return self._hash
    @property
    def name(self) -> str:
        return self._name

    @property
    def original(self) -> np.ndarray:
        return self._original

    @property
    def binary(self) -> np.ndarray:
        return self._imageAsBinary

    @property
    def threshold(self) -> int:
        return self._threshold
    @property
    def mmPerPixel(self) -> float:
        return self._mmPerPixel

    @property
    def blobs(self):
        return self._blobs

    def blobsByType(self, factors: FactorKind) -> {}:
        """
        The blobs by type (Vector or Scalar)
        :param factors: FactorKind.VECTOR or FactorKind.SCALAR
        :return: dictionary of blobs
        """
        if factors == FactorKind.SCALAR:
            return self._blobs
        elif factors == FactorKind.VECTOR:
            return self._blobsWithVectors
        else:
            self.log.fatal(f"Unknown factor kind: {factors}")
            raise ValueError(f"Unknown factor kind: {factors}")

    @mmPerPixel.setter
    def mmPerPixel(self, mm: float):
        self._mmPerPixel = mm

    @property
    def image(self):
        return self._image

    @property
    def hsv(self):
        return self._imgAsHSV

    @property
    def hsi(self):
        return self._imgAsHSI

    @property
    def rgb(self):
        return self._imageAsRGB

    @property
    def yiq(self):
        if self._imageAsYIQ is None:
            return self.toYIQ()
        else:
            return self._imageAsYIQ

    @property
    def greyscale(self):
        return self._imgAsGreyscale

    @property
    def ycbcr(self):
        return self._imgAsYCBCR

    @property
    def yuv(self):
        return self._imageAsYUV

    @property
    def croplineImage(self):
        return self.cropline_image

    @property
    def shapeIndices(self):
        return self._shapeIndices

    @classmethod
    def show(self, title: str, index: np.ndarray):
        plt.title(title)
        plt.imshow(index, cmap='gray', vmin=0, vmax=255)
        plt.show()

    @classmethod
    def statistics(self, target: np.ndarray):
        nonZeroCells = np.count_nonzero(target > 0, keepdims=False)
        count = (target != 0.0).sum()
        return nonZeroCells

    @classmethod
    def save(self, image: np.ndarray, name: str):
        data = Image.fromarray((image * 255).astype(np.uint8))
        data.save(name)

    def addGPS(self, latitude: float, longitude: float):
        pass

    @staticmethod
    def write(image: np.ndarray, name: str):
        cv.imwrite(name, image)

    def toRGB(self) -> np.ndarray:
        """
        Converts the current image to RGB from BGR.
        :return: The converted image as an ndarray
        """
        if self._imageAsRGB is None:
            self._imageAsRGB = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2RGB)
        return self._imageAsRGB

    def toYUV(self) -> np.ndarray:
        """
        Converts the current image to YUV from BGR
        :return: The converted image as an ndarray
        """
        if self._imageAsYUV is None:
            self._imageAsYUV = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2YUV)
        return self._imageAsYUV

    # The YIQ colorspace is described here:
    # https://en.wikipedia.org/wiki/YIQ

    # TODO: This method is quite slow, taking almost 200 ms on test machine
    #@profile
    def toYIQ(self) -> np.ndarray:
        """
        Converts the current image to the YIQ colorspace from RGB.
        Converts to RGB automatically
        :return: The converted image as an ndarray
        """
        # Convert to RGB, as scikit-image doesn't take BGR
        self.toRGB()

        # TODO: This is the one and only use for the scikit-image library.
        # This can be done with some matrix multiplication instead, and is something that can
        # be performed on a GPU
        # This produces an enormous object: ~450MB
        # As an added bonus, it leaks memory
        # WARNING
        # This skips the code below.  Debugging
        self._imageAsYIQ = rgb2yiq(self._imageAsRGB)
        return self._imageAsYIQ

        # self._imageAsYIQ = np.zeros_like(self._imageAsRGB)
        self._imageAsYIQ = self._imageAsRGB

        # WARNING -- debug
        #return self._imageAsYIQ

        # Modified code from:
        # https://stackoverflow.com/questions/67508567/problem-during-converting-rgb-to-yiq-color-mode-and-vice-versa-in-uint8
        BGR = self._imageAsRGB.copy().astype(float)
        R = BGR[:, :, 0]
        G = BGR[:, :, 1]
        B = BGR[:, :, 2]

        Y = (0.299 * R) + (0.587 * G) + (0.114 * B)
        I = (0.59590059 * R) + (-0.27455667 * G) + (-0.32134392 * B)
        Q = (0.21153661 * R) + (-0.52273617 * G) + (0.31119955 * B)

        YIQ = np.round(np.dstack((Y, I + 128, Q + 128))).astype(np.uint8)

        self._imageAsYIQ = YIQ


        # Code modified from
        # https://stackoverflow.com/questions/61348558/rgb-to-yiq-and-back-in-python
        # yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
        #                          [0.59590059, -0.27455667, -0.32134392],
        #                          [0.21153661, -0.52273617, 0.31119955]])
        # origShape = self._imageAsRGB.shape
        # self._imageAsYIQ = np.dot(self._imageAsRGB.reshape(-1, 3), yiq_from_rgb.transpose()).reshape(origShape)

        return self._imageAsYIQ


    def toHSV(self) -> np.ndarray:
        """
        The current image converted to the HSV colorspace
        :return:
        The HSV values as a numpy array
        """
        self._imgAsHSV = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2HSV)
        return self._imgAsHSV

    def toYCBCR(self) -> np.ndarray:
        """
        The current image converted to the YCbCr (YCC) colorspace
        :return:
        The YCbCR values as a numpy array
        """
        self._imgAsYCBCR = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2YCR_CB)
        return self._imgAsYCBCR

    def toCIELAB(self) -> np.ndarray:
        """
        The current image convered to the CIELAB colorspace
        :return:
        The CIELAB values as a numpy array
        """
        self._imgAsCIELAB = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2Lab)
        return self._imgAsCIELAB

    # This code doesn't work exactly right, as I see negative values for saturation
    # And this is unacceptably slow.  Takes over 700 ms on my machine

    #@profile
    def toHSI(self) -> np.ndarray:
        """
        The current image converted to ths HSI colorspace
        :return:
        The HSI values as numpy array
        """
        # TODO: HSI Implementation
        # W A R N I N G
        # temporary for memory debug
        # self.toRGB()
        self._imgAsHSI = self.toCIELAB()
        return self._imgAsHSI

        # blue = self._image[:, :, 0]
        # green = self._image[:, :, 1]
        # red = self._image[:, :, 2]


        # end temporary

        # Adapted from: https://stackoverflow.com/questions/52834537/rgb-to-hsi-conversion-hue-always-calculated-as-0
        #       with np.errstate(divide='ignore', invalid='ignore'):

        bgr = np.int32(cv.split(self._image))

        blue = bgr[0]
        green = bgr[1]
        red = bgr[2]
        # self.log.debug("RED min/max {}/{}".format(red.min(),red.max()))
        # self.log.debug("GREEN min/max {}/{}".format(green.min(),green.max()))
        # self.log.debug("BLUE min/max {}/{}".format(blue.min(),blue.max()))

        intensity = np.divide(blue + green + red, 3)

        minimum = np.minimum(np.minimum(red, green), blue)
        minimum = np.where(minimum == 0, .00001, minimum)
        rgb = red + green + blue
        # Avoid having missed datapoints here
        rgb = np.where(rgb == 0, .00001, rgb)

        self.log.debug("Min/max {}/{}".format(minimum.min(), minimum.max()))
        saturation = 1 - 3 * np.divide(minimum, rgb)

        sqrt_calc = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue)))

        # Avoid having missed datapoints here
        sqrt_calc = np.where(sqrt_calc == 0, 1, sqrt_calc)

        if (green > blue).any():
            hue = np.arccos((0.5 * ((red - green) + (red - blue)) / sqrt_calc))
        else:
            hue = 2 * pi - np.arccos((1 / 2 * ((red - green) + (red - blue)) / sqrt_calc))

        hue = hue * 180 / pi

        self._imgAsHSI = cv.merge((hue, saturation, intensity))
        return self._imgAsHSI

    # This code is way too slow
    def _RGB2HSI(self):
        """
             This is the function to convert RGB color image to HSI image
             :param rgm_img: RGB color image
             :return: HSI image
        """
        rgb_img = self._image
        # Save the number of rows and columns of the original image
        row = np.shape(rgb_img)[0]
        col = np.shape(rgb_img)[1]
        # Copy the original image
        hsi_img = rgb_img.copy()
        # Channel splitting the image
        B, G, R = cv.split(rgb_img)
        # Normalize the channel to [0,1]
        [B, G, R] = [i / 255.0 for i in ([B, G, R])]
        H = np.zeros((row, col))  # Define H channel
        I = (R + G + B) / 3.0  # Calculate I channel
        S = np.zeros((row, col))  # Define S channel
        for i in range(row):
            den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
            thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)  # Calculate the included angle
            h = np.zeros(col)  # Define temporary array
            # den>0 and G>=B element h is assigned to thetha
            h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
            # den>0 and G<=B element h is assigned to thetha
            h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
            # den<0 element h is assigned a value of 0
            h[den == 0] = 0
            H[i] = h / (2 * np.pi)  # Assign to the H channel after radiating
        # Calculate S channel
        for i in range(row):
            min = []
            # Find the minimum value of each group of RGB values
            for j in range(col):
                arr = [B[i][j], G[i][j], R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            # Calculate S channel
            S[i] = 1 - min * 3 / (R[i] + B[i] + G[i])
            # I is 0 directly assigned to 0
            S[i][R[i] + B[i] + G[i] == 0] = 0
        # Extend to 255 for easy display, generally H component is between [0,2pi], S and I are between [0,1]
        # hsi_img[:,:,0] = H*255
        # hsi_img[:,:,1] = S*255
        # hsi_img[:,:,2] = I*255
        self._imgAsHSI = hsi_img
        return hsi_img

    def RGB2HSI(self, img: np.ndarray):
        """
        Convert the RGB image to HSI
        Adapted from: https://github.com/SVLaursen/Python-RGB-to-HSI/blob/master/converter.py
        :param img: RGB image
        :return: hsi image
        """
        with np.errstate(divide='ignore', invalid='ignore'):

            # Load image with 32 bit floats as variable type
            bgr = np.float32(img) / 255

            # Separate color channels
            blue = bgr[:, :, 0]
            green = bgr[:, :, 1]
            red = bgr[:, :, 2]

            # Calculate Intensity
            def calc_intensity(red, blue, green):
                return np.divide(blue + green + red, 3)

            # Calculate Saturation
            def calc_saturation(red, blue, green):
                minimum = np.minimum(np.minimum(red, green), blue)
                saturation = 1 - (3 / (red + green + blue + 0.001) * minimum)

                return saturation

            # Calculate Hue
            def calc_hue(red, blue, green):
                hue = np.copy(red)

                for i in range(0, blue.shape[0]):
                    for j in range(0, blue.shape[1]):
                        hue[i][j] = 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / \
                                    math.sqrt((red[i][j] - green[i][j]) ** 2 +
                                              ((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
                        hue[i][j] = math.acos(hue[i][j])

                        if blue[i][j] <= green[i][j]:
                            hue[i][j] = hue[i][j]
                        else:
                            hue[i][j] = ((360 * math.pi) / 180.0) - hue[i][j]

                return hue

            # Merge channels into picture and return image
            hsi = cv.merge(
                (calc_hue(red, blue, green), calc_saturation(red, blue, green), calc_intensity(red, blue, green)))
            return hsi
    def toGreyscale(self) -> np.ndarray:
        """
        The current image converted to greyscale
        :return:
        The greyscale image as a numpy array
        """
        # This method of converting to greyscale is a complete hack.
        # self.save(self._image, "temporary.jpg")
        # utility.SaveMaskedImage("mask-applied.jpg")
        # img_float32 = np.float32(utility.GetMaskedImage())
        # img = cv.imread("temporary.jpg")
        # self._imgAsGreyscale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # self._imgAsGreyscale = self._image.astype(np.uint8)
        # If the conversion to uint8 is not there, opencv complains when we try to find the
        # contours.  Strictly speaking this is not required for just the greyscale conversion

        # Blurring the image before we start trying to detect objects seems to improve things
        # in that noise is not identified as objects, but this is a very slow method

        # blurred = cv.pyrMeanShiftFiltering(self._image.astype(np.uint8),31,101)
        # img = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        #
        # Sujith, A., and R. Neethu. 2021. “Classification of Plant Leaf Using Shape and Texture Features.”
        # In 4th International Conference on Inventive Communication and Computational Technologies,
        # ICICCT 2020, edited by Ranganathan G., Chen J., and Rocha A., 145:269–82.
        # Springer Science and Business Media Deutschland GmbH.
        # This article gives the greyscale conversion as:
        # grey = 0.2989 * R + 0.5870 * G + 0.1140 * B
        # TODO: Check the grayscale conversion from opencv
        if self._imgAsGreyscale is None:
            img = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2GRAY)
            # cv.imwrite("converted.jpg", img)
            self._imgAsGreyscale = img

        return self._imgAsGreyscale

    def toDGCI(self, original: str):
        """

        :param original:
        :return:
        """
        # Formula from this article
        # https://www.petiolepro.com/blog/dark-green-colour-index-dgci-a-new-measurement-of-chlorophyll/
        # DGCI = {(hue − 60)/60 + (1 − saturation) + (1 − brightness)}/3
        img = self._imageAsRGB
        height, width, channels = np.shape(self._imageAsRGB)

        # Confirm this is a color image
        assert channels == 3

        # Use this as guidance
        # https://acsess-onlinelibrary-wiley-com.ezproxy4.library.arizona.edu/doi/10.2135/cropsci2003.9430
        # Normalize values in range (0..1)
        rgbNormalized = self._imageAsRGB.astype(np.uint8) / 255

        dgci = np.zeros_like(img[:, :, 0])

        for x in range(height):
            for y in range(width):
                r, g, b = rgbNormalized[x, y]
                maximum = max(r, g, b)
                minimum = min(r, g, b)

                # H U E
                # There seem to be 3 different formulae depending on which channel is max

                # Avoid division by zero
                maximum += 0.00001

                if maximum != minimum:
                    # Red is max
                    if max(b, g, r) == r:
                        h = 60 * ((g - b) / (maximum - minimum))
                    # Green is max
                    elif max(b, g, r) == g:
                        h = 60 * (2.0 + (b - r) * (maximum - minimum))
                    # Blue is max
                    elif max(b, g, r) == b:
                        h = 60 * (4.0 + (r - g) * (maximum - minimum))
                    else:
                        assert False
                else:
                    h = 0

                # L I G H T N E S S
                lightness = maximum

                # S A T U R A T I O N
                saturation = (maximum - minimum) / maximum

                dgciAtPoint = ((h - 60) / 60 + (1 - saturation) + (1 - lightness)) / 3

                dgci[x][y] = dgciAtPoint
        pass
        self.log.debug("DGCI Calculated")
        self._imageAsDGCI = dgci

    def equalizeContrast(self):
        """
        Equalize the contrast. For color images, this means a conversion to another color space and then back
        Note that this method only applies the histogram equalization to the BGR image, so this should be called
        before and conversion to other color spaces.
        """
        img = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2HSV)
        img[:, :, 2] = cv.equalizeHist(img[:, :, 2])
        self._image = cv.cvtColor(img.astype(np.uint8), cv.COLOR_HSV2BGR)
        return self._image

    def findEdges(self, image: np.ndarray):
        self._edges = cv.Canny(self._imgAsGreyscale, 20, 30)
        return

    def cartoon(self):
        self._cartooned = np.where(self._imgAsGreyscale > 0, 255, self._imgAsGreyscale)
        return self._cartooned

    def mmBetweenPoints(self, point1: (), point2: (), mmPerPixel: float) -> int:
        """
        Find the physical distance between two points.
        :param point1: A point as a tuple
        :param point2: A point as a tuple
        :param mmPerPixel: The distance a single pixel covers
        :return:
        The distance between two points as integer
        """
        distance = 0
        (x1, y1) = point1
        (x2, y2) = point2
        distance = int((x2 - x1) * mmPerPixel)
        return distance

    @staticmethod
    def sizeRatio(sizeOfTarget: int, sizeOfLargest: int) -> float:
        """
        The percentage of the area of the target relative to the largest item.
        :param sizeOfTarget:  The area of blob to be checked
        :param sizeOfLargest:  The area of the largest blob in the current image
        :return: A float value indicating the size ratio of the target to the largest
        """
        return sizeOfTarget / sizeOfLargest

    #@profile
    def findBlobs(self, threshold: int, strategy: constants.Strategy) -> ([], np.ndarray, {}, str):
        """
        Find objects within the current image
        :param threshold: Minimum area of object to be considered a blob
        :param strategy:
        :return: (contours, hierarchy, bounding rectangles, name of largest object)
        """
        # self.log.debug(f"Memory % used before processing: {psutil.Process(os.getpid()).memory_percent()}")
        self.toGreyscale()
        # self.show("grey", self._imgAsGreyscale)
        self.write(self._imgAsGreyscale, "greyscale.jpg")

        self.cartoon()
        # self.show("cartooned", self._cartooned)
        # self.write(self._image, "original.jpg")
        self._logger.logImage("cartooned", self._cartooned)

        # self.write(self._image, "index.jpg")

        # Convert to binary image
        # Works
        # ret,thresh = cv.threshold(self._cartooned,127,255,0)
        #ret, thresh = cv.threshold(self._imgAsGreyscale, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        ret, thresh = cv.threshold(self._imgAsGreyscale, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        # self.log.debug(f"Memory % used after threshold: {psutil.Process(os.getpid()).memory_percent()}")

        self._threshold = thresh
        self.write(thresh, "threshold.jpg")
        kernel = np.ones((5, 5), np.uint8)
        # Debug -- eliminate erosion
        # erosion = cv.erode(self._cartooned, kernel, iterations=4)
        # self.write(erosion, "erosion.jpg")
        erosion = cv.dilate(thresh, kernel, iterations=1)  # originally 3
        # self.log.debug(f"Memory % used after dilation: {psutil.Process(os.getpid()).memory_percent()}")
        self.write(erosion, "after-dilation.jpg")

        closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel)
        # self.log.debug(f"Memory % used after closing: {psutil.Process(os.getpid()).memory_percent()}")
        # self.write(closing, "closing.jpg")
        self._logger.logImage("closing", closing)
        # self.show("binary", erosion)
        # self.write(erosion, "binary.jpg")
        self._logger.logImage("erosion", erosion)
        self._imageAsBinary = erosion

        # Originally
        # candidate = erosion

        largestName = "unknown"
        area = 0
        candidate = closing
        # self.write(candidate, "candidate.jpg")
        self._logger.logImage("candidate", candidate)
        # find contours in the binary image
        # im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        # We don't need the hierarchy at this point, so the RETR_EXTERNAL seems faster
        # contours, hierarchy = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        if strategy == constants.Strategy.PROCESSED:
            contours, hierarchy = cv.findContours(candidate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            # self.log.debug(f"Memory % used after finding contours: {psutil.Process(os.getpid()).memory_percent()}")
        else:
            # self.write(closing, "closing.jpg")
            # Find the contours in the threshold instead of the candidate
            #contours, hierarchy = cv.findContours(self._cartooned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            contours, hierarchy = cv.findContours(self._cartooned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # self.log.debug(f"Memory % used after finding contours: {psutil.Process(os.getpid()).memory_percent()}")

        self._contours = contours

        # Calculate the area of each box
        i = 0
        largest = 0
        for c in contours:
            M = cv.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            x, y, w, h = cv.boundingRect(c)
            # The area of the bounding rectangle
            # area = w*h
            # The area of the vegetation
            area = cv.contourArea(c)
            type = constants.TYPE_UNKNOWN
            location = (x, y, w, h)
            center = (cX, cY)
            reason = constants.REASON_UNKNOWN
            hue = 0.0
            saturationMean = 0.0
            yiqStdDeviation = 0.0
            blueDifferenceMean = 0.0

            infoAboutBlob = {constants.NAME_LOCATION: location,
                             constants.NAME_CENTER: center,
                             constants.NAME_AREA: area,
                             constants.NAME_TYPE: type,
                             constants.NAME_CONTOUR: c,
                             constants.NAME_REASON: reason,
                             constants.NAME_NEIGHBOR_COUNT: 0,
                             constants.NAME_HUE: hue,
                             constants.NAME_SATURATION: saturationMean,
                             constants.NAME_I_YIQ: yiqStdDeviation,
                             constants.NAME_BLUE_DIFFERENCE: blueDifferenceMean}

            name = constants.NAME_BLOB + constants.DASH + str(i)
            # Ignore items in the image that are smaller in area than the
            # threshold.  Things in shadow and noise will be identified as shapes
            if area > threshold:
                self._blobs[name] = infoAboutBlob
                self._blobsWithVectors[name] = infoAboutBlob.copy()
                i = i + 1

            # Determine the largest blob in the image
            if area > largest:
                largest = area
                largestName = name

        self._largestName = largestName
        self._largestArea = area

        self._hierarchy = hierarchy

        # Insert size ratios.  We can do this only once we have determined the largest item in the image
        for blobName, blobAttributes in self._blobs.items():
            blobAttributes[constants.NAME_SIZE_RATIO] = blobAttributes[constants.NAME_AREA] / largest

        return contours, hierarchy, self._blobs, largestName

    @staticmethod
    def _contains(circle: [], circle2: []) -> bool:
        """
        Check if circle1 is completely contained within circle2.
        :param circle: Array of [x, y, radius]
        :param circle2: Array of [x, y, radius]
        :return: True if circle2 contained within circle1, False otherwise
        """
        # Adapted from https://stackoverflow.com/questions/33490334/check-if-a-circle-is-contained-in-another-circle
        d = math.sqrt(
            (circle[0] - circle2[0]) ** 2 +
            (circle[1] - circle2[1]) ** 2)
        return circle2[2] > (d + circle[2])

    def nameLids(self, blobs: {}):
        """
        Find objects within the current image
        :param threshold: Minimum area of object to be considered a blob
        :param strategy:
        :param lids: The lids in the image
        :return: (contours, hierarchy, bounding rectangles)
        """
        # A Hough transform approach
        self.toGreyscale()
        blur = cv.medianBlur(self._imgAsGreyscale, 41)
        #blur = cv.blur(self._imgAsGreyscale, (7, 7))
        self._logger.logImage("blur", blur)

        # Perhaps there is an opencv bug or something I don't understand.  If the circles are perfectly concentric,
        # with centers 0 pixels apart, both the circles are found. Depending on the lighting conditions, this could be several circles
        # that are all larger than the stickers applied.  The solution to this is to throw away everything but the largest
        # and the smallest. The problem is that there may be two lids in view, so we can't just throw away everything but the
        # smallest and largest overall, we need to throw away everything in each lid _except_ the smallest.

        circles = cv.HoughCircles(blur, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=20, param1=150, param2=0.8, minRadius=5, maxRadius=0)
        circles = np.uint16(np.around(circles))
        self.log.debug(f"Circles in image prior to correction: {np.shape(circles)[1]}")

        # No circles were found
        if np.shape(circles)[1] == 0:
            return

        # This is the case where a circular lid and at least one sticker can't be located
        if np.shape(circles)[1] == 1:
            for i in circles[0, :]:
                cv.circle(self._image, (i[0], i[1]), i[2], (0, 255, 0), 3)
            self._logger.logImage("circles", self._image)
            return

        # If the circle count is 2+, there must be at least one lid

        # Initially, all the lids have 0 stickers
        stickers = np.zeros((circles[0].shape[0], 1))
        types = np.zeros((circles[0].shape[0], 1))

        withStickers = np.concatenate((circles[0], stickers), axis=1)
        hierarchy = np.concatenate((withStickers, types), axis=1)

        largestRadius = np.max(hierarchy[:, 2])
        LID = 0
        STICKER = 1
        # Determine the types of things
        lids = []
        stickers = []

        # Eliminate the concentric circles found
        x = 0
        y = 0
        candidates = []
        concentricCircleFound = False
        while not concentricCircleFound:
            concentricCircleFound = False
            candidates = []
            concentricWith = -1
            currentCircle = -1
            for i in hierarchy:
                currentCircle += 1
                if math.isclose(i[0], x, abs_tol=5) and math.isclose(i[1], y, abs_tol=5):
                    self.log.debug(f"Found concentric circle at ({i[0]},{i[1]})")
                    concentricCircleFound = True
                else:
                    concentricWith = currentCircle
                    x = i[0]
                    y = i[1]
                    candidates.append(i)


        for i in candidates:
            # Lids are several times the size of the stickers. This should even identify a lid that is further away.
            if largestRadius / i[2] < 2:
                i[4] = LID
                lids.append(i)
            else:
                i[4] = STICKER
                stickers.append(i)

            self.log.debug(f"Center of circle: ({i[0]},{i[1]}) Radius: {i[2]} Parent: {i[3]}")


        # Step through the lids and stickers to determine the number of stickers on a lid
        lidID = 0
        for lid in lids:
            stickerCount = 0
            stickerID = 0
            for sticker in stickers:
                if self._contains(sticker, lid):
                    self.log.debug(f"Lid {lidID} at ({lid[0]},{lid[1]}) contains sticker {stickerID}")
                    stickerCount += 1
                stickerID += 1
            # The name of the lid is determined by the number of stickers it contains
            lidName = constants.NAME_LID + constants.DASH + str(stickerCount)
            self.log.debug(f"Lid name is {lidName} at ({lid[0]},{lid[1]})")

            # Rename the blob to the lid name by looking at the blob center proximity to the lid
            # This gets a bit messy as we want to update the key in the dictionary, so it's easier to just create
            # a new one and use that
            lidsFinal = {}
            for blobName, blobProperties in blobs.items():
                (xBlob, yBlob) = blobProperties[constants.NAME_CENTER]
                if math.isclose(xBlob, lid[0], abs_tol=10) and math.isclose(yBlob, lid[1], abs_tol=10):
                    self.log.debug(f"Blob {blobName} should be renamed {lidName}")
                    blobProperties[constants.NAME_NAME] = lidName
                    lidsFinal[lidName] = blobProperties
                else:
                    self.log.debug(f"Blob at ({xBlob},{yBlob}) does not match position")

            lidID += 1
        self._lids = lidsFinal

        # Debug -- safe to remove
        #for i in circles[0, :]:
        circles = np.array(candidates)
        circles = np.uint16(np.around(circles))
        for i in circles:
            cv.circle(self._image, (i[0], i[1]), i[2], (0, 255, 0), 3)
        self._logger.logImage("circles", self._image)
        # End debug

        # The blobs parameter is the list of lids, but has slightly different centers to the circle we determined was a lid
        for blobName, blobProperties in blobs.items():
            (xBlob, yBlob) = blobProperties[constants.NAME_CENTER]
            self.log.debug(f"{blobName} is at ({xBlob}, {yBlob})")

        return

        # Adapted from this discusssion:
        # https://stackoverflow.com/questions/51456660/opencv-detecting-drilled-holes
        self.toGreyscale()
        blur = cv.medianBlur(self._imgAsGreyscale, 31)

        ret, thresh = cv.threshold(blur, 127, 255, cv.THRESH_OTSU)

        canny = cv.Canny(thresh, 150, 200)
        self._logger.logImage("edges", canny)
        # cv2.imshow('canny', canny)

        contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        contour_list = []
        for contour in contours:
            approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
            area = cv.contourArea(contour)
            self.log.debug(f"Contour area: {area}")
            if 1000 < area < 15000:
                contour_list.append(contour)

        msg = "Total holes: {}".format(len(contour_list) // 2)
        cv.putText(self._image, msg, (20, 40), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)

        cv.drawContours(self._image, contour_list, -1, (0, 255, 0), 2)
        self._logger.logImage("holes", self._image)

    # This is the original logic
    def _findBlobs(self, threshold: int) -> ([], np.ndarray, {}, str):
        """
        O R I G I N A L
        Find objects within the current image
        :return: (contours, hierarchy, bounding rectangles, name of largest object)
        """
        self.toGreyscale()
        # self.show("grey", self._imgAsGreyscale)
        # self.write(self._imgAsGreyscale, "greyscale.jpg")

        self.cartoon()
        # self.show("cartooned", self._cartooned)
        # self.write(self._image, "original.jpg")
        self.write(self._cartooned, "cartooned.jpg")

        # self.write(self._image, "index.jpg")

        # Convert to binary image
        # Works
        # ret,thresh = cv.threshold(self._cartooned,127,255,0)
        #ret, thresh = cv.threshold(self._imgAsGreyscale, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        ret, thresh = cv.threshold(self._imgAsGreyscale, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        self.write(thresh, "threshold.jpg")
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv.erode(self._cartooned, kernel, iterations=4)
        # self.write(erosion, "erosion.jpg")
        erosion = cv.dilate(erosion, kernel, iterations=3)  # originally 3
        self.write(erosion, "after-dilation.jpg")

        closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel)
        # self.write(closing, "closing.jpg")
        self._logger.logImage("closing", closing)
        # self.show("binary", erosion)
        # self.write(erosion, "binary.jpg")
        self._logger.logImage("erosion", erosion)
        self._imageAsBinary = erosion

        # Originally
        # candidate = erosion

        largestName = "unknown"
        area = 0
        candidate = closing
        # self.write(candidate, "candidate.jpg")
        self._logger.logImage("candidate", candidate)
        # find contours in the binary image
        # im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        # We don't need the hierarchy at this point, so the RETR_EXTERNAL seems faster
        # contours, hierarchy = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(candidate, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        # Find the contours in the threshold instead of the candidate
        #contours, hierarchy = cv.findContours(self._cartooned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        self._contours = contours

        # Calculate the area of each box
        i = 0
        largest = 0
        for c in contours:
            M = cv.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            x, y, w, h = cv.boundingRect(c)
            # The area of the bounding rectangle
            # area = w*h
            # The area of the vegetation
            area = cv.contourArea(c)
            type = constants.TYPE_UNKNOWN
            location = (x, y, w, h)
            center = (cX, cY)
            reason = constants.REASON_UNKNOWN
            hue = 0.0
            saturationMean = 0.0
            yiqStdDeviation = 0.0
            blueDifferenceMean = 0.0

            infoAboutBlob = {constants.NAME_LOCATION: location,
                             constants.NAME_CENTER: center,
                             constants.NAME_AREA: area,
                             constants.NAME_TYPE: type,
                             constants.NAME_CONTOUR: c,
                             constants.NAME_REASON: reason,
                             constants.NAME_NEIGHBOR_COUNT: 0,
                             constants.NAME_HUE: hue,
                             constants.NAME_SATURATION: saturationMean,
                             constants.NAME_I_YIQ: yiqStdDeviation,
                             constants.NAME_BLUE_DIFFERENCE: blueDifferenceMean}

            name = "blob" + str(i)
            # Ignore items in the image that are smaller in area than the
            # threshold.  Things in shadow and noise will be identified as shapes
            if area > threshold:
                self._blobs[name] = infoAboutBlob
            i = i + 1

            # Determine the largest blob in the image
            if area > largest:
                largest = area
                largestName = name

        self._largestName = largestName
        self._largestArea = area

        self._hierarchy = hierarchy

        # Insert size ratios.  We can do this only once we have determined the largest item in the image
        for blobName, blobAttributes in self._blobs.items():
            blobAttributes[constants.NAME_SIZE_RATIO] = blobAttributes[constants.NAME_AREA] / largest

        return contours, hierarchy, self._blobs, largestName

    def identifyOverlappingVegetation(self):
        i = 0

        if self._hierarchy is None:
            return

        # walk through the hierarchy to determine if any blob is contained within another
        for contour in self._hierarchy[0]:
            (next, previous, child, parent) = contour
            name = "blob" + str(i)
            # If an object has a parent, that means it is contained within another
            if parent != -1 and name in self._blobs:
                attributes = self._blobs[name]
                # print("Find: " + str(attributes[constants.NAME_CENTER]))
                # Determine if the point is within the blob
                isInsideContour = cv.pointPolygonTest(attributes[constants.NAME_CONTOUR],
                                                      attributes[constants.NAME_CENTER], False)
                if isInsideContour:
                    attributes[constants.NAME_TYPE] = constants.TYPE_IGNORED

                # Find the distance to the contour
                distance = cv.pointPolygonTest(attributes[constants.NAME_CONTOUR], attributes[constants.NAME_CENTER],
                                               True)
                if distance < self._minDistanceToContour:
                    self.log.debug("Point is {} away from contour".format(distance))
                # Determine if the blob has as a parent
                # if name in self._blobs:
                #     attributes = self._blobs[name]
                #     attributes[constants.NAME_TYPE] = constants.TYPE_UNTREATED
                #     print("detected overlap")
            i = i + 1

    def computeHOG(self):
        """
        HOG Computations for all objects in image
        """
        self.log.debug("HOG")

        hog = HOG(self._blobs, constants.NAME_IMAGE)
        hog.computeAttributes()
        self._blobs = hog.blobs

    def computeLBP(self):
        """
        LBP Computations for all objects in image
        """
        self.log.debug("LBP")

        # GREYSCALE
        lbp = LBP(self._blobs, constants.NAME_GREYSCALE_IMAGE, 24, 8)
        lbp.prefix = constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP
        lbp.compute()
        self._blobs = lbp.blobs

        colorBands = {
            constants.NAME_YIQ_Y:                 {'base': constants.NAME_IMAGE_YIQ, 'prefix': constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_LBP, "band": 0},
            constants.NAME_YIQ_I:                 {'base': constants.NAME_IMAGE_YIQ, 'prefix': constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_LBP, "band": 1},
            constants.NAME_YIQ_Q:                 {'base': constants.NAME_IMAGE_YIQ, 'prefix': constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_LBP, "band": 2},
            constants.NAME_HSI_HUE:               {'base': constants.NAME_IMAGE_HSI, 'prefix': constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_LBP, "band": 0},
            constants.NAME_HSI_SATURATION:        {'base': constants.NAME_IMAGE_HSI, 'prefix': constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_LBP, "band": 1},
            constants.NAME_HSI_INTENSITY:         {'base': constants.NAME_IMAGE_HSI, 'prefix': constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_LBP, "band": 2},
            constants.NAME_RED:                   {'base': constants.NAME_IMAGE_RGB, 'prefix': constants.NAME_RED + constants.DELIMETER + constants.NAME_LBP, "band": 0},
            constants.NAME_BLUE:                  {'base': constants.NAME_IMAGE_RGB, 'prefix': constants.NAME_BLUE + constants.DELIMETER + constants.NAME_LBP, "band": 1},
            constants.NAME_GREEN:                 {'base': constants.NAME_IMAGE_RGB, 'prefix': constants.NAME_GREEN + constants.DELIMETER + constants.NAME_LBP, "band": 2},
            constants.NAME_HSV_HUE:               {'base': constants.NAME_IMAGE_HSV, 'prefix': constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_LBP, "band": 0},
            constants.NAME_HSV_SATURATION:        {'base': constants.NAME_IMAGE_HSV, 'prefix': constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_LBP, "band": 1},
            constants.NAME_HSV_VALUE:             {'base': constants.NAME_IMAGE_HSV, 'prefix': constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_LBP, "band": 2},
            constants.NAME_YCBCR_LUMA:            {'base': constants.NAME_IMAGE_YCBCR, 'prefix': constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_LBP, "band": 0},
            constants.NAME_YCBCR_BLUE_DIFFERENCE: {'base': constants.NAME_IMAGE_YCBCR, 'prefix': constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP, "band": 1},
            constants.NAME_YCBCR_RED_DIFFERENCE:  {'base': constants.NAME_IMAGE_YCBCR, 'prefix': constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP, "band": 2},
            constants.NAME_CIELAB_L:              {'base': constants.NAME_IMAGE_CIELAB, 'prefix': constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_LBP, "band": 0},
            constants.NAME_CIELAB_A:              {'base': constants.NAME_IMAGE_CIELAB, 'prefix': constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_LBP, "band": 1},
            constants.NAME_CIELAB_B:              {'base': constants.NAME_IMAGE_CIELAB, 'prefix': constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_LBP, "band": 2},

        }

        for color, details in colorBands.items():
            lbp = LBP(self._blobs, details['base'], 24, 8, PREFIX=details['prefix'], BAND=int(details['band']))
            lbp.compute()
            self._blobs = lbp.blobs

    def computeGLCM(self):
        """
        GLCM Computations for all objects in image.
        """
        #self.log.debug("GLCM: Greyscale")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_GREYSCALE_IMAGE, PREFIX=constants.NAME_GREYSCALE)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_GREYSCALE)

        #self.log.debug("GLCM: YIQ Y")
        self.performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_YIQ, PREFIX=constants.NAME_YIQ_Y, BAND=0)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_YIQ_Y)

        #self.log.debug("GLCM: YIQ I")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_YIQ, PREFIX=constants.NAME_YIQ_I, BAND=1)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_YIQ_I)

        #self.log.debug("GLCM: YIQ Q")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_YIQ, PREFIX=constants.NAME_YIQ_Q, BAND=2)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_YIQ_Q)

        #self.log.debug("GLCM: HSV Hue")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_HSV, PREFIX=constants.NAME_HSV_HUE, BAND=0)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_HSV_H)

        #self.log.debug("GLCM: HSV Saturation")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_HSV, PREFIX=constants.NAME_HSV_SATURATION, BAND=1)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_HSV_H)

        #self.log.debug("GLCM: HSV Value")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_HSV, PREFIX=constants.NAME_HSV_VALUE, BAND=2)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_HSV_H)

        #self.log.debug("GLCM: Blue")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE, PREFIX=constants.NAME_BLUE, BAND=0)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_RGB_B)

        #self.log.debug("GLCM: Green")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE, PREFIX=constants.NAME_GREEN, BAND=1)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_RGB_B)

        #self.log.debug("GLCM: Red")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE, PREFIX=constants.NAME_RED, BAND=2)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_RGB_B)

        #self.log.debug("GLCM: HSI H")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_HSI, PREFIX=constants.NAME_HSI_HUE, BAND=0)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_HSI_H)

        #self.log.debug("GLCM: HSI S")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_HSI, PREFIX=constants.NAME_HSI_SATURATION, BAND=1)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_HSI_S)

        #self.log.debug("GLCM: HSI I")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_HSI, PREFIX=constants.NAME_HSI_INTENSITY, BAND=2)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_HSI_I)

        #self.log.debug("GLCM: YCBCR")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_YCBCR, PREFIX=constants.NAME_YCBCR_LUMA, BAND=0)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_YCBCR_Y)

        #self.log.debug("GLCM: YCBCR CB")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_YCBCR, PREFIX=constants.NAME_YCBCR_BLUE_DIFFERENCE, BAND=1)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_YCBCR_CB)

        #self.log.debug("GLCM: YCBCR CR")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_YCBCR, PREFIX=constants.NAME_YCBCR_RED_DIFFERENCE, BAND=2)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_YCBCR_CR)

        #self.log.debug("GLCM: CIELAB L")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_CIELAB, PREFIX=constants.NAME_CIELAB_L, BAND=0)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_CIELAB_L)

        #self.log.debug("GLCM: CIELAB A")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_CIELAB, PREFIX=constants.NAME_CIELAB_A, BAND=1)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_CIELAB_A)

        #self.log.debug("GLCM: CIELAB B")
        self._performance.start()
        glcm = GLCM(self._blobs, constants.NAME_IMAGE_CIELAB, PREFIX=constants.NAME_CIELAB_B, BAND=2)
        glcm.computeAttributes()
        self._blobs = glcm.blobs
        self._performance.stopAndRecord(constants.PERF_GLCM_CIELAB_B)

    def computeShapeIndices(self):
        """
        Compute shape indices for all objects in image.
        The formula for this is given by Lin as e/4*sqrt(A)
        :return:
        """
        # Taken from:
        # Lin, F., D. Zhang, Y. Huang, X. Wang, and X. Chen. 2017.
        # “Detection of Corn and Weed Species by the Combination of Spectral, Shape and Textural Features.”
        # Sustainability (Switzerland) 9 (8). https://doi.org/10.3390/su9081335.

        for blobName, blobAttributes in self._blobs.items():
            #
            # This is the case of vegetation at the edge of the image. The ratio and the shape index
            # are not accurate, as there will be a long straight section that will throw off the calculations
            #
            (maxY, maxX, depth) = self._image.shape
            # The bounding rectangle of the blob
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            if (x == 0 or x + w >= maxX):
                shapeIndex = 0
            else:
                # The perimeter of the contour of the object
                perimeter = cv.arcLength(blobAttributes[constants.NAME_CONTOUR], True)
                shapeIndex = perimeter / (4 * math.sqrt(blobAttributes[constants.NAME_AREA]))

            blobAttributes[constants.NAME_SHAPE_INDEX] = shapeIndex
            self._shapeIndices.append(shapeIndex)
        return

    def computeDistancesToImageEdge(self, pixelSize: float, resolution: ()):
        """
        Compute the distance to the image edge for all items
        """
        # TODO: I forget why 3 values were expected here.
        # maxY, maxX, bands = resolution
        maxY, maxX = resolution
        for blobName, blobAttributes in self._blobs.items():
            # The location within the image
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]

            distanceToEdge = (maxX - x) * pixelSize
            blobAttributes[constants.NAME_DIST_TO_LEADING_EDGE] = distanceToEdge
        return

    @staticmethod
    def lengthWidthRatio(contour: np.ndarray) -> float:
        """
        Returns the length/width ratio given a contour.
        :param contour:
        :return: A float of the length/width ratio
        """
        # Taken from:
        # Lin, F., D. Zhang, Y. Huang, X. Wang, and X. Chen. 2017.
        # “Detection of Corn and Weed Species by the Combination of Spectral, Shape and Textural Features.”
        # Sustainability (Switzerland) 9 (8). https://doi.org/10.3390/su9081335.

        # The X values of the contour
        xCoordinates = contour[:, 0, 0]
        # The Y values of the contour
        yCoordinates = contour[:, 0, 1]

        varX = np.var(xCoordinates)
        varY = np.var(yCoordinates)
        covXY = np.cov(xCoordinates, yCoordinates)[0, 1]

        # The covariance matrix shown in Lin 2017 equation 2
        s = np.array([[varX, covXY], [covXY, varY]])
        # The eiginvalue computation
        w, v = np.linalg.eig(s)
        ratio = w.max() / w.min()
        return ratio

    def computeLengthWidthRatios(self):
        """
        Inserts the length width ratio for all blobs.
        :return:
        """
        for blobName, blobAttributes in self._blobs.items():
            #
            # This is the case of vegetation at the edge of the image. The ratio and the shape index
            # are not accurate, as there will be a long straight section that will throw off the calculations
            #
            (maxY, maxX, depth) = self._image.shape
            # The bounding rectangle of the blob
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            if (x == 0 or x + w >= maxX):
                lengthWidthRatio = 0
                blobAttributes[constants.NAME_RATIO] = 0
            else:
                contour = blobAttributes[constants.NAME_CONTOUR]
                blobAttributes[constants.NAME_RATIO] = self.lengthWidthRatio(contour)

        return

    def _computeBoundaryChain(self):
        """
        Compute the boundary chain for all blobs and store as a blob attribute
        """
        # Chain codes
        codes = [[3, 2, 1], [4, 99, 0], [5, 6, 7]]
        translations = {-1: 0, 0: 1, 1: 2}
        for blobName, blobAttributes in self._blobs.items():
            contour = blobAttributes[constants.NAME_CONTOUR]
            boundaryChain = []
            # Testing
            # contourPoints = [[[0, 0]], [[0, 1]], [[1, 2]], [[2, 2]], [[3, 1]], [[4, 0]], [[5, 0]], [[6, 1]], [[7, 2]]]
            # contour = np.array(contourPoints)
            for coordinate in range(len(contour) - 1):
                if contour[coordinate, 0, 0] > contour[coordinate + 1, 0, 0]:
                    #xDifference = contour[coordinate, 0, 0] - contour[coordinate + 1, 0, 0]
                    i = 0
                elif contour[coordinate, 0, 0] < contour[coordinate + 1, 0, 0]:
                    #xDifference = contour[coordinate + 1, 0, 0] - contour[coordinate, 0, 0]
                    i = 2
                else:
                    i = 1
                if contour[coordinate, 0, 1] < contour[coordinate + 1, 0, 1]:
                    #yDifference = contour[coordinate + 1, 0, 1] - contour[coordinate, 0, 1]
                    j = 0
                elif contour[coordinate, 0, 1] > contour[coordinate + 1, 0, 1]:
                    #yDifference = contour[coordinate + 1, 0, 1] - contour[coordinate, 0, 1]
                    j = 2
                else:
                    j = 1

                #print(f"({contour[coordinate, 0, 0]},{contour[coordinate, 0, 1]}) differences ({xDifference}, {yDifference}))")
                boundaryChain.append(codes[j][i])
            blobAttributes[constants.NAME_BOUNDARY_CHAIN] = boundaryChain
            #self.log.info(f"Boundary chain computed for perimeter")

    def computeKslope(self, k: int):
        """
        Compute the k-slope array of boundaries for blobs in the image.
        Stores the result in constants.NAME_KSLOPE attribute
        :param k: The k for calculation -- must be even
        """
        # Make certain the k value given is even
        assert(k % 2 == 0)

        for blobName, blobAttributes in self._blobs.items():
            contour = blobAttributes[constants.NAME_CONTOUR]
            # Testing
            # contourPoints = [[[0, 0]], [[0, 1]], [[1, 2]], [[2, 2]], [[3, 1]], [[4, 0]], [[5, 0]], [[6, 1]], [[7, 2]]]
            # contour = np.array(contourPoints)
            # The contour chain contains every point, but we want to base on points that are regularly spaced apart.
            # The exact number of samples depends on the length of the chain
            pointsForEveryDegree = len(contour) / 360
            spacing = pointsForEveryDegree / k
            slopes = []
            for coordinate in range(int(k/2) - 1, len(contour) - int(k/2) - 1, k):
                point1 = contour[coordinate + int(k/2), 0]
                point2 = contour[coordinate - int(k/2), 0]
                if point1[1] - point2[1] != 0:
                    try:
                        slope = math.atan((point1[0] - point2[0]) / (point1[1] - point2[1]))
                    except ZeroDivisionError:
                        # Horizontal slope is taken a 0
                        slope = 0
                else:
                    slope = 0

                slopes.append(slope)

            blobAttributes[constants.NAME_KSLOPE] = slopes
        self.log.debug(f"k-slope computed for perimeter")


    def computeKcurvature(self, k: int):
        """
        Compute the k-curvature array of boundaries for blobs in the image.
        Stores the result in constants.NAME_KCURVATURE attribute

        :param k:
        """
        for blobName, blobAttributes in self._blobs.items():
            contour = blobAttributes[constants.NAME_CONTOUR]
            # Testing
            # contourPoints = [[[0, 0]], [[0, 1]], [[1, 2]], [[2, 2]], [[3, 1]], [[4, 0]], [[5, 0]], [[6, 1]], [[7, 2]]]
            # contour = np.array(contourPoints)
            slopes = []
            curvature = []
            # Sample the shape as if it were a circle
            for coordinate in range(k - 1, len(contour) - k - 1, k):

                # A P P R O X I M A T I O N
                # Method 1 using approximation of the curve
                #
                pointOriginal = contour[coordinate, 0]
                point1 = contour[coordinate + k, 0]
                point2 = contour[coordinate - k, 0]
                if point2[1] - pointOriginal[1] != 0:
                    try:
                        slope1 = math.atan((point2[0] - pointOriginal[0]) / (point2[1] - pointOriginal[1]))
                    except ZeroDivisionError:
                        # Horizontal slope is taken a 0
                        slope1 = 0
                else:
                    slope1 = 0

                if pointOriginal[1] - point1[1] != 0:
                    try:
                        slope2 = math.atan((pointOriginal[0] - point1[0]) / (pointOriginal[1] - point1[1]))
                    except ZeroDivisionError:
                        # Horizontal slope is taken a 0
                        slope2 = 0
                else:
                    slope2 = 0

                slopes.append(slope1 - slope2)

            # C A L C U L A T I O N
            #
            # Method 2 based algorithm from this text @ 10.4.2
            # https://www.sciencedirect.com/book/9781558608610/digital-geometry
            # Klette, R., & Rosenfeld, A.(2004).
            # Digital Geometry: Geometric Methods for Digital Picture Analysis (1st ed.)[675].
            # Elsevier Science & Technology.

            stride = k
            # Adapted from https://stackoverflow.com/questions/32629806/how-can-i-calculate-the-curvature-of-an-extracted-contour-by-opencv
            assert stride < len(contour), "stride must be shorter than length of contour"

            #self.log.debug(f"Compute k-curvature for contour of length {len(contour)}")
            for i in range(len(contour)):
                before = i - stride + len(contour) if i - stride < 0 else i - stride
                after = i + stride - len(contour) if i + stride >= len(contour) else i + stride

                f1x, f1y = (contour[after, 0] - contour[before, 0]) / stride
                f2x, f2y = (contour[after, 0] - 2 * contour[i, 0] + contour[before, 0]) / stride ** 2
                denominator = (f1x ** 2 + f1y ** 2) ** 3 + 1e-11

                denominator += 1e-12
                curvature_at_i = np.sqrt(4 * (f2y * f1x - f2x * f1y) ** 2 / denominator)

                curvature.append(curvature_at_i)


            blobAttributes[constants.NAME_KCURVATURE] = curvature
            blobAttributes[constants.NAME_KCURVATURE_APPROXIMATE] = slopes
            self._k = k
        self.log.debug(f"k-curvature computed for perimeter")

    def visualize(self, attribute: str):

        if attribute == constants.NAME_KCURVATURE_APPROXIMATE or attribute == constants.NAME_KCURVATURE:
            for blobName, blobAttributes in self._blobs.items():
                slopes = blobAttributes[attribute]
                plt.figure(figsize=(10, 10))
                ax = plt.subplot(111, polar=True)  # Create subplot
                plt.grid(color='#888888')  # Color the grid
                ax.set_theta_zero_location('N')  # Set zero to North
                # ax.set_theta_direction(-1)  # Reverse the rotation
                # ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], color='#666666',
                #                    fontsize=8)  # Customize the xtick labels
                ax.spines['polar'].set_visible(False)  # Show or hide the plot spine
                # ax.set_axis_bgcolor('#111111')

                # plt.xlabel(f"kappa = {self._k}")
                # plt.ylabel(f"k approximation")
                # plt.hlines(0, 0, len(slopes), colors="red")
                plt.title(f"{blobName} {attribute}")
                # plt.plot(slopes)


                theta = np.deg2rad(np.arange(0, 360, 360/len(slopes)))

                # plotting the polar coordinates on the system
                plt.polar(theta, slopes, marker='o')
                plt.show()

    def computeBendingEnergy(self):
        """
        Compute the bending energy for all blobs and store as a blob attribute.
        Bending Energy is stored in constants.NAME_BENDING attribute
        """
        self._computeBoundaryChain()
        self.computeKslope(6)
        self.computeKcurvature(6)
        # self.visualize(constants.NAME_KCURVATURE)
        for blobName, blobAttributes in self._blobs.items():
            slopes = np.array(blobAttributes[constants.NAME_KCURVATURE])
            bending = np.sum(slopes**2) / len(slopes)
            totalCurvature = np.sum(abs(slopes)) / len(slopes)
            blobAttributes[constants.NAME_BENDING] = bending
            blobAttributes[constants.NAME_ABSCURVATURE] = totalCurvature
            #self.log.debug(f"Bending energy for {blobName}: {bending}")
            #self.log.debug(f"Total absolute curvature for {blobName}: {totalCurvature}")


    def computeRadialDistances(self):

        for blobName, blobAttributes in self._blobs.items():
            contour = blobAttributes[constants.NAME_CONTOUR]
            #avgX = np.average(contour[:, 0, 1])
            #avgY = np.average(contour[:, 0, 0])
            centroid = blobAttributes[constants.NAME_CENTER]

            distances = []
            maxDistance = 0
            for coordinate in range(len(contour)):
                point = contour[coordinate, 0]
                distance = math.sqrt((point[1] - centroid[1])**2 + (point[0] - centroid[0])**2)
                distances.append(distance)
                if distance > maxDistance:
                    maxDistance = distance

            distancesDF = np.array(distances)
            blobAttributes[constants.NAME_RADIAL_DISTANCE] = distancesDF
            normRadialDistance = distancesDF / maxDistance
            blobAttributes[constants.NAME_RADIAL_NORM] = normRadialDistance
            normRadialAverage = np.average(normRadialDistance)
            blobAttributes[constants.NAME_RADIAL_AVG] = normRadialAverage
            blobAttributes[constants.NAME_RADIAL_VAR] = np.var(blobAttributes[constants.NAME_RADIAL_NORM])
            crossings = np.where(normRadialDistance > normRadialAverage)
            # Get the percentage  of crossings of the mean from the tuple returned
            percentCrossings = len(crossings[0]) / len(normRadialDistance)
            blobAttributes[constants.NAME_RADIAL_CROSSINGS_PCT] = percentCrossings
            #self.log.debug(f"Radial distances (sample): {distances[0]} {distances[1]}")
            #self.log.debug(f"Radial variance: {blobAttributes[constants.NAME_RADIAL_VAR]}")
            #self.log.debug(f"Radial crossings percentage: {percentCrossings}")


    def computeFourier(self):

        # We need a rectangular shape for these values, so use the bounding
        # box for each blob.  This has a problem with vegetation that appears in the image,
        # so this might get a bit messy.
        for blobName, blobAttributes in self._blobs.items():
            pass
    def computeCompactness(self):
        """
        The compactness of an object is given by 4 * pi * area / perimeter^2
        Adds the compactness to the blobAttributes as NAME_COMPACTNESS

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        """
        for blobName, blobAttributes in self._blobs.items():
            # The permimeter of the contour
            # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
            perimeter = cv.arcLength(blobAttributes[constants.NAME_CONTOUR], True)
            compactness = (4 * math.pi * blobAttributes[constants.NAME_AREA]) / perimeter ** 2
            blobAttributes[constants.NAME_COMPACTNESS] = compactness

    def computeElogation(self):
        """
        The elongation of an object is given by width/length of the bounding box.
        Adds the elongation to the blobAttributes as NAME_ELONGATION

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        """
        for blobName, blobAttributes in self._blobs.items():
            # The bounding rectangle of the blob
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            elongation = w / h
            blobAttributes[constants.NAME_ELONGATION] = elongation

    @staticmethod
    def _findMajorMinorAxis(contour) -> (float, float):
        """
        Find the major and minor axis of the contour
        :param contour: The contour of the blob

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        :return: (MajorAxis, MinorAxis)
        """
        (x, y), (majorAxis, minorAxis), angle = cv.fitEllipse(contour)
        return (majorAxis, minorAxis)

    def computeEccentricity(self):
        """
        The eccentricity of an object is the ratio of the length of the minor axis to the length of the major
        axis. Adds the eccentricity to the blobAttributes as NAME_ECCENTRICITY

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        """
        for blobName, blobAttributes in self._blobs.items():
            # The major and minor axes
            (majorAxis, minorAxis) = ImageManipulation._findMajorMinorAxis(blobAttributes[constants.NAME_CONTOUR])
            try:
                eccentricity = minorAxis / majorAxis
                blobAttributes[constants.NAME_ECCENTRICITY] = eccentricity
            except ZeroDivisionError:
                self.log.error("Division by zero in computeEccentricity")
                blobAttributes[constants.NAME_ECCENTRICITY] = 0

    def computeMiscShapeMetrics(self):
        for blobName, blobAttributes in self._blobs.items():
            # The convex hull and its perimeter
            hull = cv.convexHull(blobAttributes[constants.NAME_CONTOUR])
            blobAttributes[constants.NAME_CONVEX_HULL] = hull

    def computeRoundness(self):
        """
        The roundness of an object is the ratio of the area of an object to the area of a circle with the same
        convex perimeter. Adds the roundness to the blobAttributes as NAME_ROUNDNESS

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        """
        for blobName, blobAttributes in self._blobs.items():
            area = blobAttributes[constants.NAME_AREA]
            # The convex hull and its perimeter
            hull = cv.convexHull(blobAttributes[constants.NAME_CONTOUR])
            perimeter = cv.arcLength(hull, True)
            roundness = (4 * math.pi * blobAttributes[constants.NAME_AREA]) / perimeter ** 2
            blobAttributes[constants.NAME_ROUNDNESS] = roundness

    def computeConvexity(self):
        """
        The convexity of an object is the relative amount an object differs from a convex object
        Adds the convexity to the blobAttributes as NAME_CONVEXITY

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        """
        for blobName, blobAttributes in self._blobs.items():
            # TODO: Already computed.  Consider doing this just once instead.
            perimeter = cv.arcLength(blobAttributes[constants.NAME_CONTOUR], True)
            # The convex hull and its perimeter
            hull = cv.convexHull(blobAttributes[constants.NAME_CONTOUR])
            hullPerimeter = cv.arcLength(hull, True)
            convexity = hullPerimeter / perimeter
            blobAttributes[constants.NAME_CONVEXITY] = convexity

    def computeSolidity(self):
        """
        The solidity of an object os the ratio of the contour area to the convex hull area.
        Adds the solidity to the blobAttributes as NAME_SOLIDITY

        See: http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf
        See: https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html
        """
        for blobName, blobAttributes in self._blobs.items():
            # Already computed, but we could also get this from opencv with this call:
            # area = cv.contourArea(blobAttributes[NAME_CONTOUR])
            area = blobAttributes[constants.NAME_AREA]
            # The convex hull and its area
            hull = cv.convexHull(blobAttributes[constants.NAME_CONTOUR])
            hullArea = cv.contourArea(hull)
            solidity = area / hullArea
            blobAttributes[constants.NAME_SOLIDITY] = solidity

    def identifyCloseVegetation(self):
        return

    def identifyCropRowCandidates(self):
        """
        Create a list of the vegetation likely to be in the crop row.
        For now, use a very simple approach -- if it is roughly in the center of the image
        it is likely part of the crop
        """
        (height, width, depth) = self.image.shape

        # This is approximately the midline of the image
        middleY = int(height / 2)
        for blobName, blobAttributes in self._blobs.items():
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            (x, y) = blobAttributes[constants.NAME_CENTER]
            # Items near the midline are more likely to be crop
            # ?? Perhaps we should also consider equal spacing?
            if (y > middleY - MIDDLE_THRESHOLD and y < middleY + MIDDLE_THRESHOLD):
                self._cropRowCandidates[blobName] = blobAttributes

    def substituteRectanglesForVegetation(self):
        """
        Using the identified centers of where the vegetation, draw low height rectangles
        that are later used for crop line detection
        """
        self.log.debug(f"Substitute rectangles for image with {len(self._blobs)} blobs")
        self.cropline_image = np.zeros(self._image.shape, np.uint8)
        #for blobName, blobAttributes in self._cropRowCandidates.items():
        for blobName, blobAttributes in self._blobs.items():
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            (x, y) = blobAttributes[constants.NAME_CENTER]
            # cv.circle(self.blank_image,(x,y),1, (255,255,255), -1)
            #self.cropline_image = cv.rectangle(self.cropline_image, (x, y), (x + 100, y + 30), (255, 255, 255), 2)
            self.cropline_image = cv.rectangle(self.cropline_image, (x, y), (x + 10, y + 10), (255, 255, 255), -1)

        # filename = "candidates-" + str(uuid.uuid4()) + ".jpg"

        # cv.imwrite(filename, self.cropline_image)

    def detectLines(self):
        """
        Detect croplines in the current image, Use probabilistic Hough transform
        The resulting lines are in the lines and linesP variables
        """
        dst = cv.Canny(self.cropline_image, 50, 200, None, 3)
        cv.imwrite("edges.jpg", dst)
        # self.linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, minLineLength=100, maxLineGap=1000)
        # This allows for some plants in the crop line to be slightly offset from other plants
        # in the same row
        # What is needed is to detect roughly horizontal lines
        rho = 1
        theta = np.pi / 180
        threshold = 30
        self.linesP = cv.HoughLinesP(dst, rho, theta, threshold, None, minLineLength=75, maxLineGap=1500)
        self.log.debug(f"Detected {len(self.linesP)} croplines")

        self.lines = cv.HoughLines(dst, 50, np.pi / 2, 200)

        if self.linesP is not None:
            for i in range(0, len(self.linesP)):
                l = self.linesP[i][0]
                cv.line(self.cropline_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        #cv.imwrite("crop-lines.jpg", self.cropline_image)

    def angleOf(self, p1: (), p2: ()) -> float:
        """
        Calculate the angles between all elements. Results are in _angles
        """
        (p1x1, p1y1) = p1
        (p2x2, p2y2) = p2
        deltaY = p1y1 - p2y2
        deltaX = p1x1 - p2x2
        angle = math.atan2(deltaY, deltaX)
        degrees = math.degrees(angle)
        if degrees < 0:
            final_degrees = 180 + degrees
        else:
            final_degrees = 180 - degrees
        if p1x1 > p2x2:
            final_degrees = 180 - final_degrees

        return final_degrees

    def normalizedDistanceToCropY(self, y: int) -> float:
        """
        The normalized distance from y to the crop line y
        :param y:
        :return: The distance as a float
        """
        if y >= self._cropY:
            distanceFromY = y - self._cropY
        else:
            distanceFromY = self._cropY - y
        normalizedDistance = distanceFromY / self._maxY
        return normalizedDistance

    def findCropLine(self):
        """
        Find the likely crop line in the image by finding the biggest items in the image.\n
        WARNING: This is abandoned code that does nothing -- only here because the attributes are used elsewhere

        :return:
        """
        # Find the number of horizontal neighbors each element has.

        # By default, the crop line is in the middle of the image -- likely for images taken by hand, not so much
        # by drone
        likelyCropLineY = int(self._maxY / 2)
        weightedDistanceMax = 0

        # This is the part that is not needed anymore
        # Find the biggest item closest to the center line
        # try:
        #     for blobName, blobAttributes in self._blobs.items():
        #         weightedDistance = blobAttributes[constants.NAME_AREA] * (
        #                     1 - blobAttributes[constants.NAME_DISTANCE_NORMALIZED])
        #
        #         self.log.debug("Weighted distance of {}: {}".format(blobName, weightedDistance))
        #         if weightedDistance > weightedDistanceMax:
        #             weightedDistanceMax = weightedDistance
        #             likelyCropLineBlob = blobName
        #             likelyCropLineY = blobAttributes[constants.NAME_CENTER][1]
        # except KeyError as key:
        #     self.log.error(f"Attribute not found: {key}")
        #     likelyCropLineY = 0
        #     likelyCropLineBlob = "error"

        self._cropY = likelyCropLineY
        # self.log.debug("Crop line Y: {} for blob {}".format(self._cropY, likelyCropLineBlob))

        # Step through and replace the normalized distance to the center line
        # with the normalized distance to the crop line
        # This is the bit that sets the attributes that are needed
        for blobName, blobAttributes in self._blobs.items():
            (x, y) = blobAttributes[constants.NAME_CENTER]
            blobAttributes[constants.NAME_DISTANCE_NORMALIZED] = self.normalizedDistanceToCropY(y)

            if likelyCropLineY == 0:
                blobAttributes[constants.NAME_DISTANCE] = 0

        return self._cropY

    def findAngles(self):
        """
        Calculate three things:
        - the angles from every center to every center
        - the Y of the crop line, stored in _cropY
        - the distance from the crop line for each center
        For some machine learning, it's best if things are on the same scale, so convert the distance to a
        normalized value as well.
        :return:
        """
        self._centersUnsorted = []
        self._centers = []

        centerCount = 0
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_TYPE] != constants.TYPE_UNDESIRED and blobAttributes[
                constants.NAME_TYPE] != constants.TYPE_IGNORED:
                centerCount = centerCount + 1
                # print(blobName + ": " + str(blobAttributes[constants.NAME_CENTER]))
                # Keep the name with the coordinates so we know which blob this refers to
                self._centersUnsorted.append(blobAttributes[constants.NAME_CENTER] + tuple([blobName]))
        # print(self._centersUnsorted)

        # Create an array to hold the angles between the blobs
        self._angles = np.zeros((centerCount, centerCount))

        # Sort the centers by the X value -- the first
        # self._centers = sorted(self._centersUnsorted, key=lambda x: x[0])
        # print(self._centers)

        self._centers = self._centersUnsorted

        # There may have only one crop in the image
        if len(self._centers) > 1:
            for i in range(len(self._centers)):
                for j in range(len(self._centers)):
                    point1 = self._centers[i]
                    point2 = self._centers[j]
                    angle = self.angleOf((point1[0], point1[1]), (point2[0], point2[1]))
                    self._angles[i, j] = angle
                    # print(str(point1) + " to " + str(point2) + " angle " + str(angle))
                    # cv.line(self._image, (point1[0], point1[1]), (point2[0], point2[1]), (0,255,0), 3, cv.LINE_AA)
        else:
            (x, y, name) = self._centers[0]
            self._cropY = y
            return

        # Change 180 to zero -- this is the case when we compute from/to the same point
        self._angles = np.where(self._angles == 180, 0, self._angles)
        # Threshold the values so we can tell roughly what is in a line
        self._angles = np.where(self._angles > 5, np.nan, self._angles)

        # TODO: This logic is a mess. Clean up
        # Create a dataframe from the angles computed
        self._df = pd.DataFrame(data=self._angles)

        # Find the entry with the smallest number of NaNs
        sums = self._df.isnull().sum().nsmallest(5)
        # print(sums)

        smallestDistanceFromY = 10000
        smallestIndex = 10000

        (y, centerX, depth) = self._image.shape
        centerY = int(y / 2)

        for index, row in sums.iteritems():

            (x, y, blobName) = self._centers[index]
            distanceFromY = abs(centerY - y)
            # print("Distance from Y: " + str(distanceFromY) + " smallestY: " + str(smallestDistanceFromY))
            if (distanceFromY < smallestDistanceFromY):
                smallestIndex = index
                smallestDistanceFromY = distanceFromY

        (cropX, cropY, blobName) = self._centers[smallestIndex]

        # This is all we need. The Y location of the crop line in the image
        # self._cropY = cropY
        # Treat the centerline of the image as the potential crop line
        self._cropY = self._centerLineY

        # Add the distance from the crop line for all the blobs.
        for blobName, blobAttributes in self._blobs.items():
            (x, y) = blobAttributes[constants.NAME_CENTER]
            if y >= self._cropY:
                distanceFromY = y - self._cropY
            else:
                distanceFromY = self._cropY - y

            blobAttributes[constants.NAME_DISTANCE] = distanceFromY
            blobAttributes[constants.NAME_DISTANCE_NORMALIZED] = distanceFromY / self._maxY
        return

    def drawCropline(self):
        """
        Draw a cropline on the current image if one has been found and a centerline for reference.
        """
        (height, width, depth) = self.image.shape
        cv.line(self._image, (0, int(height / 2)), (width, int(height / 2)), (0, 127, 127), 3, cv.LINE_AA)
        cv.putText(self._image,
                   "Center Line",
                   (int(width / 2), int(height / 2) + 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.75,
                   (0, 127, 127),
                   2)

        cv.line(self._image, (0, self._cropY), (width, self._cropY), (255, 255, 255), 3, cv.LINE_AA)
        cv.putText(self._image,
                   "Crop Line",
                   (int(width / 2) + 200, self._cropY + 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.75,
                   (255, 255, 255),
                   2)

        # if self.linesP is not None:
        #     for i in range(0, len(self.linesP)):
        #         l = self.linesP[i][0]
        #         cv.line(self._image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        #         cv.putText(self._image,
        #                    str(self.mmBetweenPoints((l[0],l[1]), (l[2], l[3]), self._mmPerPixel)) + " mm",
        #                    (l[0] + 25, l[1] + 25),
        #                    cv.FONT_HERSHEY_SIMPLEX,
        #                    0.75,
        #                    (255,255,255),
        #                    2)

    def drawContours(self):
        """
        Draw the contours on the image
        """
        for blobName, blobAttributes in self._blobs.items():
            contour = blobAttributes[constants.NAME_CONTOUR]
            # Default color was (255. 0, 0), but that interfered with the blue bucket lid, so switch to red.
            cv.drawContours(self._image, contour, contourIdx=-1, color=(0, 0, 255),
                            thickness=constants.SIZE_CONTOUR_LINE)
        # self._contours_image = np.zeros(self._image.shape, np.uint8)
        # cv.drawContours(self._contours_image, self._contours, contourIdx=-1, color=(255,0,0),thickness=2)
        # cv.imwrite("contours.jpg", self._contours_image)

    def drawDistances(self):
        """
        Draw the distances for weeds to the edge of the image
        """
        self.log.debug("Draw distances for weeds. Not yet implemented")

    def drawBoxes(self, name: str, rectangles: [], decorations: [], convexHull=False):
        """
        Draw colored boxes around the blobs based on what type they are
        :param name: The name of the image
        :param rectangles: A list of rectangles surrounding the blobs
        :param decorations:
        :param convexHull: a convex hull or bounding bix should surround
        """

        cv.putText(self._image, name, (50, 75), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)

        # Convex hull
        if convexHull:
            #cv.drawContours(self._image, self._contours, -1, (255, 0, 0), 5, 8)
            # create hull array for convex hull points
            hull = []

            # calculate points for each contour
            for i in range(len(self._contours)):
                # creating convex hull object for each contour
                hull.append(cv.convexHull(self._contours[i], False))
            for i in range(len(self._contours)):
                color_contours = (0, 255, 0)  # green - color for contours
                color = (255, 0, 0)  # blue - color for convex hull
                # draw ith contour
                # cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
                # draw ith convex hull object
                cv.drawContours(self._image, hull, i, COLOR_CROP, constants.SIZE_CONTOUR_LINE, 8)

        for rectName, rectAttributes in rectangles.items():
            (x, y, w, h) = rectAttributes[constants.NAME_LOCATION]
            (cX, cY) = rectAttributes[constants.NAME_CENTER]
            area = rectAttributes[constants.NAME_AREA]
            type = rectAttributes[constants.NAME_TYPE]
            if type == constants.TYPE_UNKNOWN:
                color = COLOR_UNKNOWN
            elif type == constants.TYPE_UNDESIRED:
                color = COLOR_WEED
            elif type == constants.TYPE_UNTREATED:
                color = COLOR_UNTREATED
            elif type == constants.TYPE_IGNORED:
                color = COLOR_IGNORED
            else:
                color = COLOR_CROP

            # Not drawing the ignored type yields a cleaner image in the test set

            if type != constants.TYPE_IGNORED:
                # Draw a bounding rectangle around the blob
                if not convexHull:
                    self._image = cv.rectangle(self._image, (x, y), (x + w, y + h), color, BOUNDING_BOX_THICKNESS)
                # draw a convex hull around the blob
                else:
                    _contours, _hierarchy = cv.findContours(self.threshold, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


                cv.circle(self._image, (cX, cY), 5, (255, 255, 255), -1)
                location = "Location: (" + str(cX) + "," + str(cY) + ")"
                areaText = "Area: " + str(area)
                nameText = "Name: {}".format(rectName)


                # Determine the starting location of the text.
                # Text near the edges of the image will get cut off
                # The Y offset of each decoration
                yOffset = 25
                xOffset = 25
                decorationHeight = 45
                decorationWidth = 500
                # There is one decoration that does not take up space, contours
                if constants.NAME_CONTOUR in decorations:
                    decorationsTotalHeight = (len(decorations) - 1) * yOffset
                else:
                    decorationsTotalHeight = len(decorations) * yOffset

                # Reversed these definitions
                maxWidth = self._maxY
                maxHeight = self._maxX

                # Adjust the starting point if the text is too wide
                self.log.debug(f"cX: {cX} cY: {cY}")
                if cX - xOffset > maxWidth - decorationWidth:
                    cX = cX - 500
                # Adjust the starting point if all the decorations are too tall
                if cY - decorationsTotalHeight < yOffset:
                    cY += decorationsTotalHeight + (2 * yOffset)


                # track each decoration
                countOfDecorations = 0
                if constants.NAME_LOCATION in decorations:
                    countOfDecorations += 1
                    cv.putText(self._image, location, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_AREA in decorations:
                    countOfDecorations += 1
                    cv.putText(self._image, areaText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_BOX_DIMENSIONS in decorations:
                    dimensionText = f"Box Dimensions:  {w}w x {h}h"
                    countOfDecorations += 1
                    cv.putText(self._image, dimensionText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_SHAPE_INDEX in decorations:
                    shapeText = "Shape: " + "{:.4f}".format(rectAttributes[constants.NAME_SHAPE_INDEX])
                    countOfDecorations += 1
                    cv.putText(self._image, shapeText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_RATIO in decorations:
                    lengthWidthRatioText = "L/W Ratio: " + "{:4f}".format(rectAttributes[constants.NAME_RATIO])
                    countOfDecorations += 1
                    cv.putText(self._image, lengthWidthRatioText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_REASON in decorations:
                    reasonText = "Classified By: " + constants.REASONS[rectAttributes[constants.NAME_REASON]]
                    countOfDecorations += 1
                    cv.putText(self._image, reasonText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_TYPE in decorations:
                    classifiedText = "Classified As: " + constants.TYPES[rectAttributes[constants.NAME_TYPE]]
                    countOfDecorations += 1
                    cv.putText(self._image, classifiedText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_NAME in decorations:
                    countOfDecorations += 1
                    cv.putText(self._image, nameText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_DISTANCE_NORMALIZED in decorations:
                    distanceText = "Normalized Distance: " + "{:.4f}".format( rectAttributes[constants.NAME_DISTANCE_NORMALIZED])
                    countOfDecorations += 1
                    cv.putText(self._image, distanceText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                    cv.line(self._image, (cX, cY), (cX, self._cropY), (255, 255, 255), 3)
                if constants.NAME_HUE in decorations:
                    hueText = "Hue: {:.4f}".format(rectAttributes[constants.NAME_HUE])
                    countOfDecorations += 1
                    cv.putText(self._image, hueText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_I_YIQ in decorations:
                    yiqText = "In Phase: {:.4f}".format(rectAttributes[constants.NAME_I_YIQ])
                    countOfDecorations += 1
                    cv.putText(self._image, yiqText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_ELONGATION in decorations:
                    elongationText = "Elongation: {:.4f}".format(rectAttributes[constants.NAME_ELONGATION])
                    countOfDecorations += 1
                    cv.putText(self._image, elongationText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_ECCENTRICITY in decorations:
                    eccentricityText = "Eccentricity: {:.4f}".format(rectAttributes[constants.NAME_ECCENTRICITY])
                    countOfDecorations += 1
                    cv.putText(self._image, eccentricityText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_ROUNDNESS in decorations:
                    roundnessText = "Roundness: {:.4f}".format(rectAttributes[constants.NAME_ROUNDNESS])
                    countOfDecorations += 1
                    cv.putText(self._image, roundnessText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_CONVEXITY in decorations:
                    convexityText = "Convexity: {:.4f}".format(rectAttributes[constants.NAME_CONVEXITY])
                    countOfDecorations += 1
                    cv.putText(self._image, convexityText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_SOLIDITY in decorations:
                    solidityText = "Solidity: {:.4f}".format(rectAttributes[constants.NAME_SOLIDITY])
                    countOfDecorations += 1
                    cv.putText(self._image, solidityText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_DIST_TO_LEADING_EDGE in decorations:
                    distanceToEmitterText = "Distance to Emitter: {:.4f}".format(rectAttributes[constants.NAME_DIST_TO_LEADING_EDGE])
                    countOfDecorations += 1
                    cv.putText(self._image, distanceToEmitterText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)
                if constants.NAME_BENDING in decorations:
                    distanceToEmitterText = "Bending Energy: {:.4f}".format(rectAttributes[constants.NAME_BENDING])
                    countOfDecorations += 1
                    cv.putText(self._image, distanceToEmitterText, (cX - 25, cY - (countOfDecorations * yOffset)), cv.FONT_HERSHEY_SIMPLEX, self._fontScale, (255, 255, 255), 2)

        # cv.imwrite("opencv-centers.jpg", self._image)
        # self.show("centers", self._image)
        # cv.waitKey()

    def stitchTo(self, previous):
        # Stitch the current image to the previous one
        status, pano = self._stitcher.stitch([previous.astype(np.uint8), self._image.astype(np.uint8)])
        return pano
        # cv.imwrite("stitched.jpg", pano)

    def drawBoundingBoxes(self, contours: []):
        for c in contours:
            # calculate moments for each contour
            M = cv.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            # calculate x,y coordinate of center
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            cv.circle(self._image, (cX, cY), 5, (255, 255, 255), -1)
            cv.putText(self._image, "centroid", (cX - 25, cY - 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Do not consider rotation -- just bound the object
            x, y, w, h = cv.boundingRect(c)
            self._image = cv.rectangle(self._image, (x, y), (x + w, y + h), COLOR_CROP, 2)

            # Minimum area bounding
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(self._image, [box], 0, COLOR_UNKNOWN, BOUNDING_BOX_THICKNESS)

            # Append the current rectangle
            self._rectangles.append(rect)

            # cv.imwrite("centers.jpg", cv.cvtColor(self._image,))
        # self.show("centers", self._image)
        # cv.waitKey()
        return

    # This does not do quite what is need.  If some unwanted vegetation is inside the bounding box, it is
    # present in the extracted image.
    # TODO: Write a routine to eliminate a class of object within the image.
    def extractImages(self):
        for blobName, blobAttributes in self._blobs.items():
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            # Pull the subset from the original image so we don't see the markings
            image = self._original[y:y + h, x:x + w]
            blobAttributes[constants.NAME_IMAGE] = image
            blobAttributes[constants.NAME_IMAGE_RGB] = image

            image = self._imgAsGreyscale[y:y + h, x:x + w]
            blobAttributes[constants.NAME_GREYSCALE_IMAGE] = image

            image = self._imageAsYIQ[y:y + h, x:x + w]
            blobAttributes[constants.NAME_IMAGE_YIQ] = image

            image = self._imgAsHSV[y:y + h, x:x + w]
            blobAttributes[constants.NAME_IMAGE_HSV] = image

            image = self._imgAsHSI[y:y + h, x:x + w]
            blobAttributes[constants.NAME_IMAGE_HSI] = image

            image = self._imgAsYCBCR[y:y + h, x:x + w]
            blobAttributes[constants.NAME_IMAGE_YCBCR] = image

            image = self._imgAsCIELAB[y:y + h, x:x + w]
            blobAttributes[constants.NAME_IMAGE_CIELAB] = image


    def extractImagesFrom(self, source: np.ndarray, zslice: int, attribute: str, manipulation):
        """
        Extract the data from the given source for every object that isn't to be ignored.
        An example of this is pulling out the HUE layer from the HSV array.

        :param source: The source of the extraction (HSI, HSV, RGB, etc.)

        :param zslice: Which layer to extract -- TODO: make this work for things like greyscale as well

        :param attribute: The name of the attribute to use to store the result: i.e., NAME_HUE

        :param manipulation: The manipulation to apply

        """
        for blobName, blobAttributes in self._blobs.items():

            # For everything that isn't ignored, extract out the slice
            if blobAttributes[constants.NAME_TYPE] != constants.TYPE_IGNORED:
                (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
                # Pull the subset from the original image so we don't see the markings
                image = source[y:y + h, x:x + w, zslice]
                # This seems to be a very specific case.  After we have masked from the vegetation index
                # The black values are 0,0,0.  Convert them to NaN so when we perform calculations, we don't
                # use the black pixels
                image = np.where(image == 0, np.nan, image)
                if np.isnan(image).all():
                    self.log.error("All values for attribute are NaN: " + attribute)
                # hueMean = np.nanmean(image)
                hueMean = manipulation(image)
                #self.log.debug(attribute + ": " + str(hueMean))
                blobAttributes[attribute] = hueMean

    def normalize(self) -> bool:
        """
        Normalize all features in the range of 0..1.
        :return: Result of normalization as boolean
        """
        result = False
        for blobName, blobAttributes in self._blobs.items():
            # For everything that isn't ignored, extract out the slice
            if blobAttributes[constants.NAME_TYPE] != constants.TYPE_IGNORED:
                # Placeholder for now
                pass
        return result

    def _compactness(self, contour) -> float:
        return 1

    def extractAttributes(self):
        raise NotImplementedError

    @staticmethod
    def _area(size):
        return size[0] * size[1]

    def drawSquares(self, size: int):
        """
        Draw squares on the blobs
        """
        color = COLOR_UNKNOWN
        for blobName, blobAttributes in self._blobs.items():
            for (x, y) in blobAttributes[constants.NAME_SQUARES]:
                self._image = cv.rectangle(self._image, (x, y), (x + size, y + size), color, 1)

    # This is a computationally very expensive routine.
    def fitSquares(self, size: int):
        """
        Fits squares of specified size onto the blobs within an image
        :param size: size of the square in pixels
        """
        color = COLOR_UNKNOWN
        size -= 1
        for blobName, blobAttributes in self._blobs.items():
            squares = []
            contour = blobAttributes[constants.NAME_CONTOUR]
            # Get the attributes of the box bounding the contour
            boundingX, boundingY, boundingWidth, boundingHeight = blobAttributes[constants.NAME_LOCATION]
            self.log.debug(f"Fitting squares in contour for blob: {blobName} Bounding box: {boundingHeight} high X {boundingWidth} wide")
            occupied = np.ndarray((boundingHeight, boundingWidth), bool)
            occupied.fill(False)
            x = boundingX
            y = boundingY
            while x < (boundingX + boundingWidth):
                while y < (boundingY + boundingHeight):
                    try:
                        isUpperLeftOccupied = occupied[y - boundingY, x - boundingX]
                        isUpperRightOccupied = occupied[y - boundingY, x - boundingX + size]
                        isLowerLeftOccupied = occupied[y - boundingY + size, x - boundingX]
                        isLowerRightOccupied = occupied[y - boundingY + size, x - boundingX + size]
                    except IndexError:
                        #y += size
                        isUpperRightOccupied = True
                        isUpperLeftOccupied = True
                        isLowerRightOccupied = True
                        isLowerLeftOccupied = True

                    if not (isUpperLeftOccupied or isUpperRightOccupied or isLowerLeftOccupied or isLowerRightOccupied):
                        # Test if the point is within or on the contour. Unnecessary computations here
                        isUpperLeftInsideContour = cv.pointPolygonTest(contour, (x + size, y), False) >= 0
                        isUpperRightInsideContour = cv.pointPolygonTest(contour, (x + size, y + size), False) >= 0
                        isLowerLeftInsideContour = cv.pointPolygonTest(contour, (x, y), False) >= 0
                        isLowerRightInsideContour = cv.pointPolygonTest(contour, (x + size, y), False) >= 0
                        if isLowerRightInsideContour and isLowerLeftInsideContour and isUpperRightInsideContour and isUpperLeftInsideContour:
                            #self.log.debug(f"Square of size {size + 1} is contained at ({x},{y})")
                            #self._image = cv.rectangle(self._image, (x, y), (x + size, y + size), color, BOUNDING_BOX_THICKNESS)
                            squares.append((x, y))
                            #self._image = cv.rectangle(self._image, (x, y), (x + size, y + size), color, 1)
                            #self.log.debug(f"Marking occupied[{y - boundingY}:{(y - boundingY) + size},{x - boundingX}:{x - boundingX + size}]")
                            occupied[y - boundingY: (y - boundingY) + size, x - boundingX:x - boundingX + size] = True
                            y += size
                        else:
                            y += 1
                    else:
                        #self.log.debug(f"Position ({x},{y}) is occupied")
                        y += size
                x += 1
                y = boundingY
                self.log.debug(f"Next column: {x}")
            blobAttributes[constants.NAME_SQUARES] = squares
