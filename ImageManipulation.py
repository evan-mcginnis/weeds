#
# I M A G E
#
# Image manipulation
import uuid

from PIL import Image
from skimage import color
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
import pandas as pd
import logging
from math import pi

from collections import namedtuple
from operator import mul

from skimage.color import rgb2yiq

import constants

# Colors for the bounding boxes
COLOR_WEED = (0,0,255)
COLOR_CROP = (0,255,0)
COLOR_UNKNOWN = (255,0,0)
COLOR_UNTREATED = (0,127,0)
COLOR_IGNORED = (255,255,255)

# How far outside the midline of the image vegetation should be considered the cropline
MIDDLE_THRESHOLD = 200

# The lines for the enclosing rectangle
BOUNDING_BOX_THICKNESS = 2

class ImageManipulation:
    def __init__(self, img : np.ndarray, sequenceNumber : int):
        self._image = img
        self._name = constants.NAME_IMAGE + "-" + str(sequenceNumber)
        self._rectangles = []
        self._largestName = ""
        self._largestArea = 0
        self._blobs = {}
        self._cropRowCandidates = {}
        self._mmPerPixel = 0
        self._stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
        self._centers = []
        self._angles = None
        self._shapeIndices = []
        self._original = img.copy()
        self._imageAsBinary = np.ndarray
        self._imageAsRGB = None
        self._imageAsYIQ = None
        self._imageAsYCBCR = None

        (self._maxY, self._maxX, self._depth) = img.shape
        self._centerLineY = int(self._maxY/2)

        self.log = logging.getLogger(__name__)

    # def init(self):
    #     self._cvimage = cv.cvtColor(self._image)

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
    def mmPerPixel(self) -> float:
        return self._mmPerPixel

    @property
    def blobs(self):
        return self._blobs

    @mmPerPixel.setter
    def mmPerPixel(self, mm : float):
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
        return self._imageAsYCBCR

    @property
    def croplineImage(self):
        return self.cropline_image

    @property
    def shapeIndices(self):
        return self._shapeIndices

    @classmethod
    def show(self, title : str, index : np.ndarray):
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

    def write(self, image: np.ndarray, name: str):
        cv.imwrite(name, image)

    def toRGB(self) -> np.ndarray:
        """
        Converts the current image to RGB from BGR.
        :return: The converted image as an ndarray
        """
        if self._imageAsRGB is None:
            self._imageAsRGB = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2RGB)
        return self._imageAsRGB

    # The YIQ colorspace is described here:
    # https://en.wikipedia.org/wiki/YIQ

    # TODO: This method is quite slow, taking almost 200 ms on test machine
    def toYIQ(self) -> np.ndarray:
        """
        Converts the current image to the YIQ colorspace from RGB.
        Converts to RGB automatically
        :return: The converted image as an ndarray
        """
        # Convert to RGB, as scikit-image doesn't take BGR
        self.toRGB()

        #TODO: This is the one and only use for the scikit-image library.
        # This can be done with some matrix multiplication instead, and is something that can
        # be performed on a GPU
        self._imageAsYIQ = rgb2yiq(self._imageAsRGB)
        return self._imageAsYIQ

    def toHSV(self) -> np.ndarray:
        """
        The current image converted to the HSV colorspace
        :return:
        The HSV values as a numpy array
        """
        self._imgAsHSV =  cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2HSV)
        return self._imgAsHSV

    def toYCBCR(self) -> np.ndarray:
        """
        The current image converted to the YCbCr (YCC) colorspace
        :return:
        The YCbCR values as a numpy array
        """
        self._imageAsYCBCR = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2YCR_CB)
        return self._imageAsYCBCR

    # This code doesn't work exactly right, as I see negative values for saturation
    # And this is unacceptably slow.  Takes over 700 ms on my machine

    def toHSI(self) -> np.ndarray:
        """
        The current image converted to ths HSI colorspace
        :return:
        The HSI values as numpy array
        """
        # TODO: HSI Implementation
        # self._imgAsHSI = cv.cvtColor(self._image.astype(np.uint8), cv.CV_RGB2HLS)
        # return self._imgAsHSI

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
        # Originally: saturation = 1 - 3 * np.divide(minimum, rgb)
        saturation = 1 - 3 * np.divide(minimum, rgb)

        sqrt_calc = np.sqrt(((red - green) * (red - green)) + ((red - blue) * (green - blue)))

        # Avoid having missed datapoints here
        sqrt_calc = np.where(sqrt_calc == 0, 1, sqrt_calc)

        if (green >= blue).any():
            hue = np.arccos((1/2 * ((red-green) + (red - blue)) / sqrt_calc))
        else:
            hue = 2*pi - np.arccos((1/2 * ((red-green) + (red - blue)) / sqrt_calc))

        hue = hue*180/pi

        self._imgAsHSI = cv.merge((hue, saturation, intensity))
        return self._imgAsHSI

    # This code is way too slow
    def RGB2HSI(self):
        """
             This is the function to convert RGB color image to HSI image
             :param rgm_img: RGB color image
             :return: HSI image
        """
        rgb_img = self._image
        #Save the number of rows and columns of the original image
        row = np.shape(rgb_img)[0]
        col = np.shape(rgb_img)[1]
        #Copy the original image
        hsi_img = rgb_img.copy()
        #Channel splitting the image
        B,G,R = cv.split(rgb_img)
        # Normalize the channel to [0,1]
        [B,G,R] = [ i/ 255.0 for i in ([B,G,R])]
        H = np.zeros((row, col))    #Define H channel
        I = (R + G + B) / 3.0       #Calculate I channel
        S = np.zeros((row,col))      #Define S channel
        for i in range(row):
            den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
            thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)   #Calculate the included angle
            h = np.zeros(col)               #Define temporary array
            #den>0 and G>=B element h is assigned to thetha
            h[B[i]<=G[i]] = thetha[B[i]<=G[i]]
            #den>0 and G<=B element h is assigned to thetha
            h[G[i]<B[i]] = 2*np.pi-thetha[G[i]<B[i]]
            #den<0 element h is assigned a value of 0
            h[den == 0] = 0
            H[i] = h/(2*np.pi)      #Assign to the H channel after radiating
        #Calculate S channel
        for i in range(row):
            min = []
            #Find the minimum value of each group of RGB values
            for j in range(col):
                arr = [B[i][j],G[i][j],R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            #Calculate S channel
            S[i] = 1 - min*3/(R[i]+B[i]+G[i])
            #I is 0 directly assigned to 0
            S[i][R[i]+B[i]+G[i] == 0] = 0
        #Extend to 255 for easy display, generally H component is between [0,2pi], S and I are between [0,1]
        # hsi_img[:,:,0] = H*255
        # hsi_img[:,:,1] = S*255
        # hsi_img[:,:,2] = I*255
        self._imgAsHSI = hsi_img
        return hsi_img

    def toGreyscale(self) -> np.ndarray:
        """
        The current image converted to greyscale
        :return:
        The greyscale image as a numpy array
        """
        # This method of converting to greyscale is a complete hack.
        #self.save(self._image, "temporary.jpg")
        #utility.SaveMaskedImage("mask-applied.jpg")
        #img_float32 = np.float32(utility.GetMaskedImage())
        #img = cv.imread("temporary.jpg")
        #self._imgAsGreyscale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        #self._imgAsGreyscale = self._image.astype(np.uint8)
        # If the conversion to uint8 is not there, opencv complains when we try to find the
        # contours.  Strictly speaking this is not required for just the greyscale conversion

        # Blurring the image before we start trying to detect objects seems to improve things
        # in that noise is not identified as objects, but this is a very slow method

        #blurred = cv.pyrMeanShiftFiltering(self._image.astype(np.uint8),31,101)
        #img = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        #
        # Sujith, A., and R. Neethu. 2021. “Classification of Plant Leaf Using Shape and Texture Features.”
        # In 4th International Conference on Inventive Communication and Computational Technologies,
        # ICICCT 2020, edited by Ranganathan G., Chen J., and Rocha A., 145:269–82.
        # Springer Science and Business Media Deutschland GmbH.
        # This article gives the greyscale conversion as:
        # grey = 0.2989 * R + 0.5870 * G + 0.1140 * B
        # TODO: Check the grayscale conversion from opencv

        img = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2GRAY)
        #cv.imwrite("converted.jpg", img)
        self._imgAsGreyscale = img

        return self._imgAsGreyscale


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
    def sizeRatio(sizeOfTarget : int, sizeOfLargest: int) -> float:
        """
        The percentage of the area of the target relative to the largest item.
        :param sizeOfTarget:  The area of blob to be checked
        :param sizeOfLargest:  The area of the largest blob in the current image
        :return: A float value indicating the size ratio of the target to the largest
        """
        return sizeOfTarget / sizeOfLargest


    def findBlobs(self, threshold : int) -> ([], np.ndarray, {}, str):
        """
        Find objects within the current image
        :return: (contours, hierarchy, bounding rectangles, name of largest object)
        """
        self.toGreyscale()
        #self.show("grey", self._imgAsGreyscale)
        #self.write(self._imgAsGreyscale, "greyscale.jpg")

        self.cartoon()
        #self.show("cartooned", self._cartooned)
        #self.write(self._image, "original.jpg")
        #self.write(self._cartooned, "cartooned.jpg")

        #self.write(self._image, "index.jpg")

        # Convert to binary image
        # Works
        #ret,thresh = cv.threshold(self._cartooned,127,255,0)
        ret,thresh = cv.threshold(self._imgAsGreyscale, 127,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        self.write(thresh, "threshold.jpg")
        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(self._cartooned, kernel,iterations = 4)
        #erosion = cv.erode(erosion, kernel,iterations = 3)
        #self.write(erosion, "erosion.jpg")
        erosion = cv.dilate(erosion,kernel,iterations = 3) # originally 3

        closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel)
        self.write(closing, "closing.jpg")
        #self.show("binary", erosion)
        self.write(erosion, "binary.jpg")
        self._imageAsBinary = erosion

        # Originally
        # candidate = erosion

        largestName = "unknown"
        area = 0
        candidate = closing
        self.write(candidate, "candidate.jpg")
        # find contours in the binary image
        #im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        # We don't need the hierarchy at this point, so the RETR_EXTERNAL seems faster
        #contours, hierarchy = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(candidate,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

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

            x,y,w,h = cv.boundingRect(c)
            # The area of the bounding rectangle
            #area = w*h
            # The area of the vegetation
            area = cv.contourArea(c)
            type = constants.TYPE_UNKNOWN
            location = (x,y,w,h)
            center = (cX,cY)
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
                #print("Find: " + str(attributes[constants.NAME_CENTER]))
                # Determine if the point is within the blob
                isInsideContour = cv.pointPolygonTest(attributes[constants.NAME_CONTOUR],attributes[constants.NAME_CENTER], False)
                if isInsideContour:
                    attributes[constants.NAME_TYPE] = constants.TYPE_IGNORED
                # if name in self._blobs:
                #     attributes = self._blobs[name]
                #     attributes[constants.NAME_TYPE] = constants.TYPE_UNTREATED
                #     print("detected overlap")
            i = i + 1

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
            if(x == 0 or x+w >= maxX):
                shapeIndex = 0
            else:
                # The perimeter of the contour of the object
                perimeter = cv.arcLength(blobAttributes[constants.NAME_CONTOUR],True)
                shapeIndex = perimeter / (4 * math.sqrt(blobAttributes[constants.NAME_AREA]))

            blobAttributes[constants.NAME_SHAPE_INDEX] = shapeIndex
            self._shapeIndices.append(shapeIndex)
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
        xCoordinates = contour[:,0,0]
        # The Y values of the contour
        yCoordinates = contour[:,0,1]

        varX = np.var(xCoordinates)
        varY = np.var(yCoordinates)
        covXY = np.cov(xCoordinates, yCoordinates)[0,1]

        # The covariance matrix shown in Lin 2017 equation 2
        s = np.array([[varX,covXY],[covXY,varY]])
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
            if(x == 0 or x+w >= maxX):
                lengthWidthRatio = 0
                blobAttributes[constants.NAME_RATIO] = 0
            else:
                contour = blobAttributes[constants.NAME_CONTOUR]
                blobAttributes[constants.NAME_RATIO] = self.lengthWidthRatio(contour)

        return

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
        middleY = int(height/2)
        for blobName, blobAttributes in self._blobs.items():
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            (x, y) = blobAttributes[constants.NAME_CENTER]
            # Items near the midline are more likely to be crop
            # ?? Perhaps we should also consider equal spacing?
            if(y > middleY - MIDDLE_THRESHOLD and y < middleY + MIDDLE_THRESHOLD):
                self._cropRowCandidates[blobName] = blobAttributes

    def substituteRectanglesForVegetation(self):
        """
        Deprecated. Do not use this method.
        Using the identified centers of where the vegetation, draw low height rectangles
        that are later used for crop line detection
        """
        self.cropline_image = np.zeros(self._image.shape, np.uint8)
        for blobName, blobAttributes in self._cropRowCandidates.items():
            (x,y,w,h) = blobAttributes[constants.NAME_LOCATION]
            (x,y) = blobAttributes[constants.NAME_CENTER]
            #cv.circle(self.blank_image,(x,y),1, (255,255,255), -1)
            self.cropline_image = cv.rectangle(self.cropline_image, (x, y), (x + 100, y + 30), (255, 255, 255), 2)

        #filename = "candidates-" + str(uuid.uuid4()) + ".jpg"

        #cv.imwrite(filename, self.cropline_image)

    def detectLines(self):
        """
        Deprecated. Do not use this method.
        """
        dst = cv.Canny(self.cropline_image, 50, 200, None, 3)
        cv.imwrite("edges.jpg", dst)
        #self.linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, minLineLength=100, maxLineGap=1000)
        # This allows for some plants in the crop line to be slightly offset from other plants
        # in the same row
        # What is needed is to detect roughly horizontal lines
        self.linesP = cv.HoughLinesP(dst, 1, np.pi / 120, 50, None, minLineLength=100, maxLineGap=1500)

        self.lines = cv.HoughLines(dst, 50, np.pi/2, 200)

        # if self.linesP is not None:
        #     for i in range(0, len(self.linesP)):
        #         l = self.linesP[i][0]
        #         cv.line(self.cropline_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
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

        # Find the number of horizontal neighbors each element has.

        likelyCropLineY = int(self._maxY / 2)
        weightedDistanceMax = 0

        # for blobName, blobAttributes in self._blobs.items():
        #     blobAttributes[constants.NAME_CROP_SCORE] = 0
        #     point1 = blobAttributes[constants.NAME_CENTER]
        #     for toBlobName, toBlobAttributes in self._blobs.items():
        #         point2 = toBlobAttributes[constants.NAME_CENTER]
        #         angle = self.angleOf((point1[0], point1[1]), (point2[0], point2[1]))
        #         #print("Angle from {} to {} is {}".format(blobName, toBlobName, angle))
        #         if angle < 5:
        #             #print("Found a neighbor for {}".format(blobName))
        #             blobAttributes[constants.NAME_NEIGHBOR_COUNT] = blobAttributes[constants.NAME_NEIGHBOR_COUNT] + 1
        #     # This is potentially on the crop line if it is the largest thing in the image
        #     if blobName == self._largestName:
        #         blobAttributes[constants.NAME_CROP_SCORE] = 1

        # Find the biggest item closest to the center line
        for blobName, blobAttributes in self._blobs.items():
            weightedDistance = blobAttributes[constants.NAME_AREA] * (1 - blobAttributes[constants.NAME_DISTANCE_NORMALIZED])

            self.log.debug("Weighted distance of {}: {}".format(blobName, weightedDistance))
            if weightedDistance > weightedDistanceMax:
                weightedDistanceMax = weightedDistance
                likelyCropLineBlob = blobName
                likelyCropLineY = blobAttributes[constants.NAME_CENTER][1]

        self._cropY = likelyCropLineY
        self.log.debug("Crop line Y: {} for blob {}".format(self._cropY, likelyCropLineBlob))

        # Step through and replace the normalized distance to the center line
        # with the normalized distance to the crop line
        for blobName, blobAttributes in self._blobs.items():
            (x,y) = blobAttributes[constants.NAME_CENTER]
            blobAttributes[constants.NAME_DISTANCE_NORMALIZED] = self.normalizedDistanceToCropY(y)

        return self._cropY





        return

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
            if blobAttributes[constants.NAME_TYPE] != constants.TYPE_UNDESIRED and blobAttributes[constants.NAME_TYPE] != constants.TYPE_IGNORED:
                centerCount = centerCount + 1
                #print(blobName + ": " + str(blobAttributes[constants.NAME_CENTER]))
                # Keep the name with the coordinates so we know which blob this refers to
                self._centersUnsorted.append(blobAttributes[constants.NAME_CENTER] + tuple([blobName]))
        #print(self._centersUnsorted)

        # Create an array to hold the angles between the blobs
        self._angles = np.zeros((centerCount, centerCount))

        # Sort the centers by the X value -- the first
        #self._centers = sorted(self._centersUnsorted, key=lambda x: x[0])
        #print(self._centers)

        self._centers = self._centersUnsorted

        # There may have only one crop in the image
        if len(self._centers) > 1:
            for i in range(len(self._centers)):
                for j in range(len(self._centers)):
                    point1 = self._centers[i]
                    point2 = self._centers[j]
                    angle = self.angleOf((point1[0], point1[1]), (point2[0], point2[1]))
                    self._angles[i, j] = angle
                    #print(str(point1) + " to " + str(point2) + " angle " + str(angle))
                    #cv.line(self._image, (point1[0], point1[1]), (point2[0], point2[1]), (0,255,0), 3, cv.LINE_AA)
        else:
            (x,y,name) = self._centers[0]
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
        #print(sums)

        smallestDistanceFromY = 10000
        smallestIndex = 10000

        (y, centerX, depth) = self._image.shape
        centerY = int(y/2)



        for index, row in sums.iteritems():

            (x, y, blobName) = self._centers[index]
            distanceFromY = abs(centerY - y)
            #print("Distance from Y: " + str(distanceFromY) + " smallestY: " + str(smallestDistanceFromY))
            if(distanceFromY < smallestDistanceFromY):
                smallestIndex = index
                smallestDistanceFromY = distanceFromY

        (cropX, cropY, blobName) = self._centers[smallestIndex]

        # This is all we need. The Y location of the crop line in the image
        #self._cropY = cropY
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
        cv.line(self._image, (0,int(height/2)), (width, int(height/2)), (0,127,127), 3, cv.LINE_AA)
        cv.putText(self._image,
                   "Center Line",
                   (int(width/2), int(height/2) + 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.75,
                   (0,127,127),
                   2)

        cv.line(self._image, (0,self._cropY), (width, self._cropY), (255,255,255), 3, cv.LINE_AA)
        cv.putText(self._image,
                   "Crop Line",
                   (int(width/2) + 200, self._cropY + 20),
                   cv.FONT_HERSHEY_SIMPLEX,
                   0.75,
                   (255,255,255),
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
            cv.drawContours(self._image, contour, contourIdx=-1, color=(255,0,0),thickness=5)
        # self._contours_image = np.zeros(self._image.shape, np.uint8)
        # cv.drawContours(self._contours_image, self._contours, contourIdx=-1, color=(255,0,0),thickness=2)
        # cv.imwrite("contours.jpg", self._contours_image)

    def drawBoxes(self, name: str, rectangles: [], decorations: []):
        """
        Draw colored boxes around the blobs based on what type they are
        :param name: The name of the image
        :param rectangles: A list of rectangles surrounding the blobs
        """

        cv.putText(self._image, name, (50,75), cv.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

        for rectName, rectAttributes in rectangles.items():
            (x,y,w,h) = rectAttributes[constants.NAME_LOCATION]
            (cX,cY) = rectAttributes[constants.NAME_CENTER]
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
                self._image = cv.rectangle(self._image,(x,y),(x+w,y+h),color,2)
                cv.circle(self._image, (cX, cY), 5, (255, 255, 255), -1)
                location = "(" + str(cX) + "," + str(cY) + ")"
                areaText = "Area: " + str(area)
                shapeText = "Shape: " + "{:.4f}".format(rectAttributes[constants.NAME_SHAPE_INDEX])
                lengthWidthRatioText = "L/W Ratio: " + "{:4f}".format(rectAttributes[constants.NAME_RATIO])
                reasonText = "Classified By: " + constants.REASONS[rectAttributes[constants.NAME_REASON]]
                classifiedText = "Classified As: " + constants.TYPES[rectAttributes[constants.NAME_TYPE]]
                distanceText = "Normalized Distance: " + "{:.4f}".format(rectAttributes[constants.NAME_DISTANCE_NORMALIZED])
                nameText = "Name: {}".format(rectName)
                hueText = "Hue: {:.4f}".format(rectAttributes[constants.NAME_HUE])
                if constants.NAME_LOCATION in decorations:
                    cv.putText(self._image, location, (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_AREA in decorations:
                    cv.putText(self._image, areaText, (cX - 25, cY - 50),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_SHAPE_INDEX in decorations:
                    cv.putText(self._image, shapeText, (cX - 25, cY - 75), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_RATIO in decorations:
                    cv.putText(self._image, lengthWidthRatioText, (cX - 25, cY - 100), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_REASON in decorations:
                    cv.putText(self._image, reasonText, (cX - 25, cY - 125), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_TYPE in decorations:
                    cv.putText(self._image, classifiedText, (cX - 25, cY - 150), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_NAME in decorations:
                    cv.putText(self._image, nameText, (cX - 25, cY - 175), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                if constants.NAME_DISTANCE_NORMALIZED in decorations:
                    cv.putText(self._image, distanceText,(cX - 25, cY- 200), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
                    cv.line(self._image,(cX, cY), (cX, self._cropY), (255,255,255), 3)
                if constants.NAME_HUE in decorations:
                    cv.putText(self._image, hueText,(cX - 25, cY- 225), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)


        #cv.imwrite("opencv-centers.jpg", self._image)
        #self.show("centers", self._image)
        #cv.waitKey()

    def stitchTo(self, previous):
        # Stitch the current image to the previous one
        status, pano = self._stitcher.stitch([previous.astype(np.uint8),self._image.astype(np.uint8)])
        return pano
        #cv.imwrite("stitched.jpg", pano)

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
            cv.putText(self._image, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Do not consider rotation -- just bound the object
            x,y,w,h = cv.boundingRect(c)
            self._image = cv.rectangle(self._image,(x,y),(x+w,y+h),COLOR_CROP,2)

            # Minimum area bounding
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(self._image,[box],0,COLOR_UNKNOWN,BOUNDING_BOX_THICKNESS)

            # Append the current rectangle
            self._rectangles.append(rect)

            #cv.imwrite("centers.jpg", cv.cvtColor(self._image,))
        self.show("centers", self._image)
        cv.waitKey()
        return

    # This does not do quite what is need.  If some unwanted vegetation is inside the bounding box, it is
    # present in the extracted image.
    # TODO: Write a routine to eliminate a class of object within the image.
    def extractImages(self, classifiedAs: int):
        for blobName, blobAttributes in self._blobs.items():
            if blobAttributes[constants.NAME_TYPE] == classifiedAs:
                (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
                # Pull the subset from the original image so we don't see the markings
                image = self._original[y:y+h, x:x+w]
                blobAttributes[constants.NAME_IMAGE] = image

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
                image = source[y:y+h, x:x+w, zslice]
                # This seems to be a very specific case.  After we have masked from the vegetation index
                # The black values are 0,0,0.  Convert them to NaN so when we perform calculations, we don't
                # use the black pixels
                image = np.where(image == 0, np.nan, image)
                if np.isnan(image).all():
                    self.log.error("All values for attribute are NaN: " + attribute)
                #hueMean = np.nanmean(image)
                hueMean = manipulation(image)
                self.log.debug(attribute + ": " + str(hueMean))
                blobAttributes[attribute] = hueMean

    def _compactness(self, contour) -> float:
        return 1

    def extractAttributes(self):
        raise NotImplementedError

    @staticmethod
    def _area(size):
        return size[0] * size[1]