#
# M A S K
#
# Existing mask operations
#
import os.path
import sys
from enum import Enum

import logging

import numpy as np
import cv2 as cv
from WeedExceptions import ProcessingError

CV_RED = 2
CV_GREEN = 1
CV_BLUE = 0

class Rate(Enum):
    FPR = 0
    FNR = 1
    TPR = 2
    TNR = 3

class Mask:
    def __init__(self, **kwargs):
        self._mask = None
        self._target = None
        self._algorithm = ""
        self._countOfDifferences = 0
        self._fn = 0
        self._fp = 0
        self._tn = 0
        self._tp = 0
        self._p = 0
        self._n = 0
        self._log = logging.getLogger(__name__)

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @property
    def fp(self) -> int:
        return self._fp

    @property
    def fn(self) -> int:
        return self._fn

    @property
    def tp(self) -> int:
        return self._tp

    @property
    def tn(self) -> int:
        return self._tn

    @property
    def p(self) -> int:
        # Positives
        return self._p

    @property
    def n(self) -> int:
        # Negatives
        return self._n

    def population(self) -> int:
        return self._p + self._n

    def rate(self, stat: Rate) -> float:
        """
        Determine the specified rate
        :param stat: rate to calculate
        :return:
        """
        # Fornulae taken from
        # https://en.wikipedia.org/wiki/Sensitivity_and_specificity

        rate = 0.0
        self._log.info(f"Calculating rate: {stat}")
        try:
            # False Negative Rate
            if stat == Rate.FNR:
                rate = self._fn / (self._fn + self._tp)
                #return (self._fn / (self._mask.shape[0] * self._mask.shape[1])) * 100

            # False Positive Rate
            elif stat == Rate.FPR:
                rate = self._fp / (self._fp + self._tn)
                #return (self._fp / (self._mask.shape[0] * self._mask.shape[1])) * 100

            # True Positive
            elif stat == Rate.TPR:
                rate = self._tp / (self._tp + self._fn)

            # True Negative
            elif stat == Rate.TNR:
                rate = self._tn / self._tn + self._fp

            else:
                raise AttributeError(f"Unsupported rate {stat.name}")
        except ZeroDivisionError:
            self._log.warn(f"Caught division by zero error")

        return rate


    # Formulae for precision, recall, and f1 taken from

    # https://www.geeksforgeeks.org/f1-score-in-machine-learning/

    def precision(self) -> float:
        """
        Precision calculated as tp / (tp + fp)
        :return:
        """
        return self._tp / (self._tp + self._fp)

    def recall(self) -> float:
        """
        Recall calculated as tp / (tp + fn)
        :return:
        """
        return self._tp / (self._tp + self._fn)

    def f1(self) -> float:
        """
        F1 calulated as (2 * precision * recall) / (precision + recall)
        :return:
        """
        #f1Score = (2 * self._tp) / ((2 * self._tp) + self._fp + self._fn)
        f1Score = (2 * self.precision() * self.recall()) / (self.precision() + self.recall())
        return f1Score

    def accuracy(self) -> float:
        totalAccuracy = (self._tp + self._tn) / (self._tp + self._fp + self._fn + self._tn)
        return totalAccuracy

    @property
    def algorithm(self) -> str:
        return self._algorithm

    def fpr(self) -> float:
        """
        False Positive Rate
        :return:
        """
        assert self._n > 0
        return self._fp / self._n

    def fnr(self) -> float:
        """
        False Negative Rate
        :return:
        """
        assert self._p > 0
        return self._fn / self._p

    def load(self, source: str):
        self._mask = cv.imread(source, cv.IMREAD_GRAYSCALE)
        cv.imwrite("test-mask.jpg", self._mask)
        # The mask may be 255s and 0s.  Make it 1s and 0s to make things a bit easier
        # There seems to be a bug -- or I am making a mistake -- with photoshop.  Sometimes white is 254, not 255
        self._mask = np.where(self._mask < 254, 0, 1)

        self._n = np.count_nonzero(self._mask == 0)
        self._p = (self._mask.shape[0] * self._mask.shape[1]) - self._n

    def displayStats(self):
        print(f"Min/Max: {np.min(self._mask)}/{np.max(self._mask)}")
        print(f"{np.histogram(self._mask)}")

    @property
    def differences(self) -> int:
        return self._countOfDifferences

    def compare(self, target: str):

        self._fn = 0
        self._fp = 0
        self._tn = 0
        self._tp = 0

        # Check to see if file exists
        if not os.path.isfile(target):
            raise FileNotFoundError(f"Unable to access {target}")

        # And that the mask is already loaded
        if self._mask is None:
            raise RuntimeError(f"No mask loaded")

        self._target = cv.imread(target, cv.IMREAD_GRAYSCALE)
        cv.imwrite("test-target.jpg", self._target)
        print(f"Reference mask before: {np.histogram(self._target, 5)}")
        #self._target = np.where(self._target > 1, 1, 0)
        # Photoshop bug where white is sometimes 254
        self._target = np.where(self._target < 254, 0, 1)


        self._differences = np.zeros_like(self._mask)

        histTarget, binsTarget = np.histogram(self._target, bins=[0.0, 1.0, 255])
        histMask, binsMask = np.histogram(self._mask, bins=[0.0, 1.0, 255])
        print(f"Reference: {histTarget} Bins: {binsTarget}\nMask: {histMask} Bins: {binsMask}")

        rows, columns = self._target.shape
        for row in range(rows):
            for column in range(columns):
                # FN + NP
                if self._target[row, column] != self._mask[row, column]:
                    self._differences[row, column] = 255
                    # Determine if the difference is a false positive or false negative
                    if self._mask[row, column] == 0:
                        self._fn += 1
                    elif self._mask[row, column] == 1:
                        self._fp += 1
                    else:
                        print(f"Unexpected mask value found: {self._mask[row, column]}")
                        assert False
                else:
                    if self._mask[row, column] == 0:
                        self._tn += 1
                    else:
                        self._tp += 1



        # Determine where the two masks are the same
        same = self._target == self._mask

        # Where they are the same, set it to 0 -- where different 255
        # Black in the image means things are the same -- white means different
        self._differences = np.where(same, 0, 255)
        self._countOfDifferences = np.count_nonzero(self._differences)

        #print(f"{np.histogram(self._mask)}")
        cv.imwrite("differences.png", self._differences)

    def apply(self, image: str) -> np.ndarray:
        """
        Appy the mask to the specfied image
        :param image: The target image
        """
        if not os.path.isfile(image):
            raise FileNotFoundError

        # Read in image
        image = cv.imread(image, cv.IMREAD_COLOR)

        rowsInMask, columnsInMask = self._mask.shape
        rowsInImage, columnsInImage, _ = image.shape

        # Make certain the mask size and image size match, ignoring the color channels
        if (rowsInMask, columnsInMask) != (rowsInImage, columnsInImage):
            raise ProcessingError(
                f"Mask {(rowsInMask, columnsInMask)} and Image {(rowsInImage, rowsInImage)} are not compatible sizes")

        # Split out the bands
        self.redBand = image[:, :, CV_RED]
        self.greenBand = image[:, :, CV_GREEN]
        self.blueBand = image[:, :, CV_BLUE]

        # Apply the mask to each band
        self._redBandMasked = self._mask * self.redBand
        self._greenBandMasked = self._mask * self.greenBand
        self._blueBandMasked = self._mask * self.blueBand

        # Merge the bands back together
        maskedBGR = cv.merge((self._blueBandMasked, self._greenBandMasked, self._redBandMasked))

        return maskedBGR

    def __str__(self):
        return(f"Population: {self.population()} Differences: {self._countOfDifferences} N: {self._n} P: {self._p} FP: {self._fp} FN: {self._fn} TP: {self._tp} TN: {self._tn} F1: {self.f1()}")

if __name__ == "__main__":
    import argparse
    import sys
    import os.path

    parser = argparse.ArgumentParser("Mask test")

    parser.add_argument('-i', '--input', action="store", required=True, help="Input mask")
    parser.add_argument('-p', '--processing', action="store", required=False, default="mask", help="Operation")
    parser.add_argument('-t', '--target', action="store", required=False, help="Target Image")
    parser.add_argument('-o', '--output', action="store", required=False, help="Output file")
    parser.add_argument('-m', '--mask', action="store", required=False, help="Target mask for comparison")

    arguments = parser.parse_args()

    theMask = Mask()
    if os.path.isfile(arguments.input):
        theMask.load(arguments.input)
    else:
        print(f"Unable to access {arguments.input}")
        sys.exit(-1)

    if arguments.processing == "compare":
        theMask.compare(arguments.mask)
        print(f"Masks difference count: {theMask.differences}")
    elif arguments.processing == "mask":
        if arguments.target is not None or arguments.output is not None:
            if arguments.output is None:
                print(f"Both output and target mus be specified if either is")
                sys.exit(-1)
            if arguments.target is None:
                print(f"Both output and target mus be specified if either is")
                sys.exit(-1)

            try:
                maskedImage = theMask.apply(arguments.target)
                cv.imwrite(arguments.output, maskedImage)
            except FileNotFoundError:
                print(f"Unable to access image {arguments.image}")
                sys.exit(-1)
            except ProcessingError as p:
                print(f"{p}")
                sys.exit(-1)
    elif arguments.processing == "statistics":
        theMask.displayStats()

    sys.exit(0)
