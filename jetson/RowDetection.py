"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
# import sys
import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from PIL import Image

class RowDetection:
    """
    Detect the rows in an image using a Hough transform.
    """

    def __init__(self,
                 rho: int = None,
                 theta: float = None,
                 threshold: int = None,
                 minLineLength: int = None,
                 maxLineGap: int = None,
                 threshold1: int = None,
                 threshold2: int = None,
                 apertureSize: int = None):
        """

        :param minWidth:
        :param rho: The resolution of the parameter r in pixels
        :param theta: The resolution of the parameter θ in radians.
        :param threshold: The minimum number of intersections to "*detect*" a line
        :param minLineLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
        :param maxLineGap: The maximum gap between two points to be considered in the same line.
        """
        # Default parameters

        # For line detection
        self.rho = rho or 1
        self.theta = theta or np.pi / 180
        self.threshold = threshold or 50
        self.minLineLength = minLineLength or 50
        self.maxLineGap = maxLineGap or 10

        self.markupColors = (0, 0, 255)

        # For edge detection
        self.threshold1 = threshold1 or 50
        self.threshold2 = threshold2 or 200
        self.apertureSize = apertureSize or 3
        return

    def load(self, image_name: str):
        self.src = cv.imread(cv.samples.findFile(image_name), cv.IMREAD_GRAYSCALE)
        # Check if image is loaded fine

        if self.src is None:
            print('Error opening image')
            return -1

    # def detectEdges(self):
    #     self.dst = cv.Canny(self.src, 50, 200, None, 3)
    #
    # def detectLines(self):
    #     # Copy edges to the images that will display the results in BGR
    #     cdst = cv.cvtColor(self.dst, cv.COLOR_GRAY2BGR)
    #     cdstP = np.copy(cdst)
    #
    #     self.lines = cv.HoughLines(self.dst, 1, np.pi / 180, 150, None, 0, 0)

    def detectEdges(self):
        """
        Detect edges in the loaded image
        """
        self.dst = cv.Canny(self.src, self.threshold1, self.threshold2, None, self.apertureSize)
        self.cdst = cv.cvtColor(self.dst, cv.COLOR_GRAY2BGR)
        self.cdstP = np.copy(self.cdst)

    def detectLines(self):
        """
        Detect lines in the loaded image that has already had edges identified
        """
        # The probabilistic method
        self.linesP = cv.HoughLinesP(self.dst,
                                     self.rho,
                                     self.theta,
                                     self.threshold,
                                     None,
                                     self.minLineLength,
                                     self.maxLineGap)
        # The standard method
        self.lines = cv.HoughLines(self.dst, 1, np.pi / 180, 150, None, 0, 0)

    def markupOriginal(self):
        if self.linesP is not None:
            for i in range(0, len(self.linesP)):
                l = self.linesP[i][0]
                cv.line(self.cdstP, (l[0], l[1]), (l[2], l[3]), self.markupColors, 3, cv.LINE_AA)

        if self.lines is not None:
            for i in range(0, len(self.lines)):
                rho = self.lines[i][0][0]
                theta = self.lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv.line(self.cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    def getMarkedUp(self):
        return self.cdstP

    def getMarkedUp2(self):
        return self.cdst

    def getOriginal(self):
        return self.src

    def get_dst(self):
        return (self.dst)


if __name__ == "__main__":
    row_detector = RowDetection(threshold1=300)
    #row_detector.load("overhead.jpg")
    #image = "sudoku.png"
    #image = "overhead.jpg"
    image = "sudoku-2.png"
    # im = Image.open(image)
    # imAsNp = np.array(im)
    # imAsNp[:,:,1] *=0
    # imAsNp[:,:,2] *=0
    # img = Image.fromarray(imAsNp)
    img = cv.imread(image, cv.CV_8UC1)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)
    cv.imshow("Threshold", th3)

    if not row_detector.load(image):
        row_detector.detectEdges()
        row_detector.detectLines()
        row_detector.markupOriginal()

        cv.imshow("Source", row_detector.getOriginal())
        cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", row_detector.getMarkedUp2())
        cv.imshow("Detected YLines (in red) - Probabilistic Line Transform", row_detector.getMarkedUp())
        cv.imshow("Edges", row_detector.get_dst())
        cv.waitKey()
