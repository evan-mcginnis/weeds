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

class EdgeDetection:
    """
    Detect the edges in an object
    """

    def __init__(self,
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

        self.markupColors = (0, 0, 255)

        # For edge detection
        self.threshold1 = threshold1 or 50
        self.threshold2 = threshold2 or 200
        self.apertureSize = apertureSize or 3


        self.src = None
        self._blurred = None

    def load(self, image_name: str):
        self.src = cv.imread(cv.samples.findFile(image_name), cv.IMREAD_GRAYSCALE)
        # Check if image is loaded fine


        if self.src is None:
            print('Error opening image')
            return -1

        self._blurred = cv.GaussianBlur(self.src, (3, 3), sigmaX=0, sigmaY=0)
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
        self.dst = cv.Canny(self._blurred, self.threshold1, self.threshold2, self.apertureSize, None)
        self.cdst = cv.cvtColor(self.dst, cv.COLOR_GRAY2BGR)
        self.cdstP = np.copy(self.cdst)

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
                cv.line(self.cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    def getMarkedUp(self):
        return self.cdstP

    def getMarkedUp2(self):
        return self.cdst

    def getOriginal(self):
        return self.src

    def get_dst(self):
        return (self.dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Overlap detection")
    parser.add_argument('-i', '--input', action="store", required=True, help="Images to process")
    parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")

    parser.add_argument('-t1', '--threshold1', action="store", required=False, default=50, type=int)
    parser.add_argument('-t2', '--threshold2', action="store", required=False, default=100, type=int)

    parser.add_argument("-p", "--process", action="store_true", required=False, default=False, help="Process image")
    arguments = parser.parse_args()

    edgeDetector = EdgeDetection(threshold1=arguments.threshold1, threshold2=arguments.threshold2)

    if arguments.process:
        img = cv.imread(arguments.input, cv.CV_8UC1)
        th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        cv.imshow("Threshold", th3)

    if not edgeDetector.load(arguments.input):
        edgeDetector.detectEdges()
        #edgeDetector.markupOriginal()

        if arguments.output is not None:
            cv.imwrite(arguments.output, edgeDetector.getMarkedUp())
        else:
            cv.imshow("Source", edgeDetector.getOriginal())
            cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", edgeDetector.getMarkedUp2())
            cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", edgeDetector.getMarkedUp())
            cv.imshow("Edges", edgeDetector.get_dst())
            cv.waitKey()
