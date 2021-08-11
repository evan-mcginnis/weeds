import argparse
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os.path

from CameraFile import CameraFile, CameraPhysical
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger

parser = argparse.ArgumentParser("View HSV of image")

parser.add_argument('-i', '--image', action="store", required = True, help="Target image")
parser.add_argument('-o', '--output', action="store", help="Output directory for processed images")
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-c", "--colorspace", action="store", type=str,default="hsv")

results = parser.parse_args()

# Read image

if not os.path.exists(results.image):
    print("Image does not exist: " + results.image)
    sys.exit(1)

img = cv.imread(results.image)
manipulated = ImageManipulation(img, 0)
if results.colorspace == "hsv":
    factorNames = ["hue", "saturation", "value"]
    converted = manipulated.toHSV()
elif results.colorspace == "hsi":
    factorNames = ["hue", "saturation", "intensity"]
    converted = manipulated.toHSI()
elif results.colorspace == "yiq":
    factorNames = ["yellow", "in-phase", "quadrature"]
    converted = manipulated.toYIQ()
elif results.colorspace == "ycc":
    factorNames = ["luma (y)", "blue-difference chroma (c)", "red-difference chroma (c)"]
    converted = manipulated.toYCBCR()
else:
    print("Unknown color space: " + results.colorspace + ". Specify one of hsv, hsi, yiq, or ycc")
    sys.exit(1)

factor1 = converted[:, :, 0]
factor2 = converted[:, :, 1]
factor3 = converted[:, :, 2]
factors = [factor1, factor2, factor3]

def showGraph():
    yLen,xLen, zLen = converted.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    x, y = np.meshgrid(x, y)
    factorName = 0

    for factor in factors:
        fig = plt.figure(figsize=(10,10))
        axes = fig.gca(projection ='3d')
        plt.title(results.colorspace + " : " + factorNames[factorName])
        axes.scatter(x, y, factor, c=factor, cmap='BrBG', s=0.25)
        plt.show()
        cv.waitKey()
        factorName = factorName + 1

def thresholdHSV(saturationThresholdLow: int, saturationThresholdHigh: int):
    saturationMatrix = np.where((factor2 >= saturationThresholdLow | factor2 <= saturationThresholdHigh), 1, 0)

showGraph()
sys.exit(0)