import argparse
import sys
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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

img = cv.imread(results.image)
manipulated = ImageManipulation(img)
if results.colorspace == "hsv":
    hsv = manipulated.toHSV()
elif results.colorspace == "hsi":
    hsv = manipulated.toHSI()
else:
    print("Unknown color space: " + results.colorspace)
    sys.exit(1)

hue = hsv[:,:,0]
saturation = hsv[:,:,1]
value = hsv[:,:,2]
factors = [hue, saturation, value]

def showGraph():
    yLen,xLen, zLen = hsv.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    x, y = np.meshgrid(x, y)

    for factor in factors:
        fig = plt.figure(figsize=(10,10))
        axes = fig.gca(projection ='3d')
        plt.title("HSV")
        axes.scatter(x, y, factor, c=factor, cmap='BrBG', s=0.25)
        plt.show()
        cv.waitKey()

def thresholdHSV(saturationThresholdLow: int, saturationThresholdHigh: int):
    saturationMatrix = np.where((saturation >= saturationThresholdLow | saturation <= saturationThresholdHigh),1,0)

showGraph()
