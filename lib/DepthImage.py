#
# D E P T H  I M A G E
#
# An image from the depth camera
#

import numpy as np
import constants
from typing import Tuple

class DepthImage:
    def __init__(self, depthData: np.ndarray):
        # Later, we will extract just the subset wanted
        self._data = depthData
        self._fullData = depthData

    @property
    def array(self) -> np.ndarray:
        return self._data

    @property
    def original(self) -> np.ndarray:
        return self._fullData

    def averageDistance(self) -> float:
        return np.average(self._data)

    def applyThreshold(self, threshold: float):
        """
        Apply a threshold such that all elements of the array below the threshold are set to 0
        :param threshold: A float specifying the upper bound
        """
        self._data[self._data > threshold] = 0

    def save(self, filename: str):
        np.save(filename, self._data)

    def extractScaledSubset(self,
                            aRGB: Tuple,
                            bRGB: Tuple,
                            aDepth: Tuple,
                            bDepth: Tuple,
                            fromDimensions: Tuple) -> np.ndarray:
        # The length of the line in RGB
        distanceInRGB = abs(aRGB[0] - bRGB[0])
        # The length of the line in the depth data
        distanceInDepth = abs(aDepth[0] - bDepth[0])
        ratio = distanceInDepth / distanceInRGB

        print("Distance of line in RGB: {} in Depth: {}. Ratio is {}".format(distanceInRGB, distanceInDepth, ratio))

        # The corner of the image in the depth data
        upperLeftX = int(aDepth[0] - (aRGB[0] * ratio))
        upperLeftY = int(aDepth[1] - (aRGB[1] * ratio))
        lengthImageInDepth = int(fromDimensions[0] * ratio)
        heightImageInDepth = int(fromDimensions[1] * ratio)

        print(
            "Upper left: ({},{}).  Image in depth data is ({} x {})".format(upperLeftX, upperLeftY, lengthImageInDepth,
                                                                            heightImageInDepth))

        # Extract the sub-image we want
        self._data = self._data[upperLeftY:(upperLeftY + heightImageInDepth), upperLeftX:(upperLeftX + lengthImageInDepth)]

        return self._data


