#
# F A C T O R S
#
from typing import List
import numpy as np

import constants
import pandas as pd
import logging
from GLCM import GLCM
from enum import Enum

class FactorTypes(Enum):
    COLOR = 0
    SHAPE = 1
    TEXTURE = 2
    POSITION = 3

class FactorSubtypes(Enum):
    NONE = 0
    LBP = 1
    GLCM = 2
    HOG = 3

class FactorKind(Enum):
    SCALAR = 0
    VECTOR = 1

class ColorSpace(Enum):
    NONE = 0
    GREYSCALE = 1
    CIELAB = 2
    RGB = 3
    YIQ = 4
    HSI = 5
    HSV = 6
    YCBCR = 7


class Factors:
    def __init__(self):

        # Angles used for factor where that is relevant -- now that is just GLCM
        self._angles = [0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi]

        # This is one longer than the angles so we can have the average as well
        #self._angleNames = ["", "45", "90", "135", "180", constants.NAME_AVERAGE]
        # Just the average to avoid orientation issues
        self._angleNames = [constants.NAME_AVERAGE]

        allVectorFactors = [
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_LBP, FactorKind.VECTOR]

        ]
        _vectorFactors = [
            # Greyscale
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.GREYSCALE],
            # YIQ - Y
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.YIQ],
            # YIQ - I
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.YIQ],
            # YIQ - Q
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.YIQ],
            # HSV - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.HSV],
            # HSV - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.HSV],
            # HSV - Value
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.HSV],
            # HSI - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.HSI],
            # HSI - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.HSI],
            # HSI - Intensity
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.HSI],
            # Blue
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.RGB],
            # Green
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.RGB],
            # Red
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_RED + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.RGB],
            # YCBCR - Y
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.YCBCR],
            # YCBCR - CB
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.YCBCR],
            # YCBCR - CR
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.YCBCR],
            # CIELAB - L
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.CIELAB],
            # CIELAB - A
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.CIELAB],
            # CIELAB - B
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR, ColorSpace.CIELAB]
        ]
        # The final list of factors -- the ones here do not have angles associated with them
        allFactors = [
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_SHAPE_INDEX, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_COMPACTNESS, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_CONVEXITY, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ELONGATION, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ECCENTRICITY, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ROUNDNESS, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_SOLIDITY, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_BENDING, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_RADIAL_VAR, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_HUE, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_SATURATION, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_I_YIQ, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_BLUE_DIFFERENCE, FactorKind.SCALAR, ColorSpace.YIQ],
            #[constants.PROPERTY_FACTOR_POSITION, constants.NAME_DISTANCE],
            [FactorTypes.POSITION, FactorSubtypes.NONE, constants.NAME_DISTANCE_NORMALIZED, FactorKind.SCALAR, ColorSpace.NONE],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            # Local Binary Pattern
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN,  FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_RED + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR, ColorSpace.YCBCR]
        ]

        # The factors that need to be expanded with their angles
        _factorsWithAngles = [
            # Greyscale
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.GREYSCALE],
            # YIQ - Y
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.YIQ],
            # YIQ - I
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.YIQ],
            # YIQ - Q
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.YIQ],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.YIQ],
            # HSV - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.HSV],
            # HSV - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.HSV],
            # HSV - Value
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.HSV],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.HSV],
            # HSI - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.HSI],
            # HSI - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.HSI],
            # HSI - Intensity
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.HSI],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.HSI],
            # Blue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.RGB],
            # Green
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.RGB],
            # Red
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.RGB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.RGB],
            # YCBCR - Y
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.YCBCR],
            # YCBCR - CB
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.YCBCR],
            # YCBCR - CR
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.YCBCR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.YCBCR],
            # CIELAB - L
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.CIELAB],
            # CIELAB - A
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.CIELAB],
            # CIELAB - B
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR, ColorSpace.CIELAB],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR, ColorSpace.CIELAB],
        ]
        _columnNames = [constants.COLUMN_NAME_TYPE, constants.COLUMN_NAME_SUBTYPE, constants.COLUMN_NAME_FACTOR, constants.COLUMN_NAME_KIND, constants.COLUMN_NAME_COLORSPACE]

        # Add in the GLCM factors
        for factor in _factorsWithAngles:
            for angleName in GLCM.anglesAvailable:
                factorWithAngleName = [factor[0], factor[1], factor[2] + constants.DELIMETER + angleName, FactorKind.SCALAR, factor[4]]
                allFactors.append(factorWithAngleName)

        # This is not the right thing to do -- better to add a type (scalar or vector) and filter on that.
        # For now, just keep them separate
        self._vectors = pd.DataFrame(_vectorFactors, columns=_columnNames)
        self._vectors.set_index(constants.COLUMN_NAME_FACTOR, inplace=True)

        self._data = pd.DataFrame(allFactors, columns=_columnNames)
        self._data.set_index(constants.COLUMN_NAME_FACTOR, inplace=True)

        self._log = logging.getLogger(constants.NAME_LBP)

    @property
    def angles(self) -> []:
        return self._angles

    @property
    def angleNames(self) -> []:
        return self._angleNames

    def composedOfSubtypes(self, factorTypes: List[FactorSubtypes], factors: []) -> bool:
        """
        Determine if all of the factors are of the type provided.
        :param factorType:
        :param factors:
        :return:
        """

        countOfType = 0
        for factor in factors:
            row = self._data.loc[factor]
            #self._log.debug(f"Determine if {row[constants.COLUMN_NAME_SUBTYPE]} is in subtype {factorTypes}")
            if row[constants.COLUMN_NAME_SUBTYPE] in factorTypes:
                countOfType += 1
        # if countOfType == len(factors):
        #     print(f"{factors} is completely composed of subtype {factorTypes}")
        return countOfType == len(factors)



    def composedOfTypes(self, factorTypes: List[FactorTypes], factors: [], **kwargs) -> bool:
        """
        Determine if all of the factors are of the type provided.
        Keywords
        restricted: [(type, colorspace), ...] indicates attribute type must be from specified colorspace
         For instance (FactorTypes.TEXTURE, Colorspace.GREYSCALE) indicates that a TEXTURE attribute must be from the GREYSCALE color space
        :param factorTypes:
        :param factors:
        :return:
        """

        if "restricted" in kwargs:
            restricted = kwargs["restricted"]
        else:
            restricted = None

        countOfType = 0
        for factor in factors:
            row = self._data.loc[factor]
            #self._log.debug(f"Determine if {row[constants.COLUMN_NAME_TYPE]} is in type {factorTypes}")
            if restricted is not None:
                for restriction in restricted:
                    if row[constants.COLUMN_NAME_TYPE] == restriction[0] and row[constants.COLUMN_NAME_COLORSPACE] is not restricted[1]:
                        continue
            if row[constants.COLUMN_NAME_TYPE] in factorTypes:
                countOfType += 1
        return countOfType == len(factors)


    def getColumns(self, factor: [], subtype: [], kind: FactorKind = FactorKind.SCALAR, **kwargs) -> []:
        """
        Get the factors of the types provided
        :param kind:
        :param subtype: Array of factor subtypes (strings)
        :param factor: Array of factor types (strings)
        :param blacklist: Array of factors to exclude
        :return: Array of factors
        """
        if "blacklist" in kwargs:
            blacklist = kwargs["blacklist"]
            if blacklist is None:
                blacklist = []
        else:
            blacklist = []

        if "restricted" in kwargs:
            restricted = kwargs["restricted"]
        else:
            restricted = None


        # All of the factors and subtypes
        allFactorTypes = [e for e in FactorTypes]
        allFactorSubTypes = [e for e in FactorSubtypes]

        # If the type or subtype is an empty list, just set it to everything
        # Otherwise, convert this to an array of enums
        if len(subtype) == 0:
            factorSubtype = allFactorSubTypes
        else:
            # if a list of strings is passed, convert to enum
            if isinstance(subtype[0], str):
                factorSubtype = [FactorSubtypes[e] for e in subtype]
            else:
                factorSubtype = subtype

            #factorSubtype = subtype

        if len(factor) == 0:
            factorTypes = allFactorTypes
        else:
            # If an array of strings is passed
            if isinstance(factor[0], str):
                factorTypes = [FactorTypes[e] for e in factor]
            else:
                factorTypes = factor
            #factorTypes = factor

        subset = []
        if kind == FactorKind.SCALAR:
            for index, row in self._data.iterrows():
                #print(f"Consider: {index}\n{row}")
                if row[constants.COLUMN_NAME_TYPE] in factorTypes \
                        and row[constants.COLUMN_NAME_SUBTYPE] in factorSubtype \
                        and row[constants.COLUMN_NAME_KIND] == kind \
                        and index not in blacklist:

                    # Step through the restriction pairs to see if a row matches
                    if restricted is not None:
                        parameterOK = False
                        for restriction in restricted:
                            #print(f"Check for restriction: ({restriction[0]}, {restriction[1]}): ({row[constants.COLUMN_NAME_TYPE]}, {row[constants.COLUMN_NAME_COLORSPACE]})")
                            # Check to see if the type is not restricted
                            if row[constants.COLUMN_NAME_TYPE] != restriction[0] and row[constants.COLUMN_NAME_COLORSPACE] != restriction[1]:
                                parameterOK = True
                            # Check to see if the type and the colorspace match
                            if row[constants.COLUMN_NAME_TYPE] == restriction[0] and row[constants.COLUMN_NAME_COLORSPACE] == restriction[1]:
                                parameterOK = True

                        if parameterOK:
                            subset.append(index)
                    else:
                        #subset.append(row[constants.COLUMN_NAME_FACTOR])
                        subset.append(index)
                else:
                    pass
                    #print(f"Rejected: {row}")
        else:
            for index, row in self._vectors.iterrows():
                #print(f"Consider: {row}")
                if row[constants.COLUMN_NAME_TYPE] in factorTypes and row[constants.COLUMN_NAME_SUBTYPE] in factorSubtype and row[constants.COLUMN_NAME_KIND] == kind:
                    #subset.append(row[constants.COLUMN_NAME_FACTOR])
                    subset.append(index)
                else:
                    pass
                    #print(f"Rejectedi: {row}")
        return subset


if __name__ == "__main__":
    import argparse
    from Tidy import Tidy

    factorTypes = [e for e in FactorTypes]
    factorChoices = [e.name for e in FactorTypes]
    factorChoices.append(constants.NAME_ALL)
    factorSubtypes = [e for e in FactorSubtypes]
    factorSubtypeChoices = [e.name for e in FactorSubtypes]
    factorSubtypeChoices.append(constants.NAME_ALL)
    kindChoices = [e.name for e in FactorKind]


    parser = argparse.ArgumentParser("Show factors")
    #parser.add_argument("-n", "--name", required=False, help="Restrict names to specified prefix")
    parser.add_argument("-t", "--type", required=False, nargs='*', choices=factorChoices, default=constants.NAME_ALL, help="Types to display")
    parser.add_argument("-s", "--subtype", required=False, nargs='*', choices=factorSubtypeChoices, default=constants.NAME_ALL, help="Types to display")
    parser.add_argument("-k", "--kind", required=False, choices=kindChoices, default=FactorKind.SCALAR.name, help="Kinds to display")
    parser.add_argument("-g", "--greyscale", required=False, action="store_true", default=False, help="Restrict texture attributes to greyscale")
    parser.add_argument('-i', '--input', required=False, help="Input CSV -- used for analysis")
    arguments = parser.parse_args()

    if constants.NAME_ALL in arguments.type:
        types = factorTypes
    else:
        types = [FactorTypes[e] for e in arguments.type]

    if constants.NAME_ALL in arguments.subtype:
        subtypes = factorSubtypes
    else:
        subtypes = [FactorSubtypes[e] for e in arguments.subtype]

    # User specified a file to analyze
    if arguments.input is not None:
        t = Tidy(None)
        t.load(arguments.input)
        t.analyze()
        exclude = t.columnsToDrop
        if len(exclude) > 0:
            print(f"Exclude {exclude}")
    else:
        exclude = None

    if arguments.greyscale:
        restrictions = [(FactorTypes.TEXTURE, ColorSpace.GREYSCALE)]
    else:
        restrictions = None

    # Convert string to enum
    kind = FactorKind[arguments.kind]

    allFactors = Factors()
    print(f"Type :{arguments.type} subtype: {arguments.subtype} | {allFactors.getColumns(types, subtypes, kind, blacklist=exclude, restricted=restrictions)}")
    # print(f"Shape: {allFactors.getColumns([FactorTypes.SHAPE], [])}")
    # print(f"Color: {allFactors.getColumns([FactorTypes.COLOR], [])}")
    # print(f"GLCM: {allFactors.getColumns([FactorTypes.TEXTURE], [])}")
    # print(f"LBP: {allFactors.getColumns([FactorTypes.TEXTURE], [])}")
    # print(f"Position: {allFactors.getColumns([constants.PROPERTY_FACTOR_POSITION], [])}")
    # print(f"Color and GLCM: {allFactors.getColumns([FactorTypes.COLOR, FactorTypes.TEXTURE], [])}")
    # print(f"All: {allFactors.getColumns(None)}")
