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
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # YIQ - Y
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # YIQ - I
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # YIQ - Q
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # HSV - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # HSV - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # HSV - Value
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # HSI - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # HSI - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # HSI - Intensity
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # Blue
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # Green
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # Red
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_RED + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # YCBCR - Y
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # YCBCR - CB
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # YCBCR - CR
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # CIELAB - L
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # CIELAB - A
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR],
            # CIELAB - B
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_LBP, FactorKind.VECTOR]
        ]
        # The final list of factors -- the ones here do not have angles associated with them
        allFactors = [
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_SHAPE_INDEX, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_COMPACTNESS, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_CONVEXITY, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ELONGATION, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ECCENTRICITY, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ROUNDNESS, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_SOLIDITY, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_BENDING, FactorKind.SCALAR],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_RADIAL_VAR, FactorKind.SCALAR],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_HUE, FactorKind.SCALAR],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_SATURATION, FactorKind.SCALAR],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_I_YIQ, FactorKind.SCALAR],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_BLUE_DIFFERENCE, FactorKind.SCALAR],
            #[constants.PROPERTY_FACTOR_POSITION, constants.NAME_DISTANCE],
            [FactorTypes.POSITION, FactorSubtypes.NONE, constants.NAME_DISTANCE_NORMALIZED, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            # Local Binary Pattern
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN,  FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_RED + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR, FactorKind.SCALAR]
        ]

        # The factors that need to be expanded with their angles
        _factorsWithAngles = [
            # Greyscale
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # YIQ - Y
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # YIQ - I
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # YIQ - Q
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # HSV - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # HSV - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # HSV - Value
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # HSI - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # HSI - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # HSI - Intensity
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            # Blue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # Green
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # Red
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # YCBCR - Y
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # YCBCR - CB
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # YCBCR - CR
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # CIELAB - L
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # CIELAB - A
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
            # CIELAB - B
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CORRELATION, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_HOMOGENEITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ENERGY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CONTRAST, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_DISSIMILARITY, FactorKind.SCALAR],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ASM, FactorKind.SCALAR],
        ]
        _columnNames = [constants.COLUMN_NAME_TYPE, constants.COLUMN_NAME_SUBTYPE, constants.COLUMN_NAME_FACTOR, constants.COLUMN_NAME_KIND]

        # Add in the GLCM factors
        for factor in _factorsWithAngles:
            for angleName in GLCM.anglesAvailable:
                factorWithAngleName = [factor[0], factor[1], factor[2] + constants.DELIMETER + angleName, FactorKind.SCALAR]
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


    def composedOfTypes(self, factorTypes: List[FactorTypes], factors: []) -> bool:
        """
        Determine if all of the factors are of the type provided.
        :param factorType:
        :param factors:
        :return:
        """

        countOfType = 0
        for factor in factors:
            row = self._data.loc[factor]
            #self._log.debug(f"Determine if {row[constants.COLUMN_NAME_TYPE]} is in type {factorTypes}")
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
    parser.add_argument("-t", "--type", required=False, nargs='*', choices=factorChoices, default=constants.NAME_ALL, help="Types to display")
    parser.add_argument("-s", "--subtype", required=False, nargs='*', choices=factorSubtypeChoices, default=constants.NAME_ALL, help="Types to display")
    parser.add_argument("-k", "--kind", required=False, choices=kindChoices, default=FactorKind.SCALAR.name, help="Kinds to display")
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


    # Convert string to enum
    kind = FactorKind[arguments.kind]

    allFactors = Factors()
    print(f"Type :{arguments.type} subtype: {arguments.subtype} | {allFactors.getColumns(types, subtypes, kind, blacklist=exclude)}")
    # print(f"Shape: {allFactors.getColumns([FactorTypes.SHAPE], [])}")
    # print(f"Color: {allFactors.getColumns([FactorTypes.COLOR], [])}")
    # print(f"GLCM: {allFactors.getColumns([FactorTypes.TEXTURE], [])}")
    # print(f"LBP: {allFactors.getColumns([FactorTypes.TEXTURE], [])}")
    # print(f"Position: {allFactors.getColumns([constants.PROPERTY_FACTOR_POSITION], [])}")
    # print(f"Color and GLCM: {allFactors.getColumns([FactorTypes.COLOR, FactorTypes.TEXTURE], [])}")
    # print(f"All: {allFactors.getColumns(None)}")
