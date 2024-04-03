#
# F A C T O R S
#
import numpy as np

import constants
import pandas as pd
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


class Factors:
    def __init__(self):

        # Angles used for factor where that is relevant -- now that is just GLCM
        self._angles = [0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi]

        # This is one longer than the angles so we can have the average as well
        #self._angleNames = ["", "45", "90", "135", "180", constants.NAME_AVERAGE]
        # Just the average to avoid orientation issues
        self._angleNames = [constants.NAME_AVERAGE]

        # The final list of factors -- the ones here do not have angles associated with them
        allFactors = [
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_SHAPE_INDEX],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_COMPACTNESS],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_CONVEXITY],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ELONGATION],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ECCENTRICITY],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_ROUNDNESS],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_SOLIDITY],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_BENDING],
            [FactorTypes.SHAPE, FactorSubtypes.NONE, constants.NAME_RADIAL_VAR],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_HUE],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_SATURATION],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_I_YIQ],
            [FactorTypes.COLOR, FactorSubtypes.NONE, constants.NAME_BLUE_DIFFERENCE],
            #[constants.PROPERTY_FACTOR_POSITION, constants.NAME_DISTANCE],
            [FactorTypes.POSITION, FactorSubtypes.NONE, constants.NAME_DISTANCE_NORMALIZED],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_STDDEV],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_MEAN],
            [FactorTypes.TEXTURE, FactorSubtypes.HOG, constants.NAME_HOG + constants.DELIMETER + constants.NAME_VAR],
            # Local Binary Pattern
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_LBP + constants.DELIMETER + constants.NAME_STDDEV],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_LBP + constants.DELIMETER + constants.NAME_MEAN],
            [FactorTypes.TEXTURE, FactorSubtypes.LBP, constants.NAME_LBP + constants.DELIMETER + constants.NAME_VAR]
        ]

        # The factors that need to be expanded with their angles
        _factorsWithAngles = [
            # Greyscale
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CORRELATION],
            # YIQ - Y
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CORRELATION],
            # YIQ - I
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CORRELATION],
            # YIQ - Q
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSV - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSV - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSV - Value
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSI - Hue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSI - Saturation
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSI - Intensity
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ASM],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CORRELATION],
            # Blue
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ASM],
            # Green
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ASM],
            # Red
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ASM],
            # YCBCR - Y
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ASM],
            # YCBCR - CB
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM],
            # YCBCR - CR
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM],
            # CIELAB - L
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ASM],
            # CIELAB - A
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ASM],
            # CIELAB - B
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CORRELATION],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ENERGY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CONTRAST],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [FactorTypes.TEXTURE, FactorSubtypes.GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ASM],
        ]
        _columnNames = [constants.COLUMN_NAME_TYPE, constants.COLUMN_NAME_SUBTYPE, constants.COLUMN_NAME_FACTOR]

        # Add in the GLCM factors
        for factor in _factorsWithAngles:
            for angleName in GLCM.anglesAvailable:
                factorWithAngleName = [factor[0], factor[1], factor[2] + constants.DELIMETER + angleName]
                allFactors.append(factorWithAngleName)

        self._data = pd.DataFrame(allFactors, columns=_columnNames)
        return

    @property
    def angles(self) -> []:
        return self._angles

    @property
    def angleNames(self) -> []:
        return self._angleNames

    def containsType(self, factorType: str, factors: []) -> bool:
        """
        Determine if any of the factors are of the type provided.
        :param factorType:
        :param factors:
        :return:
        """
        found = True

        raise NotImplemented

        return found

    def getColumns(self, factor: [], subtype: []) -> []:
        """
        Get the factors of the types provided
        :param subtype: Array of factor subtypes (strings)
        :param factor: Array of factor types (strings)
        :return: Array of factors
        """
        # All of the factors and subtypes
        allFactorTypes = [e for e in FactorTypes]
        allFactorSubTypes = [e for e in FactorSubtypes]

        # If the type or subtype is an empty list, just set it to everything
        # Otherwise, convert this to an array of enums
        if len(subtype) == 0:
            factorSubtype = allFactorSubTypes
        else:
            #factorSubtype = [FactorSubtypes[e] for e in subtype]
            factorSubtype = subtype

        if len(factor) == 0:
            factorTypes = allFactorTypes
        else:
            # factorTypes = [FactorTypes[e] for e in factor]
            factorTypes = factor

        subset = []
        for index, row in self._data.iterrows():
            if row[constants.COLUMN_NAME_TYPE] in factorTypes and row[constants.COLUMN_NAME_SUBTYPE] in factorSubtype:
                subset.append(row[constants.COLUMN_NAME_FACTOR])
        return subset


if __name__ == "__main__":
    import argparse

    factorTypes = [e for e in FactorTypes]
    factorChoices = [e.name for e in FactorTypes]
    factorChoices.append(constants.NAME_ALL)
    factorSubtypes = [e for e in FactorSubtypes]
    factorSubtypeChoices = [e.name for e in FactorSubtypes]
    factorSubtypeChoices.append(constants.NAME_ALL)

    parser = argparse.ArgumentParser("Show factors")
    parser.add_argument("-t", "--type", required=False, nargs='*', choices=factorChoices, default=constants.NAME_ALL, help="Types to display")
    parser.add_argument("-s", "--subtype", required=False, nargs='*', choices=factorSubtypeChoices, default=constants.NAME_ALL, help="Types to display")

    arguments = parser.parse_args()
    if constants.NAME_ALL in arguments.type:
        types = factorTypes
    else:
        types = [FactorTypes[e] for e in arguments.type]

    if constants.NAME_ALL in arguments.subtype:
        subtypes = factorSubtypes
    else:
        subtypes = [FactorSubtypes[e] for e in arguments.subtype]

    allFactors = Factors()
    print(f"Type :{arguments.type} subtype: {arguments.subtype} | {allFactors.getColumns(types, subtypes)}")
    # print(f"Shape: {allFactors.getColumns([FactorTypes.SHAPE])}")
    # print(f"Color: {allFactors.getColumns([FactorTypes.COLOR])}")
    # print(f"GLCM: {allFactors.getColumns([FactorTypes.TEXTURE])}")
    # print(f"LBP: {allFactors.getColumns([FactorTypes.TEXTURE])}")
    # print(f"Position: {allFactors.getColumns([constants.PROPERTY_FACTOR_POSITION])}")
    # print(f"Color and GLCM: {allFactors.getColumns([FactorTypes.COLOR, FactorTypes.TEXTURE])}")
    # print(f"All: {allFactors.getColumns(None)}")
