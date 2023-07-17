#
# F A C T O R S
#
import numpy as np

import constants
import pandas as pd
from GLCM import GLCM

class Factors:
    def __init__(self):

        # Angles used for factor where that is relevant -- now that is just GLCM
        self._angles = [0, np.pi/4, np.pi/2, 3 * np.pi/4, np.pi]

        # This is one longer than the angles so we can have the average as well
        self._angleNames = ["", "45", "90", "135", "180", constants.NAME_AVERAGE]

        # The final list of factors -- the ones here do not have angles associated with them
        allFactors = [
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_SHAPE_INDEX],
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_COMPACTNESS],
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_CONVEXITY],
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_ELONGATION],
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_ECCENTRICITY],
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_ROUNDNESS],
            [constants.PROPERTY_FACTOR_SHAPE, constants.NAME_SOLIDITY],
            [constants.PROPERTY_FACTOR_COLOR, constants.NAME_HUE],
            [constants.PROPERTY_FACTOR_COLOR, constants.NAME_SATURATION],
            [constants.PROPERTY_FACTOR_COLOR, constants.NAME_I_YIQ],
            [constants.PROPERTY_FACTOR_COLOR, constants.NAME_BLUE_DIFFERENCE],
            [constants.PROPERTY_FACTOR_POSITION, constants.NAME_DISTANCE],
            [constants.PROPERTY_FACTOR_POSITION, constants.NAME_DISTANCE_NORMALIZED],
            # TODO: Create a new category, something like PROPERTY_FACTOR_STATS for both GLCM and HOG
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HOG + constants.DELIMETER + constants.NAME_STDDEV],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HOG + constants.DELIMETER + constants.NAME_MEAN],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HOG + constants.DELIMETER + constants.NAME_VAR]
        ]

        # The factors that need to be expanded with their angles
        _factorsWithAngles = [
            # Greyscale
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CORRELATION],
            # YIQ - Y
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Y + constants.DELIMETER + constants.NAME_CORRELATION],
            # YIQ - I
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_I + constants.DELIMETER + constants.NAME_CORRELATION],
            # YIQ - Q
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YIQ_Q + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSV - Hue
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_HUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSV - Saturation
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSV - Value
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSV_VALUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSI - Hue
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_HUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSI - Saturation
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_SATURATION + constants.DELIMETER + constants.NAME_CORRELATION],
            # HSI - Intensity
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_HSI_INTENSITY + constants.DELIMETER + constants.NAME_CORRELATION],
            # Blue
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_BLUE + constants.DELIMETER + constants.NAME_CORRELATION],
            # Green
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_ASM],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_GREEN + constants.DELIMETER + constants.NAME_CORRELATION],
            # Red
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_RED + constants.DELIMETER + constants.NAME_ASM],
            # YCBCR - Y
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_LUMA + constants.DELIMETER + constants.NAME_ASM],
            # YCBCR - CB
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_BLUE_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM],
            # YCBCR - CR
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_YCBCR_RED_DIFFERENCE + constants.DELIMETER + constants.NAME_ASM],
            # CIELAB - L
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_L + constants.DELIMETER + constants.NAME_ASM],
            # CIELAB - A
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_A + constants.DELIMETER + constants.NAME_ASM],
            # CIELAB - B
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CORRELATION],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_HOMOGENEITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ENERGY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_CONTRAST],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_DISSIMILARITY],
            [constants.PROPERTY_FACTOR_GLCM, constants.NAME_CIELAB_B + constants.DELIMETER + constants.NAME_ASM],
        ]
        _columnNames = [constants.COLUMN_NAME_TYPE, constants.COLUMN_NAME_FACTOR]

        # Add in the GLCM factors
        for factor in _factorsWithAngles:
            for angleName in GLCM.angleNames:
                factorWithAngleName = [factor[0], factor[1] + constants.DELIMETER + angleName]
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

    def getColumns(self, factorType: []) -> []:
        """
        Get the factors of the types provided
        :param factorType: Array of factor types
        :return: Array of factors
        """
        subset = []
        for index, row in self._data.iterrows():
            if factorType is not None:
                if row[constants.COLUMN_NAME_TYPE] in factorType:
                    subset.append(row[constants.COLUMN_NAME_FACTOR])
            else:
                subset.append(row[constants.COLUMN_NAME_FACTOR])
        return subset

if __name__ == "__main__":
    allFactors = Factors()
    print(f"Shape: {allFactors.getColumns([constants.PROPERTY_FACTOR_SHAPE])}")
    print(f"Color: {allFactors.getColumns([constants.PROPERTY_FACTOR_COLOR])}")
    print(f"GLCM: {allFactors.getColumns([constants.PROPERTY_FACTOR_GLCM])}")
    print(f"Position: {allFactors.getColumns([constants.PROPERTY_FACTOR_POSITION])}")
    print(f"Color and GLCM: {allFactors.getColumns([constants.PROPERTY_FACTOR_COLOR, constants.PROPERTY_FACTOR_GLCM])}")
    print(f"All: {allFactors.getColumns(None)}")
