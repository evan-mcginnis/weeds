#
# R E P O R T I N G
#
import os.path

import matplotlib.pyplot as plt
import constants
import logging
import numpy as np
import pandas as pd
import pandas.core
from Factors import Factors
from Factors import FactorKind
from Context import Context

class Reporting:
    def __init__(self, filename: str):
        """
        Reporting -- mostly used for writing out results
        :param filename: The base filename with no extension
        """
        self._blobs = {}
        self.log = logging.getLogger(__name__)
        self._filename = filename
        self._resultsFilename = self._filename + constants.EXTENSION_CSV
        self._normalizedFilename = self._filename + constants.DELIMETER + constants.FILENAME_NORMALIZED + constants.EXTENSION_CSV
        self._dataframeFilename = self._filename + constants.DELIMETER + constants.FILENAME_DATAFRAME + constants.EXTENSION_CSV
        self._dataframeFilenameVectors = self._filename + constants.DELIMETER + constants.FILENAME_DATAFRAME + constants.DELIMETER + constants.VECTOR + constants.EXTENSION_CSV

        self._columns = [constants.NAME_NAME,
                         constants.NAME_NUMBER,
                         constants.NAME_RATIO,
                         constants.NAME_SHAPE_INDEX,
                         constants.NAME_DISTANCE,
                         constants.NAME_DISTANCE_NORMALIZED,
                         constants.NAME_HUE,
                         constants.NAME_SATURATION,
                         constants.NAME_I_YIQ,
                         constants.NAME_BLUE_DIFFERENCE,
                         constants.NAME_COMPACTNESS,
                         constants.NAME_ELONGATION,
                         constants.NAME_ECCENTRICITY,
                         constants.NAME_ROUNDNESS,
                         constants.NAME_SOLIDITY,
                         constants.NAME_TYPE,
                         constants.NAME_HOMOGENEITY,
                         constants.NAME_ENERGY,
                         constants.NAME_CONTRAST,
                         constants.NAME_DISSIMILARITY,
                         constants.NAME_ASM,
                         constants.NAME_CORRELATION,
                         # Rename the prefix from I_YIQ to HSV
                         # constants.NAME_I_YIQ + "_" + constants.NAME_HOMOGENEITY,
                         constants.NAME_HSV_HUE + "_" + constants.NAME_HOMOGENEITY,
                         constants.NAME_HSV_HUE + "_" + constants.NAME_ENERGY,
                         constants.NAME_HSV_HUE + "_" + constants.NAME_CONTRAST,
                         constants.NAME_HSV_HUE + "_" + constants.NAME_DISSIMILARITY,
                         constants.NAME_HSV_HUE + "_" + constants.NAME_ASM,
                         constants.NAME_HSV_HUE + "_" + constants.NAME_CORRELATION]

        factors = Factors()

        # All the factors
        self._columns = factors.getColumns([], [])

        # Just the vectors
        self._columnsVectors = factors.getColumns([], [], FactorKind.VECTOR)

        # Columns to exclude from translations like normalizing values
        self._exclude = [constants.NAME_NAME, constants.NAME_NUMBER, constants.NAME_TYPE, constants.NAME_AGL]

        # All the columns includes the factors + the identifying attributes
        self._columns.extend(self._exclude)
        self._columnsVectors.extend(self._exclude)

        self._blobDF = pd.DataFrame(columns=self._columns)
        self._blobVectorDF = pd.DataFrame(columns=self._columnsVectors)
        self._startBlob = 0

        return

    @property
    def blobs(self):
        return self._blobs

    def initialize(self) -> (bool, str):
        """
        Initialize reporting by conforming access to and truncating files
        :return: (bool, str)
        """
        for filename in [self._resultsFilename, self._normalizedFilename, self._dataframeFilename]:
            if os.path.isfile(filename):
                return False, f"Results file {filename} exists. Will not overwrite."

        return True, "OK"


    def addBlobs(self, sequence: int, ctx: Context, blobs: {}):
        """
        Add the blob dictionary to the global list of everything seen so far.
        :param ctx: Context for image
        :param sequence: The sequence number of this image
        :param blobs: The detected and classified items in the image
        """
        # Add the blobs to the global list of everything we've seen so far
        for blobName, blobAttributes in blobs.items():
            newName = "image-" + str(sequence) + "-" + blobName
            # This is unbounded list without any purpose, it would seem.
            # The system runs out of memory as a result
            # self._blobs[newName] = blobAttributes
            attributes = {}
            attributes[constants.NAME_DATE] = ctx.datestamp
            attributes[constants.NAME_NAME] = newName
            # The name of the blob is blob<n>, so this should split it apart
            attributes[constants.NAME_NUMBER] = blobName.split(constants.NAME_BLOB)[1]
            vectorAttributes = attributes.copy()
            # Add only the columns needed
            for attributeName, attributeValue in blobAttributes.items():
                # Scalars
                if attributeName in self._columns:
                    attributes[attributeName] = attributeValue
                # Vectors
                elif attributeName in self._columnsVectors:
                    vectorAttributes[attributeName] = attributeValue
            # Scalars
            self._blobDF = self._blobDF.append(attributes, ignore_index=True)
            # Vectors
            self._blobVectorDF = self._blobVectorDF.append(vectorAttributes, ignore_index=True)

        self._blobDF[constants.NAME_NUMBER] = range(self._startBlob, self._startBlob + len(self._blobDF))
        self._blobVectorDF[constants.NAME_NUMBER] = range(self._startBlob, self._startBlob + len(self._blobDF))
        self._startBlob += len(self._blobDF)

        # write out the data, appending to the file if it is already there
        # Previous versions just kept everything in a dataframe and wrote it out in the end,
        # but that caused problems for large data sets
        if os.path.isfile(self._dataframeFilename):
            self._blobDF.to_csv(self._dataframeFilename, mode='a', header=False, encoding="UTF-8", index=False)
        else:
            self._blobDF.to_csv(self._dataframeFilename, encoding="UTF-8", index=False)

        if os.path.isfile(self._dataframeFilenameVectors):
            self._blobVectorDF.to_csv(self._dataframeFilenameVectors, mode='a', header=False, encoding="UTF-8", index=False)
        else:
            self._blobVectorDF.to_csv(self._dataframeFilenameVectors, encoding="UTF-8", index=False)

        self._blobDF = pd.DataFrame(columns=self._columns)
        self._blobVectorDF = pd.DataFrame(columns=self._columnsVectors)

    def _normalize(self):
        """
        Normalize the data
        """
        # apply normalization techniques
        for column in self._columns:
            if column not in self._exclude:
                try:
                    minimum = self._blobDF[column].min()
                    maximum = self._blobDF[column].max()
                    self.log.debug(f"Checking: {column}")
                    if type(minimum) == pd.Series:
                        if minimum[0] == maximum[0]:
                            self.log.error(f"Normalize (Series): Min == Max for {column}")
                            self._blobDF[column] = 1
                    elif minimum == maximum:
                        self.log.error(f"Normalize: Min == Max for {column}")
                        self._blobDF[column] = 1
                    else:
                        self._blobDF[column] = (self._blobDF[column] - self._blobDF[column].min()) / (self._blobDF[column].max() - self._blobDF[column].min())
                except ZeroDivisionError:
                    self.log.error("Division by zero error for column {}".format(column))
                    self._blobDF[column] = 1
        return

    def writeFactors(self) -> (bool, str):

        with open(self._filename, "w") as file:
            # The header row
            pass



    def writeSummary(self) -> (bool, str):
        """
        W A R N I N G
        This is deprecated, as it is the approach of writing out things at the end instead of as we go along

        Write the data to the file specified.  We will use this data later for training.
        :param filename:  The fully qualified name of the file.
        """
        return True, "Data written successfully."




    def showHistogram(self, title: str, bins: int, feature: str):
        """
        Display the specified histogram for the feature.
        :param feature:
        See constants for NAME_* items
        """
        # Just return if something is specified not in the blob dictionary
        if feature not in constants.names:
            return
        features = []

        for blobName, blobAttributes in self._blobs.items():
            attribute = blobAttributes[feature]
            features.append(attribute)

        if feature == constants.NAME_AREA:
            plt.hist(features, bins, facecolor='blue')
            plt.title("Areas of blobs")
            plt.xlabel("Area in pixels")
            plt.ylabel("Count of blobs")
            plt.show()
        elif feature == constants.NAME_SHAPE_INDEX:
            plt.hist(features, bins, facecolor='blue')
            plt.title("Shape Index of blobs")
            plt.xlabel("Index value")
            plt.ylabel("Count of index")
            plt.show()


