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

        # Columns to exclude from translations like normalizing values
        self._exclude = [constants.NAME_NAME, constants.NAME_NUMBER, constants.NAME_TYPE]

        # All the columns includes the factors + the identifying attributes
        self._columns.extend(self._exclude)

        self._blobDF = pd.DataFrame(columns=self._columns)
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
        for filename in [self._resultsFilename, self._normalizedFilename]:
            try:
                file = open(filename, "w")
                file.truncate(0)
                file.close()
            except PermissionError as p:
                self.log.error("Permission denied for file: " + filename)
                return False, "Unable to write file " + filename

        return True, "OK"

    def addBlobs(self, sequence: int, blobs: {}):
        """
        Add the blob dictionary to the global list of everything seen so far.
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
            attributes[constants.NAME_NAME] = newName
            # The name of the blob is blob<n>, so this should split it apart
            attributes[constants.NAME_NUMBER] = blobName.split(constants.NAME_BLOB)[1]
            # Add only the columns needed
            for attributeName, attributeValue in blobAttributes.items():
                if attributeName in self._columns:
                    attributes[attributeName] = attributeValue
            self._blobDF = self._blobDF.append(attributes, ignore_index=True)

        self._blobDF[constants.NAME_NUMBER] = range(self._startBlob, self._startBlob + len(self._blobDF))
        self._startBlob += len(self._blobDF)

        # write out the data, appending to the file if it is already there
        # Previous versions just kept everything in a dataframe and wrote it out in the end,
        # but that caused problems for large data sets
        if os.path.isfile(self._dataframeFilename):
            self._blobDF.to_csv(self._dataframeFilename, mode='a', header=False, encoding="UTF-8", index=False)
        else:
            self._blobDF.to_csv(self._dataframeFilename, encoding="UTF-8", index=False)

        self._blobDF = pd.DataFrame(columns=self._columns)
        return

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
        Write the data to the file specified.  We will use this data later for training.
        :param filename:  The fully qualified name of the file.
        """
        try:
            file = open(self._resultsFilename, "w")
        except PermissionError as p:
            self.log.error("Permission denied for file: " + self._filename)
            return False, "Unable to write file " + self._filename

        newdf = self._blobDF[(self._blobDF.type == constants.TYPE_DESIRED) | (self._blobDF.type == constants.TYPE_UNDESIRED)]
        newdf[constants.NAME_NUMBER] = range(1, len(self._blobDF) + 1)
        newdf.to_csv(self._dataframeFilename, encoding="UTF-8", index=False)

        self._normalize()
        newdf = self._blobDF[(self._blobDF.type == constants.TYPE_DESIRED) | (self._blobDF.type == constants.TYPE_UNDESIRED)]
        newdf.to_csv(self._normalizedFilename, encoding="UTF-8", index=False)
        blobNumber = 1
        file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(constants.NAME_NAME,
                                                                                          constants.NAME_NUMBER,
                                                                                          constants.NAME_RATIO,
                                                                                          constants.NAME_SHAPE_INDEX,
                                                                                          constants.NAME_DISTANCE,
                                                                                          constants.NAME_DISTANCE_NORMALIZED,
                                                                                          constants.NAME_HUE,
                                                                                          constants.NAME_SATURATION,
                                                                                          constants.NAME_I_YIQ,
                                                                                          constants.NAME_BLUE_DIFFERENCE,
                                                                                          # The new items
                                                                                          constants.NAME_COMPACTNESS,
                                                                                          constants.NAME_ELONGATION,
                                                                                          constants.NAME_ECCENTRICITY,
                                                                                          constants.NAME_ROUNDNESS,
                                                                                          constants.NAME_SOLIDITY,
                                                                                          # GLCM
                                                                                          constants.NAME_HOMOGENEITY,
                                                                                          constants.NAME_ENERGY,
                                                                                          constants.NAME_CONTRAST,
                                                                                          constants.NAME_DISSIMILARITY,
                                                                                          constants.NAME_ASM,
                                                                                          constants.NAME_CORRELATION,
                                                                                          constants.NAME_HSV_SATURATION + "_" + constants.NAME_HOMOGENEITY,
                                                                                          constants.NAME_HSV_SATURATION + "_" + constants.NAME_ENERGY,
                                                                                          constants.NAME_HSV_SATURATION + "_" + constants.NAME_CONTRAST,
                                                                                          constants.NAME_HSV_SATURATION + "_" + constants.NAME_DISSIMILARITY,
                                                                                          constants.NAME_HSV_SATURATION + "_" + constants.NAME_ASM,
                                                                                          constants.NAME_HSV_SATURATION + "_" + constants.NAME_CORRELATION,
                                                                                          # Type of the blob
                                                                                          constants.NAME_TYPE))
        for blobName, blobAttributes in self._blobs.items():
            # Only take desired vegetation in full view
            # if (blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED and blobAttributes[constants.NAME_RATIO] != 0) or \
            #    (blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED and blobAttributes[constants.NAME_RATIO] != 0):
            # get everything irrespective of being fully in view
            if blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED or blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED:
                try:
                    self.log.debug("Writing out training data for : " + blobName)
                    file.write("%s,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d\n" %
                               (blobName,
                                blobNumber,
                                blobAttributes[constants.NAME_RATIO],
                                blobAttributes[constants.NAME_SHAPE_INDEX],
                                #blobAttributes[constants.NAME_SIZE_RATIO],
                                blobAttributes[constants.NAME_DISTANCE],
                                blobAttributes[constants.NAME_DISTANCE_NORMALIZED],
                                blobAttributes[constants.NAME_HUE],
                                blobAttributes[constants.NAME_SATURATION],
                                blobAttributes[constants.NAME_I_YIQ],
                                blobAttributes[constants.NAME_BLUE_DIFFERENCE],
                                blobAttributes[constants.NAME_COMPACTNESS],
                                blobAttributes[constants.NAME_ELONGATION],
                                blobAttributes[constants.NAME_ECCENTRICITY],
                                blobAttributes[constants.NAME_ROUNDNESS],
                                blobAttributes[constants.NAME_SOLIDITY],
                                blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_HOMOGENEITY + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ENERGY + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CONTRAST + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_DISSIMILARITY + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_ASM + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_GREYSCALE + constants.DELIMETER + constants.NAME_CORRELATION + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_HSV_SATURATION + "_" + constants.NAME_HOMOGENEITY + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_HSV_SATURATION + "_" + constants.NAME_ENERGY + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_HSV_SATURATION + "_" + constants.NAME_CONTRAST + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_HSV_SATURATION + "_" + constants.NAME_DISSIMILARITY + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_HSV_SATURATION + "_" + constants.NAME_ASM + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_HSV_SATURATION + "_" + constants.NAME_CORRELATION + constants.DELIMETER + "avg"],
                                blobAttributes[constants.NAME_TYPE]))
                except ValueError:
                    self.log.error("Error in writing feature data.")
                except KeyError as e:
                    self.log.error("Can't find value for: ")
                    self.log.error(e)
                    print(e)
                blobNumber = blobNumber + 1

        file.close()
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


