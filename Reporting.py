#
# R E P O R T I N G
#

import matplotlib.pyplot as plt
import constants
import logging
import numpy as np

class Reporting:
    def __init__(self):
        """
        A reporting instance.

        """
        self._blobs = {}
        self.log = logging.getLogger(__name__)
        return

    @property
    def blobs(self):
        return self._blobs

    def addBlobs(self, sequence: int,blobs: {}):
        """
        Add the blob dictionary to the global list of everything seen so far.
        :param sequence: The sequence number of this image
        :param blobs: The detected and classified items in the image
        """
        # Add the blobs to the global list of everything we've seen so far
        for blobName, blobAttributes in blobs.items():
            newName = "image-" + str(sequence) + "-" + blobName
            self._blobs[newName] = blobAttributes

    def writeSummary(self, filename: str)-> bool:
        """
        Write the data to the file specified.  We will use this data later for training.
        :param filename:  The fully qualified name of the file.
        """
        try:
            file = open(filename, "w")
        except PermissionError as p:
            self.log.error("Permission denied for file: " + filename)
            return False, "Unable to write file " + filename

        blobNumber = 1
        file.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(constants.NAME_NAME,
                                                         constants.NAME_NUMBER,
                                                         constants.NAME_RATIO,
                                                         constants.NAME_SHAPE_INDEX,
                                                         constants.NAME_DISTANCE,
                                                         constants.NAME_DISTANCE_NORMALIZED,
                                                         constants.NAME_HUE,
                                                         constants.NAME_SATURATION,
                                                         constants.NAME_I_YIQ,
                                                         constants.NAME_BLUE_DIFFERENCE,
                                                         constants.NAME_TYPE))
        for blobName, blobAttributes in self._blobs.items():
            # Only take desired vegetation in full view
            if (blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED and blobAttributes[constants.NAME_RATIO] != 0) or \
               (blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED and blobAttributes[constants.NAME_RATIO] != 0):

                try:
                    self.log.debug("Writing out training data for : " + blobName)
                    file.write("%s,%d,%f,%f,%f,%f,%f,%f,%f,%f,%d\n" %
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


