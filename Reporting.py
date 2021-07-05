#
# R E P O R T I N G
#

import matplotlib.pyplot as plt
import constants

class Reporting:
    def __init__(self):
        """
        A reporting instance.

        """
        self._blobs = {}
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

    def writeSummary(self, filename: str):
        file = open(filename, "w")
        blobNumber = 0
        file.write("name,number,ratio,shape,distance,normalized_distance,type\n")
        for blobName, blobAttributes in self._blobs.items():
            # Only take desired vegetation in full view
            if (blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED and blobAttributes[constants.NAME_RATIO] != 0) or \
               (blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED and blobAttributes[constants.NAME_RATIO] != 0):
                file.write("%s,%d,%f,%f,%f,%f,%d\n" %
                           (blobName,
                            blobNumber,
                            blobAttributes[constants.NAME_RATIO],
                            blobAttributes[constants.NAME_SHAPE_INDEX],
                            #blobAttributes[constants.NAME_SIZE_RATIO],
                            blobAttributes[constants.NAME_DISTANCE],
                            blobAttributes[constants.NAME_DISTANCE_NORMALIZED],
                            blobAttributes[constants.NAME_TYPE]))
                blobNumber = blobNumber + 1


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


