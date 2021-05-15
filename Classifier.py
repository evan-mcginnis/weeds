#
# C L A S S I F I E R
#

import constants

class Classifier:

    def __init__(self, blobs: {}):
        self._blobs = blobs
        return

    @property
    def blob(self):
        return self._blobs

    def classifyByRatio(self, largest: int, ratio: int):
        """
        Classify blobs in the image my size ratio.
        :param largest:
        :param ratio:
        """
        largestArea = self._blobs[largest].get("area")
        for blobName, blobAttributes in self._blobs.items():
            if largestArea > ratio * blobAttributes.get("area"):
                blobAttributes["type"] = constants.TYPE_UNDESIRED
            else:
                blobAttributes[constants.NAME_TYPE] = constants.TYPE_DESIRED



