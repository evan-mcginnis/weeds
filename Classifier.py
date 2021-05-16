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

    def classifyByRatio(self, largest: int, size : (),ratio: int):
        """
        Classify blobs in the image my size ratio.
        :param size: size of the images
        :param largest: the size in pixels of the blob with the largest area
        :param ratio: the threshold ratio
        """
        (maxY, maxX, depth) = size
        largestArea = self._blobs[largest].get("area")
        for blobName, blobAttributes in self._blobs.items():
            # If the area of the blob is much smaller than the largest object, it must be undesirable
            if largestArea > ratio * blobAttributes.get("area"):

                # But, if the view of this is only partial in that it is at the edge of the image,
                # we can't say with confidence that it is
                (x, y, w, h) = blobAttributes.get(constants.NAME_LOCATION)
                if(x == 0 or x+w >= maxX):
                    blobAttributes["type"] = constants.TYPE_UNKNOWN
                else:
                    blobAttributes["type"] = constants.TYPE_UNDESIRED
            else:
                blobAttributes[constants.NAME_TYPE] = constants.TYPE_DESIRED



