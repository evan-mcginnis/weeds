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

    def classifyWithinCropRow(self):
        return

    def classifyByRatio(self, largest: int, size : (),ratio: int):
        """
        Classify blobs in the image my size ratio.
        :param size: size of the images
        :param largest: the size in pixels of the blob with the largest area
        :param ratio: the threshold ratio
        """
        (maxY, maxX, depth) = size
        try:
            largestArea = self._blobs[largest].get(constants.NAME_AREA)
            for blobName, blobAttributes in self._blobs.items():

                # Only for items that have not already been classified

                if blobAttributes[constants.NAME_TYPE] == constants.TYPE_UNKNOWN:
                    # If the area of the blob is much smaller than the largest object, it must be undesirable
                    if largestArea > ratio * blobAttributes.get(constants.NAME_AREA):

                        # But, if the view of this is only partial in that it is at the edge of the image,
                        # we can't say with confidence that it is
                        (x, y, w, h) = blobAttributes.get(constants.NAME_LOCATION)
                        if(x == 0 or x+w >= maxX):
                            blobAttributes[constants.NAME_TYPE] = constants.TYPE_UNKNOWN
                        else:
                            blobAttributes[constants.NAME_TYPE] = constants.TYPE_UNDESIRED
                    else:
                        blobAttributes[constants.NAME_TYPE] = constants.TYPE_DESIRED
        except KeyError:
            return


