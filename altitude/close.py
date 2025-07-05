#
# P R O X I M I T Y
#
import math
import os.path
from pathlib import Path
import logging
import constants
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger

import cv2 as cv

class Proximity:
    def __init__(self, pathToImage, pathToDebug):
        self._pathToImage = pathToImage
        self._pathToDebug = pathToDebug
        self._closestBlobs = []
        self._log = logging.getLogger(constants.NAME_PROXIMITY)
        self._lids = None

        if not os.path.isfile(pathToImage):
            raise FileNotFoundError(f"Unable to access: {pathToImage}")
        if not os.path.isdir(pathToDebug):
            raise NotADirectoryError(f"Unable to access directory: {pathToDebug}")
        self._logger = ImageLogger()
        if not self._logger.connect(arguments.output):
            self._log.error("Unable to connect to logging. {} does not exist.".format(arguments.output))


    @property
    def lids(self) -> []:
        """
        The lids found in the image
        :return:
        """
        return self._lids

    def _findLids(self):
        """
        Find the number of lids in the image and create a list of centers of the lids
        """
        # The lids haven't been located
        if self._lids is None:
            vi = VegetationIndex()
            vi.Load(arguments.input)
            rawMask = vi.bucketLid()
            threshold = 0.0
            negate = False
            direction = 1
            mask, thresholdUsed = vi.createMask(rawMask, negate, direction, threshold)
            vi.applyMask()
            image = vi.GetImage()
            # Debug
            cv.imwrite(f"{arguments.output}/{Path(self._pathToImage).stem}-bi.jpg", image)

            manipulatedImage = ImageManipulation(image, 0, self._logger)

            # Find the blobs in the image -- there should be only the lid
            # For now, this only works with a single lid
            # The threshold for area is completely arbitrary -- as we really only care about the bucket lid, this may
            # need to be changed to reflect what is seen at higher distances AGL
            contours, hierarchy, blobs, largest = manipulatedImage.findBlobs(10000, constants.Strategy.CARTOON)

            self._lids = blobs


    def hasDiskInImage(self) -> bool:
        """
        If there was a disk/lid found in the current image
        :return:
        """
        return len(self._lids) > 0

    def closestBlobs(self, blobs: {}) -> []:
        """
        The name of the blob that is the closest to the center of the blue disk
        The list of blob names is in the same order as was passed in originally
        :param blobs:
        :return:
        """
        # Find the center of the blue disks
        self._findLids()
        for lidName, lidProperties in self._lids.items():
            (cXLid, cYLid) = lidProperties[constants.NAME_CENTER]
            self._log.debug(f"Lid Center: ({cXLid},{cYLid})")
            shortestDistance = 999
            shortestBlobName = ""
            for blobName, blobProperties in blobs.items():
                (cXPlant, cYPlant) = blobProperties[constants.NAME_CENTER]
                distanceToBlob = math.sqrt((cXLid - cXPlant)**2 + (cYLid - cYPlant)**2)
                self._log.debug(f"Plant Center: ({cXPlant},{cYPlant} Distance to disk: {distanceToBlob})")
                if distanceToBlob < shortestDistance:
                    shortestDistance = distanceToBlob
                    shortestBlobName = blobName
            self._closestBlobs.append(shortestBlobName)

        return self._closestBlobs

if __name__ == "__main__":
    import os
    import sys
    import logging
    import logging.config
    import warnings
    import argparse
    import constants

    from ImageManipulation import ImageManipulation
    from VegetationIndex import VegetationIndex
    from Classifier import Classifier


    veg = VegetationIndex()

    parser = argparse.ArgumentParser("Find closest blob to disk")

    parser.add_argument("-i", "--input", action="store", required=True, help="Image to process")
    parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
    parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm", choices=veg.GetSupportedAlgorithms(), default="com2")
    # Mask options
    parser.add_argument("-t", "--threshold", action="store", required=False, help="Threshold for index mask")
    parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
    parser.add_argument("-di", "--direction", action="store", type=int, required=False, default=1, help="Direction forindex mask")
    parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
    # Execution options
    parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
    arguments = parser.parse_args()

    if not os.path.isfile(arguments.input):
        print(f"Unable to access input image: {arguments.input}")
        sys.exit(-1)

    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(-1)

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("proximity")

    target = Proximity(arguments.input, arguments.output)

    # Locate vegetation in the image

    # Read the image from disk
    rawImage = cv.imread(arguments.input, cv.IMREAD_COLOR)

    # Create the index
    veg.SetImage(rawImage)
    index = veg.Index(arguments.algorithm)
    # if the index is refined, OTSU works for indices it didn't before
    index = veg.refine()

    # Determine the threshold to use
    if arguments.threshold is not None:
        try:
            thresholdForMask = float(arguments.threshold)
        except ValueError:
            # Must be OTSU or TRIANGLE
            if arguments.threshold in VegetationIndex.thresholdChoices:
                thresholdForMask = arguments.threshold
            else:
                print(f"Threshold must be one of {VegetationIndex.thresholdChoices} or a valid float")
                sys.exit(-1)
    else:
        thresholdForMask = None

    # Create the mask & apply it
    mask, threshold = veg.createMask(index, arguments.nonegate, arguments.direction, thresholdForMask)
    #log.debug(f"Use threshold: {threshold}")
    logger = ImageLogger()
    if not logger.connect(arguments.output):
        print("Unable to connect to logging. {} does not exist.".format(arguments.output))
        sys.exit(-1)

    veg.applyMask()

    image = veg.GetImage()
    finalImage = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX)

    manipulated = ImageManipulation(finalImage, 0, logger)

    # Find the vegetation in the image
    contours, hierarchy, blobs, largest = manipulated.findBlobs(arguments.minarea, constants.Strategy.CARTOON)

    # Debug
    heuristicClassifier = Classifier()
    heuristicClassifier.blobs = blobs
    heuristicClassifier.classifyAs(constants.TYPE_WEED)
    # manipulatedImage.drawContours()
    manipulated.drawBoxes("Plants", heuristicClassifier.blobs, [constants.NAME_NAME])
    classifiedImage = manipulated.image
    cv.imwrite(f"{arguments.output}/{Path(arguments.input).stem}-classified.jpg", classifiedImage)

    if len(blobs) == 0:
        log.error(f"Unable to find vegetation in the image: {arguments.input}")
        sys.exit(-1)

    # Find the list blobs that are closest to disks
    log.info(f"Closest blobs to lids in {arguments.input}: {target.closestBlobs(blobs)}")

    # There is a bug in opencv where files are not closed when read, resulting in a warning about an unclosed file
    # This is a bit of a hack to supress all the warnings on exit(). This is safe to remove, but the warning is a bit annoying
    warnings.simplefilter("ignore")

    sys.exit(0)




