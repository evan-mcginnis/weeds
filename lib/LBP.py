import argparse
import cv2
import numpy as np

import logging
import os

from skimage import feature

import constants

class LBP:

    def __init__(self, blobs: {}, selectedImage: str, numPoints: int, radius: int, **kwargs):
        """
        Initialize the local binary pattern from the specified image.
        The image must be already converted to greyscale
        :param blobs: dictionary of all blobs
        :param selectedImage:
        """
        self._blobs = blobs
        self._selectedImage = selectedImage
        self._image = None
        self._numPoints = numPoints
        self._radius = radius
        self._histogram = None
        self._prefix = constants.NAME_LBP
        self._log = logging.getLogger(constants.NAME_LBP)

        for key, value in kwargs.items():
            if key == constants.KEYWORD_IMAGE:
                self._image = value
            elif key == constants.KEYWORD_PREFIX:
                self._prefix = value
            elif key == constants.KEYWORD_OUTPUT:
                self._outputDirectory = value
            else:
                raise TypeError(f"Unknown keyword: {key}")

    @property
    def histogram(self) -> np.histogram:
        return self._histogram

    @property
    def blobs(self) -> {}:
        return self._blobs

    def _computeHistogram(self, image: np.ndarray) -> np.histogram:
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image,
                                           self._numPoints,
                                           self._radius,
                                           method="uniform")

        # However, the lbp variable returned by the local_binary_patterns function is not directly usable as a
        # feature vector. Instead, lbp is a 2D array with the same width and height as our input image — each of
        # the values inside lbp ranges from [0, numPoints + 2], a value for each of the possible numPoints + 1
        # possible rotation invariant prototypes (see the discussion of uniform patterns at the top of this post
        # for more information) along with an extra dimension for all patterns that are not uniform, yielding a
        # total of numPoints + 2 unique possible values.
        (self._histogram, _) = np.histogram(lbp.ravel(),
                                            bins=np.arange(0, self._numPoints + 3),
                                            range=(0, self._numPoints + 2))
        # normalize the histogram
        self._histogram = self._histogram.astype("float")

        # Prevent division by xero errors by adding a tiny value
        eps = 1e-7
        self._histogram /= (self._histogram.sum() + eps)
        # return the histogram of Local Binary Patterns
        return self._histogram

    def compute(self):
        """
        Compute the LBP
        :return:
        """

        if self._image is None and self._selectedImage is None:
            raise ValueError("Image or selected blob must be specified")

        # Draws heavily from:
        # https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/

        # Debug
        if self._image is not None:
            image = cv2.imread(self._image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self._histogram = self._computeHistogram(gray)


        # Process the actual blobs
        else:
            for blobName, blobAttributes in self._blobs.items():
                #img = np.empty_like(blobAttributes[self._selectedImage])
                img = blobAttributes[self._selectedImage]
                if len(img) > 0:
                    try:
                        self._histogram = self._computeHistogram(img)

                    except ValueError as v:
                        self._log.error(f"Problems computing LBP for image with shape: {img.shape}")
                        self._log.error(v)
                        name = self._prefix + constants.DELIMETER + constants.NAME_STDDEV
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_MEAN
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_VAR
                        blobAttributes[name] = 0
                        continue

                    lbpNonZero = np.nonzero(self._histogram)
                    if not len(lbpNonZero) > 0:
                        self._log.error("LBP is empty")
                        name = self._prefix + constants.DELIMETER + constants.NAME_STDDEV
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_MEAN
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_VAR
                        blobAttributes[name] = 0
                        continue
                    #print(f"Std Deviation: {hogNonZero.std()}")

                    name = self._prefix + constants.DELIMETER + constants.NAME_STDDEV
                    blobAttributes[name] = self._histogram.std()
                    self._log.debug(f"{name}: {blobAttributes[name]}")
                    name = self._prefix + constants.DELIMETER + constants.NAME_MEAN
                    blobAttributes[name] = self._histogram.mean()
                    self._log.debug(f"{name}: {blobAttributes[name]}")
                    name = self._prefix + constants.DELIMETER + constants.NAME_VAR
                    blobAttributes[name] = self._histogram.var()
                    self._log.debug(f"{name}: {blobAttributes[name]}")
                else:
                    self._log.error("Empty blob encountered in LBP calculations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Local Binary Pattern")

    parser.add_argument("-i", "--input", action="store", required=True, help="Image to process")
    parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
    arguments = parser.parse_args()

    #startupLogger(arguments.logging)

    pattern = LBP(None, "", 24, 8, image=arguments.input, output=arguments.output)
    pattern.compute()


