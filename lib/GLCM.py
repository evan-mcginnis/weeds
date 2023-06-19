#
# G L C M
#
import logging

from skimage.feature import graycomatrix, graycoprops
from skimage import io
import cv2
import numpy as np

import constants

DISTANCE = 1
NUMBER_OF_PIXELS = 5
ANGLE = 0

class GLCM:
    def __init__(self, blobs: {}, selectedImage: str, **kwargs):
        """
        GLCM
        :param blobs: Image blobs
        :param selectedImage: Name of the attribute containing the image data
        """
        self._blobs = blobs
        self._selectedImage = selectedImage
        self._selectedBand = -1
        self._prefix = ""
        self._log = logging.getLogger("GLCM")

        for key, value in kwargs.items():
            if key == constants.KEYWORD_BAND:
                self._selectedBand = int(value)
            elif key == constants.KEYWORD_PREFIX:
                self._prefix = value
            else:
                raise TypeError(f"Unknown keyword: {key}")


        if len(self._prefix) > 0:
            self._prefix += constants.DELIMETER


        self._GCLMAttributes = [constants.NAME_ENERGY,
                                constants.NAME_CONTRAST,
                                constants.NAME_DISSIMILARITY,
                                constants.NAME_HOMOGENEITY,
                                constants.NAME_ASM,
                                constants.NAME_CORRELATION]

    @property
    def blobs(self):
        return self._blobs

    def computeAttributes(self):
        """
        Compute all GLCM attributes and insert them into the blob list
        """
        for blobName, blobAttributes in self._blobs.items():
            for glcmAttribute in self._GCLMAttributes:

                # Use a sample of the image
                #(x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
                # (x, y) = blobAttributes[constants.NAME_CENTER]
                # h = NUMBER_OF_PIXELS
                # img = self._image[y:y+h, x:x+h]

                # Works: Use the entire portion of the blob
                #img = blobAttributes[constants.NAME_GREYSCALE_IMAGE]
                # Use the image name passed in
                if self._selectedBand == -1:
                    img = blobAttributes[self._selectedImage]
                else:
                    img = blobAttributes[self._selectedImage][self._selectedBand]

                # Just compute these for a single angle and number of pixels
                if len(img) > 0:
                    corelationMatrix = graycomatrix(img, [DISTANCE], [ANGLE])
                    attribute = graycoprops(corelationMatrix, glcmAttribute)[0]
                    blobAttributes[self._prefix + glcmAttribute] = attribute[0]
                    self._log.debug(f"GLCM {self._prefix + glcmAttribute}: {attribute}")
                else:
                    self._log.warning(f"GLCM: Empty array {np.shape(img)}")
                    blobAttributes[glcmAttribute] = float("nan")

        return
