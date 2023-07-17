#
# G L C M
#
import logging

from skimage.feature import graycomatrix, graycoprops
from skimage import exposure
from skimage import io
import cv2
import numpy as np

import constants

DISTANCE = 1
NUMBER_OF_PIXELS = 5
ANGLE = 0

class GLCM:
    attributes = [constants.NAME_ENERGY,
                  constants.NAME_CONTRAST,
                  constants.NAME_DISSIMILARITY,
                  constants.NAME_HOMOGENEITY,
                  constants.NAME_ASM,
                  constants.NAME_CORRELATION]

    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    angleNames = ["0", "45", "90", "135", "180", "avg"]

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



    @property
    def blobs(self):
        return self._blobs

    def computeAttributes(self):
        """
        Compute all GLCM attributes and insert them into the blob list
        """
        for blobName, blobAttributes in self._blobs.items():
            for glcmAttribute in GLCM.attributes:

                # Use a sample of the image
                #(x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
                # (x, y) = blobAttributes[constants.NAME_CENTER]
                # h = NUMBER_OF_PIXELS
                # img = self._image[y:y+h, x:x+h]

                # Works: Use the entire portion of the blob
                #img = blobAttributes[constants.NAME_GREYSCALE_IMAGE]
                img = np.empty_like(blobAttributes[self._selectedImage])
                if self._selectedBand == -1:
                    img = blobAttributes[self._selectedImage]
                else:
                    img = blobAttributes[self._selectedImage][self._selectedBand]

                # Just compute these for a single angle and number of pixels
                if len(img) > 0:
                    if img.dtype == np.int16:
                        # self._log.debug(f"Data is int16: min: {np.amin(img)} max: {np.amax(img)}")
                        correlationMatrix = graycomatrix(img, [DISTANCE], GLCM.angles, levels=256)
                    elif img.dtype == np.float64:
                        # self._log.warning(f"Data is float. Converting. min: {np.amin(img)} max: {np.amax(img)}")
                        imgRescaled = exposure.rescale_intensity(img, out_range=(0, 1))
                        imgAsInt = imgRescaled.astype(np.int)
                        correlationMatrix = graycomatrix(imgAsInt, [DISTANCE], GLCM.angles, levels=256)
                    else:
                        correlationMatrix = graycomatrix(img, [DISTANCE], GLCM.angles)
                    attribute = graycoprops(correlationMatrix, glcmAttribute)[0]
                    # Walk through all the angles
                    total = 0
                    for angle in range(len(GLCM.angles)):
                        # Insert the angle/attribute into the result table: i.e., yiq_i_90_energy
                        name = self._prefix + glcmAttribute + constants.DELIMETER + self.angleNames[angle]
                        blobAttributes[name] = attribute[angle]
                        total += attribute[angle]
                        # This is quite noisy
                        # self._log.debug(f"GLCM {name}: {attribute[angle]}")

                    # Add in the average to make the observation rotationally independant
                    name = self._prefix + glcmAttribute + constants.DELIMETER + constants.NAME_AVERAGE
                    blobAttributes[name] = total / len(GLCM.angleNames)

                else:
                    self._log.warning(f"GLCM: Empty array {np.shape(img)}")
                    blobAttributes[glcmAttribute] = float("nan")

        return
