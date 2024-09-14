import argparse
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import logging
import os

import constants




class HOG:

    def __init__(self, blobs: {}, selectedImage: str, **kwargs):
        """
        GLCM
        :param blobs: Image blobs
        :param selectedImage: Name of the attribute containing the image data
        """
        self._blobs = blobs
        self._selectedImage = selectedImage
        self._selectedBand = -1
        self._prefix = constants.NAME_HOG
        self._log = logging.getLogger(constants.NAME_HOG)
        self._image = None
        self._outputDirectory = "."

        for key, value in kwargs.items():
            if key == constants.KEYWORD_IMAGE:
                self._image = value
            elif key == constants.KEYWORD_PREFIX:
                self._prefix = value
            elif key == "output":
                self._outputDirectory = value
            else:
                raise TypeError(f"Unknown keyword: {key}")

    @property
    def blobs(self):
        return self._blobs

    def _computeDescriptor(self, img: np.ndarray) -> np.ndarray:
        pass

    def computeAttributes(self):

        # This is just for debugging purposes to compute the HOG descriptor for a file on disk
        if self._image is not None:
            img = cv2.imread(self._image)
            img = np.float32(img) / 255.0

            # Calculate gradient
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
            # Calculate the magnitude
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

            imageName = self._outputDirectory + "/" + "magnitude" + constants.DELIMETER + os.path.basename(self._image)
            normalized = np.zeros_like(mag)
            finalImage = cv2.normalize(mag, normalized, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(imageName, finalImage)
            cv2.namedWindow("Magnitude", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Gradient X", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Gradient Y", cv2.WINDOW_NORMAL)

            cv2.imshow("Gradient X", gx)
            cv2.imshow("Gradient Y", gy)
            cv2.imshow("Magnitude", mag)

            imgAsGreyscale = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            img = cv2.imread(self._image)
            dst = cv2.Canny(img, 175, 200, apertureSize=3, L2gradient=True)
            cv2.imwrite("edges.jpg", dst)
            cv2.imshow("Edges", dst)

            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(6, 6), cells_per_block=(2, 2), visualize=True, channel_axis=-1)

            hogNonZero = np.nonzero(fd)[0]
            print(f"Std Deviation: {hogNonZero.std()}")

            #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1), out_range=(0 ,255))
            hog_image_rescaled = exposure.rescale_intensity(hog_image, out_range=(0, 2048))

            imageName = self._outputDirectory + "/" + constants.NAME_HOG + constants.DELIMETER + os.path.basename(self._image)
            cv2.imwrite(imageName, hog_image_rescaled)
            #cv2.imshow("HOG", hog_image_rescaled)
            cv2.waitKey(0)

        else:
            for blobName, blobAttributes in self._blobs.items():
                #img = np.empty_like(blobAttributes[self._selectedImage])
                img = blobAttributes[self._selectedImage]
                if len(img) > 0:
                    # creating hog features
                    # fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                    #                     cells_per_block=(2, 2), visualize=True, multichannel=True)
                    try:
                        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True, channel_axis=-1)
                    except ValueError as v:
                        self._log.error(f"Problems computing HOG for image with shape: {img.shape}")
                        self._log.error(v)
                        name = self._prefix + constants.DELIMETER + constants.NAME_STDDEV
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_MEAN
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_VAR
                        blobAttributes[name] = 0
                        continue

                    hogNonZero = np.nonzero(fd)[0]
                    if not len(hogNonZero) > 0:
                        self._log.error(f"HOG is empty for {blobName}")
                        name = self._prefix + constants.DELIMETER + constants.NAME_STDDEV
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_MEAN
                        blobAttributes[name] = 0
                        name = self._prefix + constants.DELIMETER + constants.NAME_VAR
                        blobAttributes[name] = 0
                        continue
                    #print(f"Std Deviation: {hogNonZero.std()}")

                    name = self._prefix + constants.DELIMETER + constants.NAME_STDDEV
                    blobAttributes[name] = hogNonZero.std()
                    name = self._prefix + constants.DELIMETER + constants.NAME_MEAN
                    blobAttributes[name] = hogNonZero.mean()
                    name = self._prefix + constants.DELIMETER + constants.NAME_VAR
                    blobAttributes[name] = hogNonZero.var()
                    self._log.debug(f"HOG: {blobAttributes[self._prefix + constants.DELIMETER + constants.NAME_STDDEV]} {blobAttributes[self._prefix + constants.DELIMETER + constants.NAME_MEAN]} {blobAttributes[self._prefix + constants.DELIMETER + constants.NAME_VAR]}")
                else:
                    self._log.error("Empty blob encountered in HOG calculations")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Histogram of Oriented Gradients")

    parser.add_argument("-i", "--input", action="store", required=True, help="Image to process")
    parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
    arguments = parser.parse_args()

    #startupLogger(arguments.logging)

    histogram = HOG(None, None, image=arguments.input, output=arguments.output)
    histogram.computeAttributes()

