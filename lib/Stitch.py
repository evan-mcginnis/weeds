import argparse
import cv2 as cv
import numpy as np
import logging
import os

import constants

class Stitch:
    def __init__(self, mode: int):
        #self._stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)
        self._stitcher = cv.Stitcher.create(mode)
        self._stitched = None
        self._imageNames = []

    @property
    def stitched(self):
        return self._stitched

    @property
    def images(self) -> []:
        return self._imageNames

    @images.setter
    def images(self, imageNames: []):
        self._imageNames = imageNames

    def stitchImages(self):
        images = []
        if len(self._imageNames) < 2:
            raise ValueError("Need at least 2 images specified")
        else:
            for image in self._imageNames:
                try:
                    rawImage = cv.imread(image)
                    images.append(rawImage)
                except Exception:
                    raise RuntimeError(f"Unable to read: {image}")

            (status, pano) = self._stitcher.stitch(images)
            if status == cv.STITCHER_OK:
                cv.imwrite("stitched.jpg", pano)
            else:
                print(f"Error in stitching: {status}")
        return pano


if __name__ == "__main__":
    modes = (cv.Stitcher_SCANS, cv.Stitcher_PANORAMA)
    parser = argparse.ArgumentParser("Stitch Images")

    parser.add_argument("-i", "--input", action="store", required=True, nargs="*", help="Image directory to process")
    parser.add_argument("-o", "--output", action="store", required=True, default=".", help="Output directory")
    parser.add_argument("-m", "--mode", choices=modes, required=False, default=cv.Stitcher_PANORAMA)
    arguments = parser.parse_args()

    #startupLogger(arguments.logging)

    allImages = []
    for image in arguments.input:
        allImages.append(image)

    composite = Stitch(arguments.mode)
    composite.images = allImages
    composite.stitchImages()




