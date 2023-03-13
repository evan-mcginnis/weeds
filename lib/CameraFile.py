#
# C A M E R A F I L E
#
import glob
import pathlib
import logging
import logging.config
import time
from collections import deque
import signal

import numpy as np
import os
import cv2 as cv
#from abc import ABC, abstractmethod



import constants
from ProcessedImage import ProcessedImage
from Performance import Performance
from Camera import Camera


class CameraFile(Camera):
    def __init__(self, **kwargs):
        self._connected = False
        self.directory = kwargs[constants.KEYWORD_DIRECTORY]
        self._type = kwargs[constants.KEYWORD_TYPE]
        self._currentImage = 0
        super().__init__(**kwargs)
        self.log = logging.getLogger(__name__)
        return

    def connect(self) -> bool:
        """
        Connects to a directory and finds all images there. This method will not traverse subdirectories
        :return:
        """
        self._connected = os.path.isdir(self.directory)
        if not self._connected:
            self.log.error("Unable to connect to directory: {}".format(self.directory))
        else:
            self.log.debug("Connected to directory: {}".format(self.directory))

        if self._connected:
            # Images are .jpg files
            if self._type == constants.ImageType.RGB.name:
                pattern = "/*" + constants.EXTENSION_IMAGE
            # Depth data are .npy files
            elif self._type == constants.ImageType.DEPTH.name:
                pattern = "/*" + constants.EXTENSION_NPY
            else:
                self.log.error("Can't process type: {}".format(self._type))
            files = self.directory + pattern
            self._flist = glob.glob(files)
            #self._flist = [p for p in pathlib.Path(self.directory).iterdir() if p.is_file()]
        return self._connected

    def disconnect(self):
        self._connected = False
        return True

    def diagnostics(self):
        return True, "Camera diagnostics passed"

    def initialize(self):
        return

    def start(self):
        return

    def capture(self) -> ProcessedImage:
        """
        Each time capture() is called, the next image in the directory is returned
        :return:
        The image as a numpy array.  Raises EOFError when no more images exist
        """
        if self._currentImage < len(self._flist):
            image = None
            if self._type == constants.ImageType.RGB.name:
                imageName = str(self._flist[self._currentImage])
                image = cv.imread(imageName, cv.IMREAD_COLOR)
                self._currentImage = self._currentImage + 1
            elif self._type == constants.ImageType.DEPTH.name:
                imageName = str(self._flist[self._currentImage])
                image = np.load(imageName)
            else:
                self.log.error("Can't process type: {}".format(self._type))

            processed = ProcessedImage(constants.ImageType.RGB, image, 0)

            return processed
        # Raise an EOFError  when we get through the sequence of images
        else:
            raise EOFError

    def getResolution(self) -> ():
        # TODO: Get the first image and return the image size
        #return self._flist[self._currentImage].shape()
        return (0,0)

    def getMMPerPixel(self) -> float:
        return 0.5

if __name__ == "__main__":
    print("No test method")
