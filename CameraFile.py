#
# C A M E R A F I L E
#
import pathlib
import numpy as np
#import Camera
import numpy as np
import os
import cv2 as cv
from abc import ABC, abstractmethod

class Camera(ABC):

    def __init__(self, options: str):
        self.options = options

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError()
        return True

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError()
        return True

    @abstractmethod
    def diagnostics(self):
        self._connected = False
        return 0

    @abstractmethod
    def capture(self) -> np.ndarray:
        self._connected = False
        return

    @abstractmethod
    def getResolution(self):
        self._connected = False
        return (0,0)

    @abstractmethod
    def getMMPerPixel(self):
        return



class CameraFile(Camera):
    def __init__(self, options: str):
     self._connected = False
     self.directory = options
     self._currentImage = 0
     super().__init__(options)
     return

    def connect(self) -> bool:
        """
        Connects to a directory and finds all images there. This method will not traverse subdirectories
        :return:
        """
        self._connected = os.path.isdir(self.directory)
        # Find all the files in the directory.
        # TODO: find only the images.
        if self._connected:
            self._flist = [p for p in pathlib.Path(self.directory).iterdir() if p.is_file()]
        return self._connected

    def disconnect(self):
        self._connected = False
        return True

    def diagnostics(self):
        return True, "Camera diagnostics passed"

    def capture(self) -> np.ndarray:
        """
        Each time capture() is called, the next image in the directory is returned
        :return:
        The image as a numpy array.  Raises EOFError when no more images exist
        """
        if self._currentImage < len(self._flist):
            imageName = str(self._flist[self._currentImage])
            image = cv.imread(imageName,cv.IMREAD_COLOR)
            self._currentImage = self._currentImage + 1
            return(image)
        # Raise an EOFError  when we get through the sequence of images
        else:
            raise EOFError

    def getResolution(self):
        # TODO: Get the first image and return the image size
        #return self._flist[self._currentImage].shape()
        return

    def getMMPerPixel(self):
        raise NotImplementedError

class CameraPhysical(Camera):
    def __init__(self, options: str):
     self._connected = False
     self._currentImage = 0
     super().__init__(options)
     return

    def connect(self):
        """
        Connects to a directory and finds all images there. This method will not traverse subdirectories
        :return:
        """
        # Read calibration information here
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def diagnostics(self):
        raise NotImplementedError

    def capture(self) -> np.ndarray:
        raise NotImplementedError

    def getResolution(self):
        raise NotImplementedError

    # This should be part of the calibration procedure
    def getMMPerPixel(self):
        raise NotImplementedError
