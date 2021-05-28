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
    def getResolution(self) -> ():
        self._connected = False
        return (0,0)

    @abstractmethod
    def getMMPerPixel(self) -> float:
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

    def getResolution(self) -> ():
        # TODO: Get the first image and return the image size
        #return self._flist[self._currentImage].shape()
        return (0,0)

    def getMMPerPixel(self) -> float:
        return 0.5

class CameraPhysical(Camera):
    def __init__(self, options: str):
     self._connected = False
     self._currentImage = 0
     self._cam = cv.VideoCapture(0)
     super().__init__(options)
     return

    def connect(self):
        """
        Connects to the camera and sets it to to highest resolution for capture.
        :return:
        True if connection was successful
        """
        # Read calibration information here
        HIGH_VALUE = 10000
        WIDTH = HIGH_VALUE
        HEIGHT = HIGH_VALUE

        # A bit a hack to set the camera to the highest resolution
        self._cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
        self._cam.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        return True

    def disconnect(self):
        self._cam.release()

    def diagnostics(self) -> (bool, str):
        """
        Execute diagnostics on the camera.
        :return:
        Boolean result of the diagnostics and a string of the details
        """
        return True, "Camera diagnostics not provided"

    def capture(self) -> np.ndarray:
        """
        Capture a single image from the camera.
        Requires calling the connect() method before this call.
        :return:
        The image as a numpy array
        """
        ret, frame = self._cam.read()
        if not ret:
            raise IOError("There was an error encountered communicating with the camera")
        #cv.imwrite("camera.jpg", frame)
        return frame

    def getResolution(self) -> ():
        w = self._cam.get(cv.CAP_PROP_FRAME_WIDTH)
        h = self._cam.get(cv.CAP_PROP_FRAME_HEIGHT)
        return (w, h)

    # This should be part of the calibration procedure
    def getMMPerPixel(self) -> float:
        return 0.0
