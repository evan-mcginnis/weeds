#
# C A M E R A F I L E
#
import pathlib
import logging
import logging.config
import numpy as np
import os
import cv2 as cv
from abc import ABC, abstractmethod
import yaml

import constants
from Performance import Performance

class Camera(ABC):

    def __init__(self, options: str):
        self.options = options

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError()
        return True

    @abstractmethod
    def initialize(self):
        return

    @abstractmethod
    def start(self):
        return

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
        self.log = logging.getLogger(__name__)
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

    def initialize(self):
        return

    def start(self):
        return

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

    def initialize(self):
        return

    def start(self):
        return

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

#
# The Basler camera is accessed through the pylon API
# Perhaps this can be through openCV, but this will do for now
#

from pypylon import pylon
from pypylon import genicam

class CameraBasler(Camera):
    def __init__(self, options: str):
     self._connected = False
     self._currentImage = 0
     self._camera = None
     self.log = logging.getLogger(__name__)
     super().__init__(options)

     # Initialize the converter for images
     # The images stream of in YUV color space.  An optimization here might be to make
     # both formats available, as YUV is something we will use later

     self._converter = pylon.ImageFormatConverter()

     # converting to opencv bgr format
     self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
     self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

     return

    def connect(self, cameraIP: str) -> bool:
        """
        Connects to the camera with the specified IP address.
        """
        tl_factory = pylon.TlFactory.GetInstance()

        self._connected = False

        for dev_info in tl_factory.EnumerateDevices():
            self.log.debug("Looking for {} device is {}".format(cameraIP, dev_info.GetIpAddress()))
            if dev_info.GetIpAddress() == cameraIP:
            #if dev_info.GetDeviceClass() == 'BaslerGigE':
                #print("using %s @ %s" % (dev_info.GetModelName(), dev_info.GetIpAddress()))
                self._camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
                self.log.info("Using device {} at {}".format(self._camera.GetDeviceInfo().GetModelName(), dev_info.GetIpAddress()))
                self._connected = True
                break

        if not self._connected:
            self.log.error("No GigE device found")
            raise EnvironmentError("No GigE device found")

        return self._connected

    def initialize(self):
        """
        Set the camera parameters to reflect what we want them to be.
        :return:
        """

        if not self._connected:
            raise IOError("Camera is not connected.")

        return

    def start(self):

        if not self._connected:
            raise IOError("Camera is not connected.")

        # Grabbing Continuously (video) with minimal delay
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.log.debug("Started grabbing images")

        return

    def disconnect(self):
        """
        Disconnected from the current camera and stop grabbing images
        """
        if self._connected:
            self._camera.StopGrabbing()

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

        if not self._connected:
            raise IOError("Camera is not connected")

        grabResult = self._camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            self.log.debug("Image grab succeeded.")
        else:
            raise IOError("There was an error encountered communicating with the camera")
        image = self._converter.Convert(grabResult)
        img = image.GetArray()
        #cv.imwrite("camera.jpg", img)
        return img

    def getResolution(self) -> ():
        w = self._camera.get(cv.CAP_PROP_FRAME_WIDTH)
        h = self._camera.get(cv.CAP_PROP_FRAME_HEIGHT)
        return (w, h)

    # This should be part of the calibration procedure
    def getMMPerPixel(self) -> float:
        return 0.0

#
# The PhysicalCamera class as a utility
#
if __name__ == "__main__":
    import argparse
    import sys
    from OptionsFile import OptionsFile

    parser = argparse.ArgumentParser("Basler Camera Utility")

    parser.add_argument('-s', '--single', action="store", required=True, help="Take a single picture")
    parser.add_argument('-l', '--logging', action="store", required=False, default="info-logging.yaml", help="Log file configuration")
    parser.add_argument('-p', '--performance', action="store", required=False, default="camera.csv", help="Performance file")
    parser.add_argument('-o', '--options', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
    arguments = parser.parse_args()

    # Initialize logging
    with open(arguments.logging, "rt") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    # Check that the format of the lines is what we expect
    #evalutionText, lines = checkLineNames(arguments.emitter)
    performance = Performance(arguments.performance)
    (performanceOK, performanceDiagnostics) = performance.initialize()

    # Parse the options file
    options = OptionsFile(arguments.options)
    options.load()

    camera = CameraBasler("")

    # Connect to the camera and take an image
    if camera.connect(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP)):
        camera.initialize()
        camera.start()

        # Check the performance of the image capture
        performance.start()
        image = camera.capture()
        performance.stopAndRecord(constants.PERF_ACQUIRE)

        # Write out the image
        cv.imwrite(arguments.single, image)
        rc = 0
    else:
        rc = -1

    sys.exit(rc)

