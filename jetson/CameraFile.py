#
# C A M E R A F I L E
#
import datetime
import pathlib
import logging
import logging.config
import time
from collections import deque

import numpy as np
import os
import cv2 as cv
from abc import ABC, abstractmethod

import pypylon.pylon
import yaml

import constants
from Performance import Performance

class Camera(ABC):

    def __init__(self, **kwargs):
        return

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
    def __init__(self, **kwargs):
        self._connected = False
        self.directory =  kwargs[constants.KEYWORD_DIRECTORY]
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
    def __init__(self, **kwargs):
     self._connected = False
     self._currentImage = 0
     self._cam = cv.VideoCapture(0)
     super().__init__(**kwargs)
     return

    def connect(self):
        """
        Connects to the camera and sets it to highest resolution for capture.
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

class ProcessedImage():
    def __init__(self, image, timestamp: int):
        self._image = image
        self._timestamp = timestamp
        self._indexed = False

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def timestamp(self) -> int:
        return self._timestamp

from pypylon import pylon
from pypylon import genicam

class CameraBasler(Camera):
    def __init__(self, **kwargs):
        """
        Tha basler camera object.
        :param kwargs: ip=<ip-address of camera>
        """
        self._connected = False
        self._currentImage = 0
        self._camera = None
        self.log = logging.getLogger(__name__)
        self._strategy = constants.STRATEGY_ASYNC
        self._capturing = False
        self._images = deque(maxlen=constants.IMAGE_QUEUE_LEN)
        self._camera = pylon.InstantCamera()

        # Assume a GigE camera for now
        self._ip = kwargs[constants.KEYWORD_IP]

        super().__init__(**kwargs)

        # Initialize the converter for images
        # The images stream of in YUV color space.  An optimization here might be to make
        # both formats available, as YUV is something we will use later

        self._converter = pylon.ImageFormatConverter()

        # converting to opencv bgr format
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        return

    def connect(self) -> bool:
        """
        Connects to the camera with the specified IP address.
        """
        tl_factory = pylon.TlFactory.GetInstance()

        self._connected = False

        for dev_info in tl_factory.EnumerateDevices():
            self.log.debug("Looking for {}. Current device is {}".format(self._ip, dev_info.GetIpAddress()))
            if dev_info.GetIpAddress() == self._ip:
                self._camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
                self._camera.MaxNumBuffer = 15

                try:
                    self._camera.Open()
                    self.log.info("Using device {} at {}".format(self._camera.GetDeviceInfo().GetModelName(), dev_info.GetIpAddress()))
                    self._connected = True
                except Exception as e:
                    self.log.error("Error encountered in opening camera")
                # This shows how to get the list of what is available as attributes.  Not particularly useful for what
                # we need here
                # info = pylon.DeviceInfo()
                # info = self._camera.GetDeviceInfo()
                # tlc = pylon.GigETransportLayer()
                # tlc = self._camera.GetTLNodeMap()
                #
                # properties = info.GetPropertyNames()

                #self.log.debug("Current counter {}".format())
                break

        if not self._connected:
            self.log.error("Failed to connect to camera")
            #raise EnvironmentError("No GigE device found")

        return self._connected


    def initialize(self):
        """
        Set the camera parameters to reflect what we want them to be.
        :return:
        """

        if not self._connected:
            raise IOError("Camera is not connected.")

        # TODO: Setup network parameters
        # Inter-packet gap should be 5000
        # This mess is an attempt to do this
        # info = pylon.DeviceInfo(self._camera.GetDeviceInfo())
        # # tlc = pylon.GigETransportLayer()
        # tlc = self._camera.GetTLNodeMap()
        # #
        # properties = info.GetPropertyNames()
        #
        # # This lets me know the option is there:
        # if info.GetPropertyAvailable("IpConfigOptions"):
        #     ipOptions = info.GetIpConfigOptions()
        # tl = pylon.TransportLayer(self._camera.TransportLayer)
        # tlInfo = tl.GetTlInfo()

        self.log.debug("Camera initialized")
        return

    def start(self):
        """
        Begin capturing images and store them in a queue for later retrieval.
        """

        if not self._connected:
            raise IOError("Camera is not connected.")

        # The scheme here is to get the images and store them for later consumption.
        # The basler library does not have quite what is needed here, as we can't quite tell
        # when an image is needed, as that is based on distance tranversed (let's say images every 10 cm to allow for
        # overlap.

        # Start grabbing images
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.log.debug("Started grabbing images")
        # Fetch the images from the camera and store the results in a buffer
        if self._strategy == constants.STRATEGY_ASYNC:
            self.log.debug("Asynchronous capture")
            self._capturing = True
            while self._capturing:
                timestamped = self._grab()
                # Add the image to the queue
                self._images.append(timestamped)
            self._camera.StopGrabbing()

        # For synchronous capture, we don't do anything but retrieve the image on demand
        else:
            self.log.debug("Synchronous capture")



    def stop(self):
        """
        Stop collecting from the current camera.
        :return:
        """

        self.log.debug("Stopping image capture")

        # Stop only if camera is connected.  This doesn't directly stop the collection, but clears the flag so the
        # collection loop will stop
        if self._connected:
            self._capturing = False
        # if self._strategy == constants.STRATEGY_ASYNC:
        #     self._camera.StopGrabbing()

        return

    def disconnect(self):
        """
        Disconnected from the current camera and stop grabbing images
        """

        self.log.debug("Disconnecting from camera")
        self.stop()
        self._camera.Close()

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

        If the image is in the queue, it will be served from there -- otherwise the method will retrieve if
        synchronously from the camera.

        :return:
        The image as a numpy array
        """

        if not self._connected:
            raise IOError("Camera is not connected")

        # If there are no images in the queue, just wait for one.
        if len(self._images) == 0:
            self.log.debug("Image queue is empty.   Waiting for grab")
            img = self._grab()

        else:
            self.log.debug("Serving image from queue")
            processed = self._images.popleft()
            img = processed.image
            timestamp = processed.timestamp
            self.log.debug("Image captured at " + str(timestamp))
        return img

    def getResolution(self) -> ():
        w = self._camera.get(cv.CAP_PROP_FRAME_WIDTH)
        h = self._camera.get(cv.CAP_PROP_FRAME_HEIGHT)
        return (w, h)

    # This should be part of the calibration procedure
    def getMMPerPixel(self) -> float:
        return 0.0

    @property
    def camera(self) -> pylon.InstantCamera:
        return self._camera

    def _grab(self) -> ProcessedImage:
        """
        Grab the image from the camera
        :return: ProcessedImage
        """

        grabResult = pypylon.pylon.GrabResult(self._camera.RetrieveResult(constants.TIMEOUT_CAMERA, pylon.TimeoutHandling_ThrowException))
        # If the camera is close while we are capturing, this may be null.
        # TODO: More elegant way of doing this
        if grabResult and grabResult.GrabSucceeded():
            pass
            # This is very noisy -- a bit more than we need here
            #self.log.debug("Image grab succeeded at timestamp " + str(grabResult.TimeStamp))
        else:
            raise IOError("Failed to grab image. Pylon error code: {}".format(grabResult.GetErrorCode()))

        image = self._converter.Convert(grabResult)
        img = image.GetArray()
        timestamped = ProcessedImage(img, grabResult.TimeStamp)
        return timestamped

    def save(self, filename: str) -> bool:
        """
        Save the camera settings
        :param filename: The file to contain the settings
        :return: True on success
        """
        pylon.FeaturePersistence.Save(filename, self._camera.GetNodeMap())
        return True

    def load(self, filename: str) -> bool:
        pylon.FeaturePersistence.Load(filename,self._camera.GetNodeMap(),True)
        return True
#
# The PhysicalCamera class as a utility
#
if __name__ == "__main__":

    import argparse
    import sys
    from OptionsFile import OptionsFile

    import threading

    def takeImages(camera: CameraBasler):

        # Connect to the camera and take an image
        if camera.connect():
            try:
                camera.initialize()
                camera.start()
            except IOError as io:
                camera.log.error(io)
            rc = 0
        else:
            rc = -1


    parser = argparse.ArgumentParser("Basler Camera Utility")

    parser.add_argument('-s', '--single', action="store", required=True, help="Take a single picture")
    parser.add_argument('-l', '--logging', action="store", required=False, default="info-logging.yaml", help="Log file configuration")
    parser.add_argument('-p', '--performance', action="store", required=False, default="camera.csv", help="Performance file")
    parser.add_argument('-o', '--options', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
    parser.add_argument('-a', '--asynch', action="store_true", required=False, default=False, help="Use asynchronous image acquisition")
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
    if not options.load():
        print("Error encountered with option load for: {}".format(arguments.options))
        sys.exit(1)

    cameraIP = options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP)
    camera = CameraBasler(ip = cameraIP)

    # Start the thread that will begin acquiring images
    acquire = threading.Thread(target=takeImages, args=(camera,))
    acquire.start()

    time.sleep(10)

    # This is how you save the camera features
    camera.save("basler1920.txt")
    # You can see all settings that way, and also reverse the process (edit the file, then call
    # pylon.FeaturePersistence.Load() to set new values). The viewer app from Basler can also load or save these files.

    timenow = time.time()
    logging.debug("Image needed from {}".format(timenow))
    try:
        performance.start()
        img = camera.capture()
        performance.stopAndRecord(constants.PERF_ACQUIRE)
        cv.imwrite(arguments.single, img)
    except IOError as io:
        camera.log.error("Failed to capture image: {0}".format(io))
    rc = 0

    # Stop the camera, and this should stop the thread as well

    camera.disconnect()

    sys.exit(rc)

