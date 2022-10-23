#
# W E E D S
#

import argparse
import glob
import platform
import sys
import threading
import time
from typing import Callable

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import logging
import logging.config
import yaml
import os
import shutil

import xmpp
#from xmpp import protocol

# This does not work
#from CameraFile import CameraFile, CameraBasler

from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SuppportVectorMachineClassifier
from OptionsFile import OptionsFile
from Performance import Performance
from Reporting import Reporting
from Treatment import Treatment
from MUCCommunicator import MUCCommunicator
from Messages import OdometryMessage, SystemMessage, TreatmentMessage
from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure

#from Selection import Selection

import constants

#
# C A M E R A S
#
# TODO: Move to Camera.py file
# This is very sloppy work, and has completely defeated me, so I give up
# This works just fine in another file, but fails whenever it is imported,
# so I'm giving up and copying it here

import pathlib
import logging
import logging.config
import time
from collections import deque
import signal

import numpy as np
import os
import cv2 as cv
from abc import ABC, abstractmethod

import pypylon.pylon
from pypylon import _genicam

import constants
from Performance import Performance



#
# B A S L E R  E V E N T  H A N D L E R S
#

# Handle various basler camera events

class ConfigurationEventPrinter(pypylon.pylon.ConfigurationEventHandler):
    def OnAttach(self, camera):
        print("OnAttach event")

    def OnAttached(self, camera):
        print("OnAttached event for device ", camera.GetDeviceInfo().GetModelName())

    def OnOpen(self, camera):
        print("OnOpen event for device ", camera.GetDeviceInfo().GetModelName())

    def OnOpened(self, camera):
        print("OnOpened event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStart(self, camera):
        print("OnGrabStart event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStarted(self, camera):
        print("OnGrabStarted event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStop(self, camera):
        print("OnGrabStop event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabStopped(self, camera):
        print("OnGrabStopped event for device ", camera.GetDeviceInfo().GetModelName())

    def OnClose(self, camera):
        print("OnClose event for device ", camera.GetDeviceInfo().GetModelName())

    def OnClosed(self, camera):
        print("OnClosed event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDestroy(self, camera):
        print("OnDestroy event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDestroyed(self, camera):
        print("OnDestroyed event")

    def OnDetach(self, camera):
        print("OnDetach event for device ", camera.GetDeviceInfo().GetModelName())

    def OnDetached(self, camera):
        print("OnDetached event for device ", camera.GetDeviceInfo().GetModelName())

    def OnGrabError(self, camera, errorMessage):
        print("OnGrabError event for device ", camera.GetDeviceInfo().GetModelName())
        print("Error Message: ", errorMessage)

    def OnCameraDeviceRemoved(self, camera):
        print("OnCameraDeviceRemoved event for device ", camera.GetDeviceInfo().GetModelName())

# Handle image grab notifications

class ImageEvents(pypylon.pylon.ImageEventHandler):
    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print()

    def OnImageGrabbed(self, camera, grabResult):
        """
        Called when an image has been grabbed by the camera
        :param camera:
        :param grabResult:
        """
        #log.debug("OnImageGrabbed event for device: {}".format(camera.GetDeviceInfo().GetModelName()))

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # Convert the image grabbed to something we like
            image = CameraBasler.convert(grabResult)
            img = image.GetArray()
            # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
            # We will mark the images based on when we got them -- ideally, this should be:
            # timestamped = ProcessedImage(img, grabResult.TimeStamp)
            timestamped = ProcessedImage(img, round(time.time() * 1000))

            cameraNumber = camera.GetCameraContext()
            camera = Camera.cameras[cameraNumber]
            #log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
            camera._images.append(timestamped)

            # print("SizeX: ", grabResult.GetWidth())
            # print("SizeY: ", grabResult.GetHeight())
            # img = grabResult.GetArray()
            # print("Gray values of first row: ", img[0])
            # print()
        else:
            log.error("Image Grab error code: {} {}".format(grabResult.GetErrorCode(), grabResult.GetErrorDescription()))

# Example of an image event handler.
class SampleImageEventHandler(pypylon.pylon.ImageEventHandler):
    def OnImageGrabbed(self, camera, grabResult):
        print("CSampleImageEventHandler::OnImageGrabbed called.")
        print()
        print()

class Camera(ABC):
    cameras = list()
    cameraCount = 0

    def __init__(self, **kwargs):

        # Register the camera on the global list so we can keep track of them
        # Even though there will probably be only one
        self.cameraID = Camera.cameraCount
        Camera.cameraCount += 1
        Camera.cameras.append(self)

        self._status = constants.OperationalStatus.UNKNOWN

        self._gsd = 0
        return

    @property
    def status(self) -> constants.OperationalStatus:
        return self._status

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

    @property
    def gsd(self) -> int:
        """
        The ground sampling distance as specified in the options file. As we can't determine how high off the ground
        the camera is, this is a pre-computed value
        :return: width of the ground capture.
        """
        return self._gsd

    @gsd.setter
    def gsd(self, distance: int):
        self._gsd = distance

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
    # Initialize the converter for images
    # The images stream of in YUV color space.  An optimization here might be to make
    # both formats available, as YUV is something we will use later

    _converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    _converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    _converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

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



        return

    @classmethod
    def convert(cls, grabResult):
        image = CameraBasler._converter.Convert(grabResult)
        return image


    def connect(self) -> bool:
        """
        Connects to the camera with the specified IP address.
        """
        tl_factory = pylon.TlFactory.GetInstance()

        self._connected = False

        for dev_info in tl_factory.EnumerateDevices():
            self.log.debug("Looking for {}. Current device is {}".format(self._ip, dev_info.GetIpAddress()))
            if dev_info.GetIpAddress() == self._ip:
                try:
                    self._camera = pylon.InstantCamera()
                    self._camera.Attach(tl_factory.CreateDevice(dev_info))
                except Exception as e:
                    log.fatal("Error encountered in attaching camera")
                    log.fatal("{}".format(e))
                    self._status = constants.OperationalStatus.FAIL
                #self._camera.MaxNumBuffer = 100
                try:
                    self._camera.Open()
                    self.log.info("Using device {} at {}".format(self._camera.GetDeviceInfo().GetModelName(), dev_info.GetIpAddress()))
                    self._connected = True
                    self._status = constants.OperationalStatus.OK
                except Exception as e:
                    self.log.fatal("Error encountered in opening camera")
                    self._status = constants.OperationalStatus.FAIL
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


    def initializeCapture(self):

        initialized = False
        try:
            # Register the standard configuration event handler for enabling software triggering.
            # The software trigger configuration handler replaces the default configuration
            # as all currently registered configuration handlers are removed by setting the registration mode to RegistrationMode_ReplaceAll.
            self._camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(),
                                               pylon.RegistrationMode_ReplaceAll,
                                               pylon.Cleanup_Delete)

            # For demonstration purposes only, add a sample configuration event handler to print out information
            # about camera use.t
            self._camera.RegisterConfiguration(ConfigurationEventPrinter(),
                                               pylon.RegistrationMode_Append,
                                               pylon.Cleanup_Delete)

            # The image event printer serves as sample image processing.
            # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
            # results must be created and registered.
            self._camera.RegisterImageEventHandler(ImageEvents(),
                                                   pylon.RegistrationMode_Append,
                                                   pylon.Cleanup_Delete)

            # For demonstration purposes only, register another image event handler.
            # self._camera.RegisterImageEventHandler(SampleImageEventHandler(),
            #                                        pylon.RegistrationMode_Append,
            #                                        pylon.Cleanup_Delete)

            self._camera.SetCameraContext(self.cameraID)
            initialized = True

        except genicam.GenericException as e:
            # Error handling.
            self.log.fatal("Unable to initialize the capture", e.GetDescription())
            initialized = False

        return initialized

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
    #
    # def startGrabbingImages(self):
    #
    #
    #     self.log.debug("Start to grab images")
    #     self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    #     # Execute the software trigger, wait actively until the camera accepts the next frame trigger or until the timeout occurs.
    #     for i in range(3):
    #         if self._camera.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
    #             self._camera.ExecuteSoftwareTrigger()
    #
    #     # Wait for all images.
    #     time.sleep(0.2)
    #
    #     # Check whether the grab result is waiting.
    #     if self._camera.GetGrabResultWaitObject().Wait(0):
    #         print("A grab result waits in the output queue.")
    #
    #     # Only the last received image is waiting in the internal output queue
    #     # and is now retrieved.
    #     # The grabbing continues in the background, e.g. when using hardware trigger mode.
    #     buffersInQueue = 0
    #
    #     while True:
    #         grabResult = self._camera.RetrieveResult(0, pylon.TimeoutHandling_Return)
    #         if not grabResult.IsValid():
    #             break
    #         print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
    #         buffersInQueue += 1
    #
    #     print("Retrieved ", buffersInQueue, " grab result from output queue.")
    #
    #     # Stop the grabbing.
    #     self.log.debug("Finished grabbing")
    #     self._camera.StopGrabbing()
    #
    # def startCapturing(self):
    #     """
    #     Begin capturing images and store them in a queue for later retrieval.
    #     """
    #
    #     if not self._connected:
    #         raise IOError("Camera is not connected.")
    #
    #     # The scheme here is to get the images and store them for later consumption.
    #     # The basler library does not have quite what is needed here, as we can't quite tell
    #     # when an image is needed, as that is based on distance tranversed (let's say images every 10 cm to allow for
    #     # overlap.
    #
    #     # Start grabbing images
    #     #self._camera.GevSCPSPacketSize = 8192
    #     self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    #     self.log.debug("Started grabbing images")
    #     # Fetch the images from the camera and store the results in a buffer
    #     if self._strategy == constants.STRATEGY_ASYNC:
    #         self.log.debug("Asynchronous capture")
    #         self._capturing = True
    #         while self._camera.IsGrabbing():
    #             try:
    #                 timestamped = self._grabImage()
    #                 self._images.append(timestamped)
    #             except IOError as e:
    #                 self.log.error(e)
    #             self.log.debug("Image queue size: {}".format(len(self._images)))
    #         self._camera.StopGrabbing()
    #
    #     # For synchronous capture, we don't do anything but retrieve the image on demand
    #     else:
    #         self.log.debug("Synchronous capture")

    def startCapturing(self):
        # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
        # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
        # The GrabStrategy_OneByOne default grab strategy is used.
        self.log.debug("Begin grab with OneByOne Strategy")
        try:
            #self.camera.Open()
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
        except _genicam.RuntimeException as e:
            log.fatal("Failed to open the camera and start grabbing.")
            log.fatal("{}".format(e))

        # If we immediately start waiting for the trigger, we get an error
        time.sleep(2)
        self._capturing = True
        while self._capturing:
            try:
                if self.camera.WaitForFrameTriggerReady(400, pylon.TimeoutHandling_ThrowException):
                    self.camera.ExecuteSoftwareTrigger();
            except _genicam.TimeoutException as e:
                self.log.fatal("Timeout from camera")
            except _genicam.RuntimeException as e:
                if not self._capturing:
                    self.log.warning("Errors encountered in shutdown.  This is normal")
                else:
                    self.log.error("Unexpected errors in capture")
                    self.log.error("Device: {}".format(self._camera.GetDeviceInfo().GetModelName()))
                    self.log.error("{}".format(e))

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
        self._camera.GevSCPSPacketSize = 8192
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.log.debug("Started grabbing images")
        # Fetch the images from the camera and store the results in a buffer
        if self._strategy == constants.STRATEGY_ASYNC:
            self.log.debug("Asynchronous capture")
            self._capturing = True
            while self._capturing:
                try:
                    timestamped = self._grab()
                    self._images.append(timestamped)
                except IOError as e:
                    self.log.error(e)
                #self.log.debug("Image queue size: {}".format(len(self._images)))
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
            self.log.error("Image queue is empty.")
            os.kill(os.getpid(), signal.SIGINT)
            #img = self._grab()
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

    @camera.setter
    def camera(self, openedCamera: pylon.InstantCamera):
        self._camera = openedCamera


    def _grabImage(self) -> ProcessedImage:

        try:
            grabResult = self._camera.RetrieveResult(200, pylon.TimeoutHandling_ThrowException)

        except _genicam.RuntimeException as e:
            self.log.fatal("Genicam runtime error encountered.")
            self.log.fatal("{}".format(e))

        if grabResult.GrabSucceeded():
            # This is very noisy -- a bit more than we need here
            self.log.debug("Image grab succeeded at timestamp " + str(grabResult.TimeStamp))
        else:
            raise IOError("Failed to grab image. Pylon error code: {}".format(grabResult.GetErrorCode()))

        image = self._converter.Convert(grabResult)
        img = image.GetArray()
        # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
        # We will mark the images based on when we got them -- ideally, this should be:
        # timestamped = ProcessedImage(img, grabResult.TimeStamp)
        timestamped = ProcessedImage(img, round(time.time() * 1000))
        return timestamped

    def _grab(self) -> ProcessedImage:
        """
        Grab the image from the camera
        :return: ProcessedImag
        """


        try:
            self.log.debug("Grab start")
            grabResult = pypylon.pylon.GrabResult(self._camera.RetrieveResult(constants.TIMEOUT_CAMERA, pylon.TimeoutHandling_ThrowException))
            self.log.debug("Grab complete")
        except _genicam.RuntimeException as e:
            self.log.fatal("Genicam runtime error encountered.")
            self.log.fatal("{}".format(e))
        # If the camera is close while we are capturing, this may be null.
        if not grabResult.IsValid():
            self.log.error("Image is not valid")
            raise IOError("Image is not valid")

        if grabResult.GrabSucceeded():
            # This is very noisy -- a bit more than we need here
            #self.log.debug("Image grab succeeded at timestamp " + str(grabResult.TimeStamp))
            pass
        else:
            raise IOError("Failed to grab image. Pylon error code: {}".format(grabResult.GetErrorCode()))

        image = self._converter.Convert(grabResult)
        img = image.GetArray()
        # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
        # We will mark the images based on when we got them -- ideally, this should be:
        #timestamped = ProcessedImage(img, grabResult.TimeStamp)
        timestamped = ProcessedImage(img, round(time.time() * 1000))
        return timestamped

    def save(self, filename: str) -> bool:
        """
        Save the camera settings
        :param filename: The file to contain the settings
        :return: True on success
        """
        #self._camera.Open()
        self.log.info("Saving camera configuration to: {}".format(filename))
        pylon.FeaturePersistence.Save(filename, self._camera.GetNodeMap())
        return True

    def load(self, filename: str) -> bool:
        """
        Load the camera configuration from a file. Usually, this is the .pfs file saved from the pylon viewer
        :param filename: The name of the file on disk
        :return: True on success
        """
        loaded = False

        #self._camera.Open()
        # If the camera configuration exists, use that, otherwise warn
        if os.path.isfile(filename):
            self.log.info("Using saved camera configuration: {}".format(filename))
            try:
                pylon.FeaturePersistence.Load(filename,self._camera.GetNodeMap(),True)
            except _genicam.RuntimeException as geni:
                log.error("Unable to load configuration: {}".format(geni))

            loaded = True
        else:
            self.log.warning("Unable to find configuration file: {}.  Camera configuration unchanged".format(filename))
        return loaded
#
# E N D  C A M A R A S
#

# Used in command line processing so we can accept thresholds that are tuples
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

# This is here so we can extract the supported algorithms

veg = VegetationIndex()

parser = argparse.ArgumentParser("Weed recognition system")

parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms(),
                    default="ngrdi")
parser.add_argument("-c", "--contours", action="store_true", default=False, help="Show contours on images")
parser.add_argument("-d", "--decorations", action="store", type=str, default="all", help="Decorations on output images (all and none are shortcuts)")
parser.add_argument("-df", "--data", action="store", help="Name of the data in CSV for use in logistic regression or KNN")
parser.add_argument("-hg", "--histograms", action="store_true", default=False, help="Show histograms")
parser.add_argument("-he", "--height", action="store_true", default=False, help="Consider height in scoring")
parser.add_argument('-i', '--input', action="store", required=False, help="Images directory")
parser.add_argument("-gr", "--grab", action="store_true", default=False, help="Just grab images. No processing")
group = parser.add_mutually_exclusive_group()
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
group.add_argument("-k", "--knn", action="store_true", default=False, help="Predict using KNN. Requires data file to be specified")
group.add_argument("-l", "--logistic", action="store_true", default=False, help="Predict using logistic regression. Requires data file to be specified")
group.add_argument("-dt", "--tree", action="store_true", default=False, help="Predict using decision tree. Requires data file to be specified")
group.add_argument("-f", "--forest", action="store_true", default=False, help="Predict using random forest. Requires data file to be specified")
group.add_argument("-g", "--gradient", action="store_true", default=False, help="Predict using gradient boosting. Requires data file to be specified")
group.add_argument("-svm", "--support", action="store_true", default=False, help="Predict using support vector machine. Requires data file to be specified")
parser.add_argument("-im", "--image", action="store", default=200, type=int, help="Horizontal length of image")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
parser.add_argument("-mr", "--minratio", action="store", default=5, type=int, help="Minimum size ratio for classifier")
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
parser.add_argument("-r", "--results", action="store", default="results.csv", help="Name of results file")
parser.add_argument("-s", "--stitch", action="store_true", help="Stitch adjacent images together")
parser.add_argument("-sc", "--score", action="store_true", help="Score the prediction method")
parser.add_argument("-se", "--selection", action="store", default="all-parameters.csv", help="Parameter selection file")
parser.add_argument("-sp", "--spray", action="store_true", help="Generate spray treatment grid")
parser.add_argument("-spe", "--speed", action="store", default=1, type=int, help="Speed in meters per second")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Generate debugging data and text")
parser.add_argument("-x", "--xtract", action="store_true", default=False, help="Extract each crop plant into images")


arguments = parser.parse_args()

# This is just the root of the output directory, typically ../output.  Later, this will be ../output/<UUID> for
# a specific session

outputDirectory = arguments.output

# The list of decorations on the output.
# index
# classifier
# ratio
# center
# area
# distance
decorations = [item for item in arguments.decorations.split(',')]

if (arguments.logistic or arguments.knn or arguments.tree or arguments.forest) and arguments.data is None:
    print("Data file is not specified.")
    sys.exit(1)

#
# C A M E R A
#

def startupCamera(options: OptionsFile):
    if arguments.input is not None:
        # Get the images from a directory
        theCamera = CameraFile(directory=arguments.input)
    else:
        # Get the images from an actual camera
        cameraIP = options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP)
        theCamera = CameraBasler(ip=cameraIP)
        # Set the ground sampling distance, so we know when to take a picture
        theCamera.gsd = int(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_IMAGE_WIDTH))

    # Load the camera with the settings we want, otherwise save what is there
    # camera.connect()
    # filename = camera.camera.GetDeviceInfo().GetModelName() + ".pfs"
    # camera.camera.Open()
    # if not camera.load(filename):
    #     # If we don't have a configuration, save what was used
    #     filename = "default-" + filename
    #     camera.save(filename)
    # camera.camera.Close()

    # Test the camera
    diagnosticResult, diagnosticText = theCamera.diagnostics()
    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)


    return theCamera



def startupPerformance() -> Performance:
    """
    Start up the performance subsystem.
    :return:
    """
    performance = Performance(arguments.performance)
    (performanceOK, performanceDiagnostics) = performance.initialize()
    if not performanceOK:
        print(performanceDiagnostics)
        sys.exit(1)
    return performance

#
# X M P P   C O M M U N I C A T I O N S
#
# def process(conn,msg):# xmpp.protocol.Message):
#     log.debug("Callback for distance")
#     return

def startupCommunications(options: OptionsFile, callbackOdometer: Callable, callbackSystem: Callable, callbackTreatment: Callable) -> ():
    """

    :param options:
    :param callbackOdometer:
    :param callbackSystem:
    :return:
    """
    # print("Joining room with options {},{},{},{}".format(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_JETSON),
    #     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON),
    #     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
    #     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY)))

    # The room that will get the announcements about forward or backward progress
    odometryRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_JETSON),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
                                   callbackOdometer,
                                   None)

    # The room that will get status reports about this process
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_JETSON),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
                                 callbackSystem,
                                 None)

    treatmentRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_JETSON),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
                                    callbackTreatment,
                                    None)
    #print("XMPP communications started")

    return (odometryRoom, systemRoom, treatmentRoom)
#
# L O G G E R
#

def startupLogger(outputDirectory: str) -> ():
    """
    Initializes two logging systems: the image logger and python centric logging.
    :param outputDirectory: The output directory for the images
    :return: The image logger instance
    """

    # The command line argument contains the name of the YAML configuration file.

    # Confirm the INI exists
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)


    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("jetson")

    logger = Logger()
    if not logger.connect(outputDirectory):
        print("Unable to connect to logging. ./output does not exist.")
        sys.exit(1)
    return (logger, log)

def plot3D(index, title):
    yLen,xLen = index.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,10))
    axes = fig.gca(projection ='3d')
    plt.title(title)
    axes.scatter(x, y, index, c=index, cmap='BrBG', s=0.25)
    plt.show()
    cv.waitKey()

def readINI() -> OptionsFile:
    options = OptionsFile(arguments.ini)
    options.load()
    return options

# Keep track of attributes in processing

reporting = Reporting(arguments.results)

(reportingOK, reportingReason) = reporting.initialize()
if not reportingOK:
    print(reportingReason)
    sys.exit(1)

# Used in stitching
previousImage = None
sequence = 0

# The factors considered in classification
#
# factors = [constants.NAME_RATIO,
#            constants.NAME_DISTANCE_NORMALIZED,
#            constants.NAME_SHAPE_INDEX]
#
# if arguments.height:
#     factors.append(constants.NAME_HEIGHT)

# Initialize logistic regression only if the user specified a data file

if arguments.logistic:
    try:
        classifier = LogisticRegressionClassifier()
        classifier.loadSelections(arguments.selection)
        classifier.load(arguments.data, stratify=False)
        classifier.createModel(arguments.score)
        #classifier.scatterPlotDataset()
    except FileNotFoundError:
        print("Regression data file %s not found\n" % arguments.regression)
        sys.exit(1)
elif arguments.knn:
   classifier = KNNClassifier()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=False)
   classifier.createModel(arguments.score)
elif arguments.tree:
   classifier = DecisionTree()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=False)
   classifier.createModel(arguments.score)
   classifier.visualize()
elif arguments.forest:
   classifier = RandomForest()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=True)
   classifier.createModel(arguments.score)
   classifier.visualize()
elif arguments.gradient:
   classifier = GradientBoosting()
   # Load selected parameters
   classifier.loadSelections(arguments.selection)
   classifier.load(arguments.data, stratify=False)
   classifier.createModel(arguments.score)
   classifier.visualize()
elif arguments.support:
    classifier = SuppportVectorMachineClassifier()
    # Load selected parameters
    classifier.loadSelections(arguments.selection)
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    classifier.visualize()
else:
    # TODO: This should be HeuristicClassifier
    classifier = Classifier()
    print("Classify using heuristics\n")

# These are the attributes that will decorate objects in the images
if constants.NAME_ALL in arguments.decorations:
    featuresToShow = [constants.NAME_AREA,
                      constants.NAME_TYPE,
                      constants.NAME_LOCATION,
                      constants.NAME_CENTER,
                      constants.NAME_SHAPE_INDEX,
                      constants.NAME_RATIO,
                      constants.NAME_REASON,
                      constants.NAME_DISTANCE_NORMALIZED,
                      constants.NAME_NAME,
                      constants.NAME_HUE,
                      constants.NAME_TYPE,
                      constants.NAME_SOLIDITY,
                      constants.NAME_ROUNDNESS,
                      constants.NAME_CONVEXITY,
                      constants.NAME_ECCENTRICITY,
                      constants.NAME_I_YIQ]
elif constants.NAME_NONE in arguments.decorations:
    featuresToShow = []
else:
    featuresToShow = [arguments.decorations]



# The contours are a bit distracting
if arguments.contours:
    featuresToShow.append(constants.NAME_CONTOUR)

imageNumber = 0
processing = False

def storeImage() -> bool:
    global imageNumber

    if not processing:
        log.info("Not collecting images (This is normal if the weeding has not started")
        return False

    if arguments.verbose:
        print("Processing image " + str(imageNumber))
    log.info("Processing image " + str(imageNumber))
    performance.start()
    try:
        rawImage = camera.capture()
    except IOError as e:
        log.fatal("Cannot capture image. ({})".format(e))
        return False

    performance.stopAndRecord(constants.PERF_ACQUIRE)

    # ImageManipulation.show("Source",image)
    veg.SetImage(rawImage)

    manipulated = ImageManipulation(rawImage, imageNumber, logger)
    fileName = logger.logImage("original", manipulated.image)

    # Send out a message to the treatment channel that an image has been taken
    message = TreatmentMessage()
    message.plan = constants.Treatment.RAW_IMAGE
    message.name = "original"
    message.url = "http://" + platform.node() + "/" + currentSessionName + "/" + fileName

    try:
        position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
        message.position = position
    except KeyError:
        log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

    messageText = message.formMessage()
    log.debug("Sending: {}".format(messageText))
    roomTreatment.sendMessage(messageText)

    imageNumber += 1
    return True

def processImage() -> bool:
    global imageNumber
    global sequence
    global previousImage

    try:

        if arguments.verbose:
            print("Processing image " + str(imageNumber))
        log.info("Processing image " + str(imageNumber))
        performance.start()
        rawImage = camera.capture()
        performance.stopAndRecord(constants.PERF_ACQUIRE)

        #ImageManipulation.show("Source",image)
        veg.SetImage(rawImage)

        manipulated = ImageManipulation(rawImage, imageNumber, logger)
        logger.logImage("original", manipulated.image)

        manipulated.mmPerPixel = mmPerPixel
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        performance.start()
        index = veg.Index(arguments.algorithm)
        performance.stopAndRecord(constants.PERF_INDEX)

        #ImageManipulation.show("index", index)
        #cv.imwrite("index.jpg", index)
        if arguments.plot:
            plot3D(index, arguments.algorithm)

        # Get the mask
        #mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)
        mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, 1, arguments.threshold)

        veg.applyMask()
        # This is the slow call
        #image = veg.GetMaskedImage()
        image = veg.GetImage()
        normalized = np.zeros_like(image)
        finalImage = cv.normalize(image,  normalized, 0, 255, cv.NORM_MINMAX)
        if arguments.mask:
            filledMask = mask.copy().astype(np.uint8)
            cv.floodFill(filledMask, None, (0,0),255)
            filledMaskInverted = cv.bitwise_not(filledMask)
            manipulated.toGreyscale()
            threshold, imageThresholded = cv.threshold(manipulated.greyscale, 0,255, cv.THRESH_BINARY_INV)
            finalMask = cv.bitwise_not(filledMaskInverted)
            logger.logImage("processed", finalImage)
            veg.ShowImage("Thresholded", imageThresholded)
            logger.logImage("inverted", filledMaskInverted)
            veg.ShowImage("Filled", filledMask)
            veg.ShowImage("Inverted", filledMaskInverted)
            veg.ShowImage("Final", finalMask)
            logger.logImage("final", finalMask)
            #plt.imshow(veg.imageMask, cmap='gray', vmin=0, vmax=1)
            plt.imshow(finalImage)
            plt.show()
            #logger.logImage("mask", veg.imageMask)
        #ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(finalImage, imageNumber, logger)
        manipulated.mmPerPixel = mmPerPixel

        # TODO: Conversion to HSV should be done automatically
        performance.start()
        manipulated.toYCBCR()
        performance.stopAndRecord(constants.PERF_YCC)
        performance.start()
        manipulated.toHSV()
        performance.stopAndRecord(constants.PERF_HSV)
        performance.start()
        manipulated.toHSI()
        performance.stopAndRecord(constants.PERF_HSI)
        performance.start()
        manipulated.toYIQ()
        performance.stopAndRecord(constants.PERF_YIQ)

        # Find the plants in the image
        performance.start()
        contours, hierarchy, blobs, largest = manipulated.findBlobs(arguments.minarea)
        performance.stopAndRecord(constants.PERF_CONTOURS)

        # The test should probably be if we did not find any blobs
        if largest == "unknown":
            logger.logImage("error", manipulated.image)
            return

        performance.start()
        manipulated.identifyOverlappingVegetation()
        performance.stopAndRecord(constants.PERF_OVERLAP)

        # Set the classifier blob set to be the set just identified
        classifier.blobs = blobs


        performance.start()
        manipulated.computeShapeIndices()
        performance.stopAndRecord(constants.PERF_SHAPES_IDX)

        performance.start()
        manipulated.computeLengthWidthRatios()
        performance.stopAndRecord(constants.PERF_LW_RATIO)

        # New image analysis based on readings here:
        # http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf

        performance.start()
        manipulated.computeCompactness()
        manipulated.computeElogation()
        manipulated.computeEccentricity()
        manipulated.computeRoundness()
        manipulated.computeConvexity()
        manipulated.computeSolidity()
        performance.stopAndRecord(constants.PERF_SHAPES)
        # End image analysis

        # Classify items by where they are in image
        # This only marks items that can't be fully seen (at edges) of image
        classifier.classifyByPosition(size=manipulated.image.shape)



        classifiedBlobs = classifier.blobs


        performance.start()
        manipulated.findAngles()
        manipulated.findCropLine()
        performance.stopAndRecord(constants.PERF_ANGLES)

        # Crop row processing
        manipulated.identifyCropRowCandidates()

        # Extract various features
        performance.start()

        # Compute the mean of the hue across the plant
        manipulated.extractImagesFrom(manipulated.hsi,0, constants.NAME_HUE, np.nanmean)
        performance.stopAndRecord(constants.PERF_MEAN)
        manipulated.extractImagesFrom(manipulated.hsv,1, constants.NAME_SATURATION, np.nanmean)

        # Discussion of YIQ can be found here
        # Sabzi, Sajad, Yousef Abbaspour-Gilandeh, and Juan Ignacio Arribas. 2020.
        # “An Automatic Visible-Range Video Weed Detection, Segmentation and Classification Prototype in Potato Field.”
        # Heliyon 6 (5): e03685.
        # The article refers to the I component as in-phase, but its orange-blue in the wikipedia description
        # of YIQ.  Not sure which is correct.

        # Compute the standard deviation of the I portion of the YIQ color space
        performance.start()
        manipulated.extractImagesFrom(manipulated.yiq,1, constants.NAME_I_YIQ, np.nanstd)
        performance.stopAndRecord(constants.PERF_STDDEV)

        # Compute the mean of the blue difference in the ycc color space
        performance.start()
        manipulated.extractImagesFrom(manipulated.ycbcr,1, constants.NAME_BLUE_DIFFERENCE, np.nanmean)
        performance.stopAndRecord(constants.PERF_MEAN)

        #Use either heuristics or logistic regression
        if arguments.logistic or arguments.knn or arguments.tree or arguments.forest or arguments.gradient:
            performance.start()
            classifier.classify()
            performance.stopAndRecord(constants.PERF_CLASSIFY)
            classifiedBlobs = classifier.blobs
        else:
            performance.start()
            classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=arguments.minratio)
            performance.stopAndRecord(constants.PERF_CLASSIFY)

        # Draw boxes around the images we found with decorations for attributes selected
        #manipulated.drawBoundingBoxes(contours)
        manipulated.drawBoxes(manipulated.name, classifiedBlobs, featuresToShow)

        #logger.logImage("cropline", manipulated.croplineImage)
        # This is using the hough transform which we abandoned as a technique
        #manipulated.detectLines()
        #TODO: Draw crop line as part of image decoration
        manipulated.drawCropline()
        #logger.logImage("crop-line", manipulated.croplineImage)
        if arguments.contours:
            manipulated.drawContours()


        # Just a test of stitching. This needs some more thought
        # we can't stitch things where there is nothing in common between the two images
        # even if there is overlap. It may just be black background after the segmentation.
        # One possibility here is to put something in the known overlap area that can them be used
        # to align the images.
        # The alternative is to use the original images and use the soil as the element that is common between
        # the two.  The worry here is computational efficiency

        if arguments.stitch:
            if previousImage is not None:
                manipulated.stitchTo(previousImage)
            else:
                previousImage = image

        # Write out the processed image
        #cv.imwrite("processed.jpg", manipulated.image)
        logger.logImage("processed", manipulated.image)
        logger.logImage("original", manipulated.original)
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # Write out the crop images so we can use them later
        if arguments.xtract:
            manipulated.extractImages(classifiedAs=constants.TYPE_DESIRED)
            for blobName, blobAttributes in manipulated.blobs.items():
                if blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED:
                    logger.logImage("crop", blobAttributes[constants.NAME_IMAGE])

        reporting.addBlobs(sequence, blobs)
        sequence = sequence + 1

        if arguments.spray:
            if arguments.verbose:
                print("Forming treatment")
            performance.start()
            treatment = Treatment(manipulated.original, manipulated.binary)
            treatment.overlayTreatmentLanes()
            treatment.generatePlan(classifiedBlobs)
            #treatment.drawTreatmentLanes(classifiedBlobs)
            performance.stopAndRecord(constants.PERF_TREATMENT)
            logger.logImage("treatment", treatment.image)

        imageNumber = imageNumber + 1

    except IOError as e:
        print("There was a problem communicating with the camera")
        print(e)
        sys.exit(1)
    except EOFError:
        print("End of input")
        return False

    if arguments.histograms:
        reporting.showHistogram("Areas", 20, constants.NAME_AREA)
        reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)

    return True

#
# Set up the processor for the image.
#
# This could be simplified a bit by having only one processing routine
# and figuring out the intent there

if arguments.grab:
    # If all we want is just to take pictures
    processor = storeImage
else:
    # This is the normal run state, where items in images are classified
    processor = processImage

def postWeedingCleanup():
    global processing

    # Copy the camera parameters used in the capture
    cameraSettings = glob.glob("camera*pfs")
    for file in cameraSettings:
        shutil.copy(file, outputDirectory)

    path = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_ROOT) + "/output"
    root = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_ROOT)


    # Move the log file over to the output directory
    try:

        source = root + "/jetson/*.log"
        destination = outputDirectory
        for data in glob.glob(source):
            shutil.move(data, destination)

    except OSError as oserr:
        log.critical("Unable to move {} to {}".format(source, destination))
        log.critical(oserr)

    finished = os.path.join(logger.directory, options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_FILENAME_FINISHED))
    log.debug("Writing session statistics to: {}".format(finished))
    try:
        with open(finished, 'w') as fp:
            fp.write("Session complete")
    except IOError as e:
        log.error("Unable to write out end of run data to file: {}".format(finished))
        log.error("{}".format(e))

    processing = False

def sendCurrentOperation(systemRoom: MUCCommunicator):
    systemMessage = SystemMessage()
    systemMessage.action = constants.Action.ACK.name
    systemMessage.operation = currentOperation
    try:
        position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
        systemMessage.position = position
    except KeyError:
        log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

    systemRoom.sendMessage(systemMessage.formMessage())

def runDiagnostics(systemRoom: MUCCommunicator, camera: Camera):
    """
    Run diagnostics for this subsystem, collecting information about the camera connectivity.
    :param systemRoom: The room to send the results
    """
    systemMessage = SystemMessage()
    systemMessage.action = constants.Action.DIAG_REPORT.name
    systemMessage.diagnostics =  camera.status.name
    systemMessage.gsdCamera = camera.gsd
    try:
        position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
        systemMessage.position = position
    except KeyError:
        log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))
    systemMessage.statusCamera = camera.status.name

    systemRoom.sendMessage(systemMessage.formMessage())

totalMovement = 0.0
keepAliveMessages = 0
movementSinceLastProcessing = 0.0
#
# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageSystemCB(conn,msg: xmpp.protocol.Message):
    global logger
    global processing
    global outputDirectory
    global currentSessionName
    global currentOperation
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
            body = msg.getBody()
            # Check if this is a real message and not just an empty keep-alive message
            if body is not None:
                log.debug("system message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
                systemMessage = SystemMessage(raw=msg.getBody())
                if systemMessage.action == constants.Action.START.name:
                    processing = True
                    currentSessionName = systemMessage.name
                    currentOperation = systemMessage.operation
                    outputDirectory = arguments.output + "/" + currentSessionName
                    log.debug("Begin processing to: {}".format(outputDirectory))
                    logger = Logger()
                    if not logger.connect(outputDirectory):
                        log.error("Unable to connect to logging. {} does not exist.".format(outputDirectory))
                if systemMessage.action == constants.Action.STOP.name:
                    log.debug("----- Stop weeding ------")
                    currentOperation = constants.Operation.QUIESCENT.name
                    postWeedingCleanup()
                if systemMessage.action == constants.Action.CURRENT.name:
                    sendCurrentOperation(roomSystem)
                if systemMessage.action == constants.Action.START_DIAG.name:
                    log.debug("Request for diagnostics")
                    runDiagnostics(roomSystem, camera)

    elif msg.getType() == "chat":
            print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

#
# The callback for messages received in the odometry room.
# When the total distance is the width of the image, grab an image and process it.
#

def messageOdometryCB(conn, msg: xmpp.protocol.Message):
    global totalMovement
    global keepAliveMessages
    global movementSinceLastProcessing
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
        body = msg.getBody()
        # Check if this is a real message and not just an empty keep-alive message
        if body is not None:
            log.debug("Distance message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
            odometryMessage = OdometryMessage(raw=body)
            log.debug("Message: {}".format(odometryMessage.data))
            totalMovement += odometryMessage.distance
            movementSinceLastProcessing += odometryMessage.distance
            # The time of the observation
            timeRead = odometryMessage.timestamp
            # Determine how old the observation is
            # The version of python on the jetson does not support time_ns, so this a bit of a workaround until I
            # get that sorted out.  Just convert the reading to milliseconds for now
            #timeDelta = (time.time() * 1000) - (timeRead / 1000000)
            timeDelta = (time.time() * 1000) - timeRead
            log.debug("Total movement: {} at time: {}. Movement since last acquisition: {} Time now is {} delta from now {} ms".
                      format(totalMovement, timeRead, movementSinceLastProcessing, time.time() * 1000, timeDelta))

            if timeDelta > 5000:
                log.debug("Old message seen.  Ignored")

            # If the movement is equal to the size of the image, take a picture
            # We need to allow for some overlap so the images can be stitched together.
            # So reduce this by the overlap factor
            elif movementSinceLastProcessing > ((1 - float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * camera.gsd):
                gsd = (1 - float( options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * camera.gsd
                log.info("Acquiring image.  Movement since last processing {} GSD {}".format(movementSinceLastProcessing,gsd))
                movementSinceLastProcessing = 0
                processor()
        else:
            # There's not much to do here for keepalive messages
            keepAliveMessages += 1
            #print("weeds: keepalive message from chatroom")
    elif msg.getType() == "chat":
            print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageTreatmentCB(conn,msg: xmpp.protocol.Message):
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
            body = msg.getBody()
            # Check if this is a real message and not just an empty keep-alive message
            if body is not None:
                log.debug("treatment message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
    elif msg.getType() == "chat":
            print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

#
#
# This method will never return.  Connect and start processing messages
#
def processMessages(communicator: MUCCommunicator):
    """
    Process messages for the chatroom -- note that this routine will never return.
    :param communicator: The chatroom communicator
    """
    log.info("Connecting to chatroom")
    processing = True

    while processing:
        try:
            communicator.connect(True)
            log.debug("Connected and processed messages")
        except XMPPServerUnreachable:
            log.warning("Unable to connect and process messages.  Will retry.")
            time.sleep(5)
            processing = True
        except XMPPServerAuthFailure:
            log.fatal("Unable to authenticate using parameters")
            processing = False

#
# Take the images -- this method will not return, only add new images to the queue
#
def takeImages(camera: CameraBasler):

    cameraConnected = False

    # Connect to the camera and take an image
    log.debug("Connecting to camera")
    cameraConnected = camera.connect()

    if cameraConnected:
        # The camera settings are stored in files like aca-2500-gc.pfs
        # This will be used for call capture parameters
        filename = camera.camera.GetDeviceInfo().GetModelName() + ".pfs"
        camera.camera.Open()
        if not camera.load(filename):
            # If we don't have a configuration, warn about this
            log.warning("Unable to locate camera config {}".format(filename))

        # Save what was used in capture
        filename = "camera-" + options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP) + "-" + filename
        camera.save(filename)
        camera.camera.Close()

        if camera.initializeCapture():
            try:
                camera.startCapturing()
            except IOError as io:
                camera.log.error(io)
            rc = 0
        else:
            rc = -1

def _takeImages(theCamera: CameraBasler):

    # Connect to the camera and take an image
    log.debug("Connecting to camera")
    theCamera.connect()

    # Load the configuration for the camera -- if we have one
    filename = theCamera.camera.GetDeviceInfo().GetModelName() + ".pfs"
    theCamera.camera.Open()
    if not theCamera.load(filename):
        # If we don't have a configuration, save what was used
        filename = "default-" + filename
        theCamera.save(filename)
    theCamera.camera.Close()

    if theCamera.initializeCapture():
        try:
            log.debug("Beginning capture")
            theCamera.startCapturing()
        except IOError as io:
            theCamera.log.fatal("I/O error encountered in start of capture")
            theCamera.log.error(io)
        rc = 0
    else:
        log.error("Capture initialization failed.")
        rc = -1

# Start up various subsystems
#
options = readINI()

currentSessionName = ""
currentOperation = constants.Operation.QUIESCENT.name

(logger, log) = startupLogger(arguments.output)
#log = logging.getLogger(__name__)

camera = startupCamera(options)
log.debug("camera started")

#cameraIP = options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP)
#camera = CameraBasler(ip = cameraIP)

(roomOdometry, roomSystem, roomTreatment) = startupCommunications(options, messageOdometryCB, messageSystemCB, messageTreatmentCB)
log.debug("Communications started")

performance = startupPerformance()
log.debug("Performance started")

# Start the worker threads, putting them in a list
threads = list()


# Start the thread that will begin acquiring images
log.debug("Start image acquisition")
acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE,target=takeImages, args=(camera,))
threads.append(acquire)
acquire.start()


log.debug("Starting odometry receiver")
#generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=processMessages, args=(roomOdometry,))
generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=roomOdometry.processMessages, args=())
generator.daemon = True
threads.append(generator)
generator.start()

log.debug("Starting system receiver")
#sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=processMessages, args=(roomSystem,))
sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=roomSystem.processMessages, args=())
sys.daemon = True
threads.append(sys)
sys.start()

log.debug("Starting treatment thread")
#treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=processMessages, args=(roomTreatment,))
treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=roomTreatment.processMessages, args=())
treat.daemon = True
threads.append(treat)
treat.start()


# Wait for the workers to finish
for index, thread in enumerate(threads):
    thread.join()


performance.cleanup()

# Not quite right here to get the list of all blobs from the reporting module
#classifier.train(reporting.blobs)

result, reason = reporting.writeSummary()

if not result:
    print(reason)
    sys.exit(1)
else:
    sys.exit(0)