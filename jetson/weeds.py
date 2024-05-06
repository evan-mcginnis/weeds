#
# W E E D S
#

import argparse
import glob
import platform
import sys
import threading
from typing import Callable
from pathlib import Path

import configparser

from bson import ObjectId
from hashlib import sha1
import numpy as np

try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import pandas as pd

    supportsPlotting = True
except ImportError:
    print("Unable to import plotting libraries.")
    supportsPlotting = False

import scipy.ndimage

import logging.config
import shutil

import xmpp
# from xmpp import protocol

import shortuuid

from urllib.request import url2pathname
from urllib.parse import urlparse

from pypylon import pylon
# This does not work
# from CameraFile import CameraFile, CameraBasler

from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger
from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, \
    GradientBoosting, SuppportVectorMachineClassifier, LDA, MLP, ExtraTrees
from Classifier import Subset
from OptionsFile import OptionsFile
from Reporting import Reporting
from Treatment import Treatment
from MUCCommunicator import MUCCommunicator
from MQCommunicator import ClientMQCommunicator
from Messages import OdometryMessage, SystemMessage, TreatmentMessage
from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure
from CameraDepth import CameraDepth
from ProcessedImage import Images
from Enrich import Enrich
from Context import Context
from Diagnostics import Diagnostics
from CameraFile import CameraFile
from GLCM import GLCM
from Persistence import Mongo
from Persistence import Disk
from Persistence import RawImage
from Persistence import Blob
from Factors import Factors
from Metadata import Metadata
from Classifier import ImbalanceCorrection

SQUARE_SIZE = 40

# Operations
OPERATION_VISUALIZE         = "visualize"
OPERATION_NORMAL            = "normal"
OPERATION_EVALUATE          = "evaluate"
OPERATION_GENERATE          = "generate"
allOperations = [OPERATION_NORMAL, OPERATION_VISUALIZE, OPERATION_EVALUATE, OPERATION_GENERATE]

#
# W A R N I N G
#
# This is the bit that I can't get to work correctly
# The Basler methods work just fine from the test routines if I put the camera logic
# in another file, but not if it is imported

# from CameraBasler import CameraBasler
# from CameraBasler import ConfigurationEventPrinter
# from CameraBasler import ImageEvents
# from CameraBasler import SampleImageEventHandler


# from Selection import Selection

#
# C A M E R A S
#
# TODO: Move to Camera.py file
# This is very sloppy work, and has completely defeated me, so I give up
# This works just fine in another file, but fails whenever it is imported,
# so I'm giving up and copying it here

import logging.config

import numpy as np

import pypylon.pylon

from PIL import Image

from Performance import Performance

### B A S L E R  S T A R T ###

from Camera import Camera
from pypylon import pylon
from pypylon import genicam
import pypylon.pylon
from pypylon import _genicam

import logging
import logging.config
from collections import deque
import os

import time
from datetime import datetime

import cv2 as cv

import constants
from ProcessedImage import ProcessedImage


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
        # log.debug("OnImageGrabbed event for device: {}".format(camera.GetDeviceInfo().GetModelName()))

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            # log.debug("Image grabbed successfully")
            start = time.time()
            # Convert the image grabbed to something we like
            # image = CameraBasler(grabResult)
            # self.log.debug(f"Basler image converted time: {time.time() - start} s")
            image = CameraBasler.convert(grabResult)
            img = image.GetArray()
            # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
            # We will mark the images based on when we got them -- ideally, this should be:
            # timestamped = ProcessedImage(img, grabResult.TimeStamp)
            timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))

            cameraNumber = camera.GetCameraContext()
            camera = Camera.cameras[cameraNumber]
            # log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
            camera._images.append(timestamped)

            # print("SizeX: ", grabResult.GetWidth())
            # print("SizeY: ", grabResult.GetHeight())
            # img = grabResult.GetArray()
            # print("Gray values of first row: ", img[0])
            # print()
        else:
            log.error(
                "Image Grab error code: {} {}".format(grabResult.GetErrorCode(), grabResult.GetErrorDescription()))


# Example of an image event handler.
class SampleImageEventHandler(pypylon.pylon.ImageEventHandler):
    def OnImageGrabbed(self, _camera, grabResult):
        print("CSampleImageEventHandler::OnImageGrabbed called.")
        print()
        print()


class CameraBasler(Camera):
    # Initialize the converter for images
    # The images stream of in YUV color space.  An optimization here might be to make
    # both formats available, as YUV is something we will use later

    _converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    _converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    # Temporary -- the setting is already in place on the camera
    # _converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    _configurationEvents = ConfigurationEventPrinter()
    _imageEvents = ImageEvents()

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
        self._countOfImagesToGrab = 1
        self._useCallbacks = False

        if constants.KEYWORD_GSD in kwargs:
            self._gsd = kwargs[constants.KEYWORD_GSD]
        else:
            self.log.info(
                "The GSD keyword is not specified with {}. Calculated instead.".format(constants.KEYWORD_GSD))
            # This is just a placeholder
            self._gsd = 0.5

        # Assume a GigE camera for now
        if constants.KEYWORD_IP in kwargs:
            self._ip = kwargs[constants.KEYWORD_IP]
        else:
            self.log.fatal(
                "The IP address of the camera must be specified with the keyword {}".format(constants.KEYWORD_IP))

        # Set up the capture strategy for this camera
        if constants.KEYWORD_CAPTURE_STRATEGY in kwargs:
            self._captureType = kwargs[constants.KEYWORD_CAPTURE_STRATEGY]
            self.log.debug("Capture type: {}".format(self._captureType))
        else:
            self._captureType = constants.CAPTURE_STRATEGY_QUEUED
            self.log.debug("Default Capture type: {}".format(self._captureType))

        if constants.KEYWORD_CONFIGURATION_EVENTS in kwargs:
            self.log.debug("Using supplied configuration event printer")
            self._configurationEvents = kwargs[constants.KEYWORD_CONFIGURATION_EVENTS]
            self._useCallbacks = True

        if constants.KEYWORD_IMAGE_EVENTS in kwargs:
            self.log.debug("Using supplied image event printer")
            self._imageEvents = kwargs[constants.KEYWORD_IMAGE_EVENTS]
            self._useCallbacks = True

        super().__init__(**kwargs)

    @property
    def ip(self):
        return self._ip

    @classmethod
    def convert(cls, grabResult):
        """
        Converts the grab result into the format expected by the rest of the system
        :param grabResult: A grab from the basler camera
        :return:
        """
        start = time.time()
        image = CameraBasler._converter.Convert(grabResult)
        # self.log.debug(f"Image conversion took: {time.time() - start} seconds")
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
                    self.log.fatal("Error encountered in attaching camera")
                    self.log.fatal("{}".format(e))
                    self._status = constants.OperationalStatus.FAIL
                # self._camera.MaxNumBuffer = 100
                try:
                    self._camera.Open()
                    self.log.info("Using device {} at {}".format(self._camera.GetDeviceInfo().GetModelName(),
                                                                 dev_info.GetIpAddress()))
                    self._camera.AcquisitionMode.SetValue('Continuous')
                    self._connected = True
                    self._status = constants.OperationalStatus.OK
                except Exception as e:
                    self.log.fatal("Error encountered in opening camera")
                    self.log.fatal("{}".format(e))
                    self._status = constants.OperationalStatus.FAIL
                # This shows how to get the list of what is available as attributes.  Not particularly useful for what
                # we need here
                # info = pylon.DeviceInfo()
                # info = self._camera.GetDeviceInfo()
                # tlc = pylon.GigETransportLayer()
                # tlc = self._camera.GetTLNodeMap()
                #
                # properties = info.GetPropertyNames()

                # self.log.debug("Current counter {}".format())
                break

        if not self._connected:
            self.log.error("Failed to connect to camera")
            # raise EnvironmentError("No GigE device found")

        return self._connected

    def initializeCapture(self):

        initialized = False
        if self._useCallbacks:
            try:
                # Register the standard configuration event handler for enabling software triggering.
                # The software trigger configuration handler replaces the default configuration
                # as all currently registered configuration handlers are removed by setting the registration mode to RegistrationMode_ReplaceAll.
                self._camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(),
                                                   pylon.RegistrationMode_ReplaceAll,
                                                   pylon.Cleanup_Delete)

                # Originally
                # self._camera.RegisterConfiguration(ConfigurationEventPrinter(),
                #                                    pylon.RegistrationMode_Append,
                #                                    pylon.Cleanup_Delete)

                self._camera.RegisterConfiguration(self._configurationEvents,
                                                   pylon.RegistrationMode_Append,
                                                   pylon.Cleanup_Delete)

                # The image event printer serves as sample image processing.
                # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
                # results must be created and registered.
                # Originally
                # self._camera.RegisterImageEventHandler(ImageEvents(),
                #                                        pylon.RegistrationMode_Append,
                #                                        pylon.Cleanup_Delete)

                self._camera.RegisterImageEventHandler(self._imageEvents,
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
        else:
            self.log.debug("Initialized direct capture")
            initialized = True

        return initialized

    def initialize(self):
        """
        Set the camera parameters to reflect what we want them to be.
        :return:
        """

        if not self._connected:
            raise IOError("Camera is not connected.")

        self.log.debug("Camera initialized")

    def startCapturing(self):
        """
        Start capturing using callbacks when the image is obtained
        """
        # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
        # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
        # The GrabStrategy_OneByOne default grab strategy is used.
        try:
            # self.camera.Open()
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
            self.log.debug("Start Capturing with OneByOne Strategy")
        except _genicam.RuntimeException as e:
            self.log.fatal("Failed to open the camera and start grabbing.")
            self.log.fatal("{}".format(e))

        # If we immediately start waiting for the trigger, we get an error
        time.sleep(3)
        self._capturing = True

        # This is for a dummy loop for when we use the capture loop of the camera
        while self._capturing:
            time.sleep(10)
            self.log.debug("Dummy capture of Basler RGB")

        # This is the loop for when the code supplies the grab loop
        # while self._capturing:
        #     try:
        #         if not self.camera.IsGrabbing():
        #             self.log.error("Camera is not grabbing")
        #         else:
        #             for i in range(5):
        #                 if self.camera.WaitForFrameTriggerReady(1000, pylon.TimeoutHandling_ThrowException):
        #                     self.camera.ExecuteSoftwareTrigger()
        #     except _genicam.TimeoutException as e:
        #         self.log.fatal("Timeout from camera {}".format(e))
        #         time.sleep(0.5)
        #     except _genicam.RuntimeException as e:
        #         if not self._capturing:
        #             self.log.warning("Errors encountered in shutdown.  This is normal")
        #         else:
        #             self.log.error("Unexpected errors in capture")
        #             self.log.error("Device: {}".format(self._camera.GetDeviceInfo().GetModelName()))
        #             self.log.error("{}".format(e))
        #     except Exception as e:
        #         self.log.error("Unable to execute wait for trigger")
        #         self.log.error(e)

    def start(self):
        """
        Begin capturing images and store them in a queue for later retrieval.
        """
        self._camera.Open()
        self._camera.MaxNumBuffer = 40
        self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        i = 0
        self.log.debug('Starting to acquire')
        while self._camera.IsGrabbing():
            try:
                t0 = time.time()
                grab = self._camera.RetrieveResult(700, pylon.TimeoutHandling_ThrowException)
            except _genicam.TimeoutException:
                self.log.error("Basler camera timeout seen")
                continue
            if grab.GrabSucceeded():
                self.log.debug(f'Acquired frames in {time.time() - t0} seconds')
                i += 1
            if grab.GrabSucceeded():
                # self.log.debug("Basler Image grabbed")
                # Convert the image grabbed to something we like
                start = time.time()

                # image = CameraBasler.convert(grab)
                # self.log.debug(f"Basler image converted time: {time.time() - start} s")
                # img = image.GetArray()

                # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
                # We will mark the images based on when we got them -- ideally, this should be:
                # timestamped = ProcessedImage(img, grabResult.TimeStamp)
                # Orignally
                # timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
                # Create a processed image that has not yet been converted
                # Use the pylon methods for everything
                # img = pylon.PylonImage()
                # img.AttachGrabResultBuffer(grab)
                img = grab.Array
                grab.Release()
                # timestamped = ProcessedImage(constants.Capture.RGB, grab, round(time.time() * 1000))
                # Temporary -- attach the image, not the grab
                timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
                timestamped.type = constants.ImageType.BASLER_RAW

                cameraNumber = self._camera.GetCameraContext()
                # self._camera = Camera.cameras[cameraNumber]
                log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
                self.log.debug(f"Grabbed image processed and enqueued in {time.time() - start:.8f} seconds")
                self._images.append(timestamped)

                # print("SizeX: ", grabResult.GetWidth())
                # print("SizeY: ", grabResult.GetHeight())
                # img = grabResult.GetArray()
                # print("Gray values of first row: ", img[0])
                # print()
            else:
                self.log.error("Image Grab error code: {} {}".format(grab.GetErrorCode(), grab.GetErrorDescription()))

        self._camera.Close()

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

    def capture(self) -> ProcessedImage:
        """
            Capture a single image from the camera.
            Requires calling the connect() method before this call.

            If the image is in the queue, it will be served from there -- otherwise the method will retrieve if
            synchronously from the camera.

            :return:
            The image as a numpy array
            """
        processed = None

        if not self._connected:
            raise IOError("Camera is not connected")

        # This implementation will just grab the image directly
        if self._strategy == constants.CAPTURE_STRATEGY_LIVE:
            self._camera.MaxNumBuffer = 10

            # Start the grabbing of c_countOfImagesToGrab images.
            # The camera device is parameterized with a default configuration which
            # sets up free-running continuous acquisition.
            self._camera.StartGrabbingMax(self._countOfImagesToGrab)
            while self._camera.IsGrabbing():
                # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
                grabResult = self._camera.RetrieveResult(200, pylon.TimeoutHandling_ThrowException)

                # Image grabbed successfully?
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    print("SizeX: ", grabResult.Width)
                    print("SizeY: ", grabResult.Height)
                    img = grabResult.Array
                    image = self._converter.Convert(grabResult)
                    img = image.GetArray()
                    # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
                    # We will mark the images based on when we got them -- ideally, this should be:
                    processed = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))

                else:
                    print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
                grabResult.Release()

        # Otherwise, get it from the queue
        else:
            # If there are no images in the queue, just wait for one.
            while len(self._images) == 0:
                self.log.error("Basler image queue is empty. Wait for a new image to appear")
                time.sleep(0.1)

            # The image we want is the one closest to the current time. The queue may contain a bunch of older images
            processed = self._images.popleft()

            # The image is not yet converted to a form we can process
            # grab = processed.image
            # start = time.time()
            # image = CameraBasler.convert(grab)
            # self.log.debug(f'Basler image conversion took {time.time() - start} seconds')
            # processed.image = image.GetArray()

            # The timestamp is in milliseconds
            timestamp = processed.timestamp / 1000
            self.log.debug(
                "Image captured at UTC: {}".format(datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')))
        return processed

    def getResolution(self) -> ():
        raise NotImplementedError
        return w, h

    # This should be part of the calibration procedure
    def getMMPerPixel(self) -> float:
        return 0.0

    @property
    def camera(self) -> pylon.InstantCamera:
        return self._camera

    @camera.setter
    def camera(self, openedCamera: pylon.InstantCamera):
        self._camera = openedCamera

    def _grab(self) -> ProcessedImage:
        """
            Grab the image from the camera
            :return: ProcessedImag
            """

        try:
            self.log.debug("Grab start")
            grabResult = pypylon.pylon.GrabResult(
                self._camera.RetrieveResult(constants.TIMEOUT_CAMERA, pylon.TimeoutHandling_ThrowException))
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
            # self.log.debug("Image grab succeeded at timestamp " + str(grabResult.TimeStamp))
            pass
        else:
            raise IOError("Failed to grab image. Pylon error code: {}".format(grabResult.GetErrorCode()))

        image = self._converter.Convert(grabResult)
        img = image.GetArray()
        # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
        # We will mark the images based on when we got them -- ideally, this should be:
        # timestamped = ProcessedImage(img, grabResult.TimeStamp)
        timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
        return timestamped

    def save(self, filename: str) -> bool:
        """
            Save the camera settings
            :param filename: The file to contain the settings
            :return: True on success
            """
        # self._camera.Open()
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

        # self._camera.Open()
        # If the camera configuration exists, use that, otherwise warn
        if os.path.isfile(filename):
            self.log.info("Using saved camera configuration: {}".format(filename))
            try:
                pylon.FeaturePersistence.Load(filename, self._camera.GetNodeMap(), True)
            except _genicam.RuntimeException as geni:
                self.log.error("Unable to load configuration: {}".format(geni))

            loaded = True
        else:
            self.log.warning(
                "Unable to find configuration file: {}.  Camera configuration unchanged".format(filename))
        return loaded


### B A S L E R  E N D ###

# #
# # E N D  C A M A R A S
# #
################################
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

    logger = ImageLogger()
    if not logger.connect(outputDirectory):
        print("Unable to connect to logging. ./output does not exist.")
        sys.exit(1)
    return logger, log

def readINI() -> OptionsFile:
    options = OptionsFile(arguments.ini)
    options.load()
    return options

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
parser.add_argument("-ch", "--hull", action="store_true", default=False, help="Show convex hull instead of bounding box")
parser.add_argument("-cl", "--cropline", action="store_true", default=False, help="Detect and show cropline in image")
parser.add_argument("-d", "--decorations", action="store", type=str, default="all",
                    help="Decorations on output images (all and none are shortcuts)")
parser.add_argument("-plain", "--plain", action="store_true", default=False, help="Plain")

# Database specifications
parser.add_argument("-db", "--database", action="store_true", required=False, default=False, help="Store results in DB")
parser.add_argument("-host", "--host", action="store", required=False, help="DB Host")
parser.add_argument("-port", "--port", type=int, action="store", required=False, help="DB Port")
parser.add_argument("-dbname", "--dbname", action="store", required=False, help="DB Name")

parser.add_argument("-alt", "--altitude", action="store", required=False, default=0.0, type=float, help="Override altitude in image EXIF")

# functionGroup = parser.add_mutually_exclusive_group()
# functionGroup.add_argument("-w", "--weeds", action="store_true", default=False, help="Classify and treat weeds")
# functionGroup.add_argument("-e", "--emitter", action="store_true", default=False, help="Assess treatment post emitters")

parser.add_argument("-cr", "--crop", action="store", required=False, default="lettuce", choices=["cotton", "guayule", "spinach", "cantaloupe", "unknown"], help="Crop in the image")
parser.add_argument("-df", "--data", action="store",
                    help="Name of the data in CSV for use in logistic regression or KNN")
parser.add_argument("-e", "--edge", action="store_true", default=False, help="Ignore items at edge of image")
parser.add_argument("-hg", "--histograms", action="store_true", default=False, help="Show histograms")
parser.add_argument("-he", "--height", action="store_true", default=False, help="Consider height in scoring")
parser.add_argument('-i', '--input', action="store", required=False, help="Images directory")


parser.add_argument('-s', '--split', action="store", required=False, default=0.4, type=float, help="Split for training")

parser.add_argument("-gr", "--grab", action="store_true", default=False,
                    help="Just grab images. No processing")

# Imbalance in data
parser.add_argument('-mi', '--minority', action="store", required=False, type=float, default=1.0, help="Adjust minority class to represent this percentage")
parser.add_argument("-ic", '--correct', action="store_true", required=False, default=False, help="Correct data imbalance")
parser.add_argument("-ia", '--imbalance', action="store", required=False, choices=Classifier.correctionAlgorithms(), default='SMOTE', help="Data imbalance algorithm")
parser.add_argument("-sub", "--subset", action="store", required=False, default=Subset.TRAIN.name.lower(), choices=[i.name.lower() for i in Subset])

group = parser.add_mutually_exclusive_group()
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME,
                    help="Options INI")
group.add_argument("-k", "--knn", action="store_true", default=False,
                   help="Predict using KNN. Requires data file to be specified")
group.add_argument("-lda", "--lda", action="store_true", default=False,
                   help="Predict using Linear Discriminate Analysis. Requires data file to be specified")
group.add_argument("-l", "--logistic", action="store_true", default=False,
                   help="Predict using logistic regression. Requires data file to be specified")
group.add_argument("-dt", "--tree", action="store_true", default=False,
                   help="Predict using decision tree. Requires data file to be specified")
group.add_argument("-f", "--forest", action="store_true", default=False,
                   help="Predict using random forest. Requires data file to be specified")
group.add_argument("-g", "--gradient", action="store_true", default=False,
                   help="Predict using gradient boosting. Requires data file to be specified")
group.add_argument("-svm", "--support", action="store_true", default=False,
                   help="Predict using support vector machine. Requires data file to be specified")
group.add_argument("-mlp", "--perceptron", action="store_true", default=False,
                   help="Predict using multi-layer perceptron. Requires data file to be specified")
group.add_argument("-extra", "--extra", action="store_true", default=False,
                   help="Predict using extra trees. Requires data file to be specified")
parser.add_argument("-im", "--image", action="store", default=200, type=int, help="Horizontal length of image")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-ma", "--minarea", action="store", default=500, type=int, help="Minimum area of a blob")
parser.add_argument("-mr", "--minratio", action="store", default=5, type=int, help="Minimum size ratio for classifier")
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv",
                    help="Name of performance file")
parser.add_argument("-r", "--results", action="store", default="results", help="Name of results file")
#parser.add_argument("-s", "--stitch", action="store_true", help="Stitch adjacent images together")
parser.add_argument("-sc", "--score", action="store_true", help="Score the prediction method")
# Not needed anymore, as parameter selection is moved to the .INI file
#parser.add_argument("-se", "--selection", action="store", default="all-parameters.csv", help="Parameter selection file")
parser.add_argument("-sp", "--spray", action="store_true", help="Generate spray treatment grid")
parser.add_argument("-spe", "--speed", action="store", default=1, type=int, help="Speed in meters per second")
parser.add_argument("-stand", "--standalone", action="store_true", default=False,
                    help="Run standalone and just process the images")
parser.add_argument("-st", "--strategy", action="store", required=False, default="CARTOON", help="Blob strategy")
parser.add_argument("-di", "--direction", action="store", type=int, required=False, default=1, help="Threshold for index mask")
parser.add_argument("-t", "--threshold", action="store", required=False, help="Threshold for index mask")
parser.add_argument("-op", "--operation", action="store", default=OPERATION_NORMAL, choices=allOperations, help="Operation")
parser.add_argument("-x", "--xtract", action="store_true", default=False, help="Extract each crop plant into images")

arguments = parser.parse_args()


# This creates a unique session -- nothing special here & could be changed to something human readable
currentSessionName = shortuuid.uuid()
outputDirectory = arguments.output + "/" + currentSessionName + "/"

factorsToExtract = []

decorations = [item for item in arguments.decorations.split(',')]

(logger, log) = startupLogger(outputDirectory)

if (arguments.logistic or arguments.knn or arguments.tree or arguments.forest or arguments.lda or arguments.perceptron) and arguments.data is None:
    print("Data file is not specified.")
    sys.exit(1)

options = readINI()


#
# I N P U T
#
# Either a specific test set is specified or a split is, but not both -- let argparse take care of that.

# Ensure if the split is set it is between 0 and 1
if arguments.split is not None:
    if arguments.split > 1.0 or arguments.split < 0.0:
        print(f"Split must be between 0 and 1.")
        sys.exit(-1)
    else:
        split = 0.4

# If neither the split or a test set is specified, set the split to a default value
if arguments.split is None and arguments.test is None:
    split = 0.4

#
# D E P T H  C A M E R A
#
def startupRGBDepthCamera(options: OptionsFile) -> CameraDepth:
    """
    Starts the attached depth camera
    :return: The depth camera instance or None if the camera cannot be found.
    """

    return None


def startupCamera(options: OptionsFile) -> Camera:
    theCamera = None
    # The camera that takes crop images
    if arguments.input is not None:
        # Get the images from a directory
        theCamera = CameraFile(directory=arguments.input, TYPE=constants.ImageType.RGB.name)
        theCamera.gsd = int(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_IMAGE_WIDTH))
    else:
        log.error("Connecting to physical camera")



    # Set the ground sampling distance, so we know when to take a picture
    theCamera.gsd = int(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_IMAGE_WIDTH))

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
# Z e r o M Q  C O M M U N I C A T I O N S
#
def startupMQCommunications(options: OptionsFile, processMessage: Callable) -> ClientMQCommunicator:
    """
    Startup communication to the MQ server.  This sets the command to retrieve odometry readings

    :param options:
    :param processMessage: a callback for each message received
    :return: ClientMQCommunicator
    """
    try:
        communicator = ClientMQCommunicator(
            SERVER=options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_SERVER),
            PORT=constants.PORT_ODOMETRY)
    except KeyError:
        log.error(
            "Unable to find {}/{} in ini file".format(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_SERVER))
        communicator = None

    communicator.connect()
    communicator.callback = processMessage

    return communicator


#
# X M P P   C O M M U N I C A T I O N S
#
# def process(conn,msg):# xmpp.protocol.Message):
#     log.debug("Callback for distance")
#     return

def startupCommunications(options: OptionsFile, callbackOdometer: Callable, callbackSystem: Callable,
                          callbackTreatment: Callable) -> ():
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
                                    options.option(constants.PROPERTY_SECTION_XMPP,
                                                   constants.PROPERTY_DEFAULT_PASSWORD),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
                                    callbackTreatment,
                                    None)
    # print("XMPP communications started")

    return odometryRoom, systemRoom, treatmentRoom


#
# L O G G E R
#




def resample(index: np.ndarray, targetX: int, targetY: int) -> np.ndarray:
    # Hardcode this for now -- depth is 1280x720, and we want 1920x1080

    # z = (1920 / 1280, 1080 / 720)
    z = (targetY / 1920, targetX / 1080)

    transformed = scipy.ndimage.zoom(index, z, order=0)
    return transformed


# The plotly version
def plot3D(index: np.ndarray, planeLocation: int, title: str):
    if not supportsPlotting:
        log.error("Unable to produce plots on this platform")
        return

    # I can get plotly to work only with square arrays, not rectangular, so just take a subset
    height, width = index.shape
    subsetLength = min(height, width)
    if subsetLength > 2100:
        subsetLength = 2100
    subset = index[0:subsetLength, 0:subsetLength]
    log.debug("Index is {}".format(index.shape))
    log.debug("Subset is {}".format(subset.shape))
    xi = np.linspace(0, subset.shape[0], num=subset.shape[0])
    yi = np.linspace(0, subset.shape[1], num=subset.shape[1])

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=subset)])

    # The plane represents the threshold value
    x1 = np.linspace(0, subsetLength, subsetLength)
    y1 = np.linspace(0, subsetLength, subsetLength)
    z1 = np.ones(subsetLength) * planeLocation
    plane = go.Surface(x=x1, y=y1, z=np.array([z1] * len(x1)).T, opacity=0.5, showscale=False, showlegend=False)

    fig.add_traces([plane])


    # Can't get these to work
    # fig = go.Figure(data=[go.Mesh3d(x=xi, y=yi, z=subset, color='lightpink', opacity=0.50)])
    # fig = go.Figure(data=go.Isosurface(x=xi, y=yi,z=subset, isomin=-1, isomax=1))

    fig.update_layout(title=title, autosize=False,
                      width=1000, height=1000,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show()


# The matplotlib version is very slow to visualize and then rotate.
def plot3Dmatplotlib(index, title):
    if not supportsPlotting:
        log.error("Unable to produce plots on this platform")
        return

    downsampled = resample(index, 720, 1280)

    yLen, xLen = downsampled.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    log.debug("3D plot x: {} y: {}".format(x, y))

    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 10))
    axes = fig.gca(projection='3d')
    plt.title(title)
    axes.scatter(x, y, downsampled, c=downsampled.flatten(), cmap='BrBG', s=0.25)
    plt.show()
    cv.waitKey()




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

selections = [e.strip() for e in options.option(constants.PROPERTY_SECTION_IMAGE_PROCESSING, constants.PROPERTY_FACTORS).split(',')]
log.debug(f"Selected parameters from INI file: {selections}")

mlApproach = "unknown"

if arguments.logistic:
    try:
        classifier = LogisticRegressionClassifier()
        #classifier.loadSelections(arguments.selection)
        classifier.selections = selections
        classifier.correct = arguments.correct
        classifier.correctionAlgorithm = ImbalanceCorrection[arguments.imbalance]
        classifier.minority = arguments.minority
        classifier.correctSubset = Subset[arguments.subset.upper()]
        log.debug(f"Loaded selections: {classifier.selections}")
        classifier.load(arguments.data, stratify=False)
        classifier.createModel(arguments.score)
        mlApproach = "lr"
        # classifier.scatterPlotDataset()
    except FileNotFoundError:
        print("Regression data file %s not found\n" % arguments.regression)
        sys.exit(1)
elif arguments.knn:
    classifier = KNNClassifier()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    mlApproach = "knn"
elif arguments.tree:
    classifier = DecisionTree()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    mlApproach = "decision"
    #classifier.visualize()
elif arguments.forest:
    classifier = RandomForest()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=True)
    classifier.createModel(arguments.score)
    #classifier.visualize()
    mlApproach = "forest"
elif arguments.gradient:
    classifier = GradientBoosting()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    #classifier.visualize()
    mlApproach = "gradient"
elif arguments.support:
    classifier = SuppportVectorMachineClassifier()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    #classifier.visualize()
    mlApproach = "svm"
elif arguments.lda:
    classifier = LDA()
    # Load selected parameters
    # classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    # classifier.visualize()
    mlApproach = "lda"
elif arguments.perceptron:
    classifier = MLP()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    #classifier.visualize()
    mlApproach = "mlp"
elif arguments.extra:
    classifier = ExtraTrees()
    # Load selected parameters
    #classifier.loadSelections(arguments.selection)
    classifier.selections = selections
    classifier.load(arguments.data, stratify=False)
    classifier.createModel(arguments.score)
    #classifier.visualize()
    mlApproach = "extra"
else:
    # TODO: This should be HeuristicClassifier
    classifier = Classifier()
    print("Classify using heuristics\n")


# If this is just to visualize, exit afterwards
if arguments.operation == OPERATION_VISUALIZE:
    #classifier.visualize()
    classifier.visualizeFolds()

    sys.exit(0)

# Likewise, if this is just to evaluate the model
if arguments.operation == OPERATION_EVALUATE:
    sys.exit(0)

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
                      constants.NAME_I_YIQ,
                      constants.NAME_DIST_TO_LEADING_EDGE]
elif constants.NAME_NONE in arguments.decorations:
    featuresToShow = []
else:
    featuresToShow = [arguments.decorations]

# The contours are a bit distracting
if arguments.contours:
    featuresToShow.append(constants.NAME_CONTOUR)

imageNumberBasler = 0
imageNumberIntel = 0
processing = False
processingOdometry = False


def nullProcessor(contextForImage: Context, captureType: constants.Capture,
                  capturePosition: constants.PositionWithEmitter) -> bool:
    """
    The null processor ignores context.  Just puts an empty file on disk for the capture type.
    :param contextForImage:
    :param captureType:
    :return:
    """
    global imageNumberBasler

    if not processing:
        log.debug("Not collecting images (This is normal if the weeding has not started")
        return False

    if captureType == constants.Capture.RGB:
        imageName = "{}-{}-{:05d}".format(constants.FILENAME_RAW,
                                          options.option(constants.PROPERTY_SECTION_GENERAL,
                                                         constants.PROPERTY_POSITION),
                                          imageNumberBasler)
    else:
        imageName = "{}-{}-{:05d}".format(constants.FILENAME_INTEL_RGB,
                                          options.option(constants.PROPERTY_SECTION_GENERAL,
                                                         constants.PROPERTY_POSITION),
                                          imageNumberBasler)

    dummyPath = os.path.join(logger.directory, imageName)

    # Create an empty file so we can keep track of image counts
    with open(dummyPath, "w") as fp:
        pass

    imageNumberBasler += 1
    return True


def storeImage(contextForImage: Context, captureType: constants.Capture,
               capturePosition: constants.PositionWithEmitter) -> bool:
    global imageNumberBasler
    global imageNumberIntel
    global intelCaptureType

    if not processing:
        # log.debug("Not collecting images (This is normal if the weeding has not started")
        return False

    log.info("Storing image {} type {} position {}".format(imageNumberBasler, captureType, capturePosition))

    # Depending on the position of the camera, assign different names to the images
    if capturePosition == constants.PositionWithEmitter.PRE:
        imageNamePrefix = constants.FILENAME_RAW
    elif capturePosition == constants.PositionWithEmitter.POST:
        imageNamePrefix = constants.FILENAME_POST_EMITTER
    else:
        imageNamePrefix = "unknown-position"

    start = time.time()
    if captureType == constants.Capture.DEPTH_RGB:
        pass

    elif captureType == constants.Capture.RGB:
        performance.start()
        try:
            processed = camera.capture()
            rawImage = processed.image
        except IOError as e:
            log.fatal("Cannot capture Basler image. ({})".format(e))
            return False

        performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        # Set the context and enqueue the image
        imageName = "{}-basler-{}-{:05d}".format(imageNamePrefix, options.option(constants.PROPERTY_SECTION_GENERAL,
                                                                                 constants.PROPERTY_POSITION),
                                                 imageNumberBasler)
        imagePath = os.path.join(logger.directory, imageName)

        processed = ProcessedImage(constants.Capture.RGB, rawImage, 0)
        processed.urlFilename = currentSessionName + "/" + imageName
        processed.make = contextForImage.make
        processed.model = contextForImage.model
        processed.exposure = contextForImage.exposure
        processed.latitude = contextForImage.latitude
        processed.longitude = contextForImage.longitude
        processed.filename = imagePath
        # Enqueue the image to be written out later
        log.debug("Enqueue processed image {} from {} for enrichment".format(processed.filename, processed.make))
        rawImages.enqueue(processed)
        log.debug(f"Added Basler RGB image to queue in {time.time() - start} seconds")

        imageNumberBasler += 1

    log.debug("Image Processing time: {} seconds".format(time.time() - start))

    return True


mmPerPixel = 0.01


def processImage(contextForImage: Context) -> constants.ProcessResult:
    global imageNumberBasler
    global sequence
    global previousImage
    global currentSessionName

    try:
        log.info("Processing image " + str(imageNumberBasler))
        performance.start()

        # Attempt to capture the image.
        try:
            processed = camera.capture()
            rawImage = processed.image
        except EOFError as eof:
            # This case is where we just hit the end of an image set from disk
            log.info("Encountered end of image set")
            return constants.ProcessResult.EOF
        except IOError as io:
            # This is the case where something went wrong with a grab from a camera
            log.error("Encountered I/O Error {}".format(io))
            return constants.ProcessResult.INTERNAL_ERROR

        performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        # ImageManipulation.show("Source",image)
        veg.SetImage(rawImage)
        veg.TemporaryLoad(processed.source)

        performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        manipulated = ImageManipulation(rawImage, imageNumberBasler, logger)



        rawImage = logger.logImage(constants.FILENAME_RAW, manipulated.image)

        # manipulated.mmPerPixel = mmPerPixel
        # ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        performance.start()
        index = veg.Index(arguments.algorithm)
        performance.stopAndRecord(constants.PERF_INDEX)



        # Get the mask
        # mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)
        if arguments.threshold is not None:
            try:
                thresholdForMask = float(arguments.threshold)
            except ValueError:
                # Must be OTSU or TRIANGLE
                if arguments.threshold in VegetationIndex.thresholdChoices:
                    thresholdForMask = arguments.threshold
                else:
                    print(f"Threshold must be one of {VegetationIndex.thresholdChoices} or a valid float")
                    sys.exit(-1)
        else:
            thresholdForMask = None

        # Orignal before direction was configurable
        #mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, 1, thresholdForMask)
        mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, arguments.direction, thresholdForMask)
        log.debug(f"Use threshold: {threshold}")
        normalized = np.zeros_like(mask)
        finalMask = cv.normalize(mask, normalized, 0, 255, cv.NORM_MINMAX)
        logger.logImage("mask", finalMask)

        # ImageManipulation.show("index", index)
        # cv.imwrite("index.jpg", index)
        if arguments.plot:
            plot3D(index, threshold, arguments.algorithm)

        veg.applyMask()
        # This is the slow call
        # image = veg.GetMaskedImage()
        image = veg.GetImage()
        normalized = np.zeros_like(image)
        finalImage = cv.normalize(image, normalized, 0, 255, cv.NORM_MINMAX)
        if arguments.mask:
            filledMask = mask.copy().astype(np.uint8)
            cv.floodFill(filledMask, None, (0, 0), 255)
            filledMaskInverted = cv.bitwise_not(filledMask)
            manipulated.toGreyscale()
            threshold, imageThresholded = cv.threshold(manipulated.greyscale, 0, 255, cv.THRESH_BINARY_INV)
            finalMask = cv.bitwise_not(filledMaskInverted)
            logger.logImage("processed", finalImage)
            # veg.ShowImage("Thresholded", imageThresholded)
            logger.logImage("inverted", filledMaskInverted)
            # veg.ShowImage("Filled", filledMask)
            # veg.ShowImage("Inverted", filledMaskInverted)
            # veg.ShowImage("Final", finalMask)
            logger.logImage("final", finalMask)
            # plt.imshow(veg.imageMask, cmap='gray', vmin=0, vmax=1)
            # plt.imshow(finalImage)

        # print("X={}".format(x))            #plt.show()
        # logger.logImage("mask", veg.imageMask)
        # ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(finalImage, imageNumberBasler, logger)

        if arguments.database:
            imageInDB = RawImage.find(manipulated.hash, persistenceConnection)

            # if imageInDB is not None:
            #     log.error(f"Image with hash {manipulated.hash} already exists in DB. Skipping")
            #     sequence += 1
            #     imageNumberBasler += 1
            #     return constants.ProcessResult.NOT_PROCESSED
            # else:
            #     log.debug(f"Image with hash {manipulated.hash} does not exist in DB, Continuing")

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
        performance.start()
        manipulated.toCIELAB()
        performance.stopAndRecord(constants.PERF_CIELAB)

        # use finalImage for indexed image
        manipulated.toDGCI(processed.source)

        # Find the plants in the image
        performance.start()
        strategy = constants.Strategy.CARTOON
        try:
            strategy = constants.Strategy[arguments.strategy]
        except ValueError:
            log.error(f"Unknown strategy: {arguments.strategy}")

        contours, hierarchy, blobs, largest = manipulated.findBlobs(arguments.minarea, strategy)
        log.debug("Found blobs: {}".format(len(blobs)))
        performance.stopAndRecord(constants.PERF_CONTOURS)

        if len(blobs) == 0:
            log.error(f"Unable to find blobs in image")
            logger.increment()
            sequence = sequence + 1
            imageNumberBasler = imageNumberBasler + 1
            return constants.ProcessResult.NOT_PROCESSED

        # The test should probably be if we did not find any blobs
        if largest == "unknown":
            logger.logImage("error", manipulated.image)
            return constants.ProcessResult.NOT_PROCESSED

        performance.start()
        manipulated.identifyOverlappingVegetation()
        performance.stopAndRecord(constants.PERF_OVERLAP)

        # Set the classifier blob set to be the set just identified
        classifier.blobs = blobs

        # G L C M
        performance.start()
        # This takes too long and is not needed right now
        # manipulated.fitSquares(SQUARE_SIZE)
        # manipulated.drawSquares(SQUARE_SIZE)
        performance.start()
        manipulated.extractImages()
        manipulated.computeGLCM()
        performance.stopAndRecord(constants.PERF_GLCM)

        performance.start()
        manipulated.computeShapeIndices()
        performance.stopAndRecord(constants.PERF_SHAPES_IDX)

        performance.start()
        manipulated.computeLengthWidthRatios()
        performance.stopAndRecord(constants.PERF_LW_RATIO)

        # H O G
        performance.start()
        manipulated.computeHOG()

        # L B P
        performance.start()
        manipulated.computeLBP()
        performance.stopAndRecord(constants.PERF_LBP)

        # New image analysis based on readings here:
        # http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth10.pdf

        performance.start()
        manipulated.computeCompactness()
        manipulated.computeElogation()
        manipulated.computeEccentricity()
        manipulated.computeRoundness()
        manipulated.computeConvexity()
        manipulated.computeSolidity()
        manipulated.computeMiscShapeMetrics()
        manipulated.computeBendingEnergy()
        manipulated.computeRadialDistances()
        performance.stopAndRecord(constants.PERF_SHAPES)

        # # GLCM attributes
        # theGLCM = GLCM(blobs, manipulated.greyscale)
        # theGLCM.computeAttributes()

        # End image analysis

        # Classify items by where they are in image
        # This only marks items that can't be fully seen (at edges) of image
        if arguments.edge:
            classifier.classifyByPosition(size=manipulated.image.shape)

        # Determine the distance from the object to the edge of the image given  the pixel size of the camera
        performance.start()
        manipulated.computeDistancesToImageEdge(camera.getMMPerPixel(), camera.getResolution())
        performance.stopAndRecord(constants.PERF_DISTANCE)

        classifiedBlobs = classifier.blobs

        performance.start()
        # manipulated.findAngles()
        manipulated.findCropLine()
        performance.stopAndRecord(constants.PERF_ANGLES)

        # Crop row processing
        manipulated.identifyCropRowCandidates()

        # Extract various features
        performance.start()

        # Compute the mean of the hue across the plant
        manipulated.extractImagesFrom(manipulated.hsi, 0, constants.NAME_HUE, np.nanmean)
        performance.stopAndRecord(constants.PERF_MEAN)
        manipulated.extractImagesFrom(manipulated.hsv, 1, constants.NAME_SATURATION, np.nanmean)

        # Discussion of YIQ can be found here
        # Sabzi, Sajad, Yousef Abbaspour-Gilandeh, and Juan Ignacio Arribas. 2020.
        # “An Automatic Visible-Range Video Weed Detection, Segmentation and Classification Prototype in Potato Field.”
        # Heliyon 6 (5): e03685.
        # The article refers to the I component as in-phase, but its orange-blue in the wikipedia description
        # of YIQ.  Not sure which is correct.

        # Compute the standard deviation of the I portion of the YIQ color space
        performance.start()
        manipulated.extractImagesFrom(manipulated.yiq, 1, constants.NAME_I_YIQ, np.nanstd)
        performance.stopAndRecord(constants.PERF_STDDEV)

        # Compute the mean of the blue difference in the ycc color space
        performance.start()
        manipulated.extractImagesFrom(manipulated.ycbcr, 1, constants.NAME_BLUE_DIFFERENCE, np.nanmean)
        performance.stopAndRecord(constants.PERF_MEAN)

        # Use either heuristics or logistic regression
        if arguments.logistic or arguments.knn or arguments.tree or arguments.forest or arguments.gradient or arguments.support or arguments.lda or arguments.perceptron or arguments.extra:
            log.debug(f"Classify by {mlApproach}")
            performance.start()
            classifier.classify()
            performance.stopAndRecord(constants.PERF_CLASSIFY)
            classifiedBlobs = classifier.blobs
        else:
            performance.start()
            classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=arguments.minratio)
            performance.stopAndRecord(constants.PERF_CLASSIFY)

        # Draw boxes around the images we found with decorations for attributes selected
        if not arguments.plain:
            manipulated.drawBoxes(manipulated.name, classifiedBlobs, featuresToShow, arguments.hull)

        # Eliminate vegetation we would damage
        classifier.classifyByDamage(classifiedBlobs)

        #
        # C R O P L I N E S
        #
        if arguments.cropline:
            manipulated.substituteRectanglesForVegetation()
            logger.logImage("rectangles", manipulated.croplineImage)
            # This is using the hough transform which we abandoned as a technique
            manipulated.detectLines()
            logger.logImage("cropline", manipulated.cropline_image)

            # TODO: Draw crop line as part of image decoration
            manipulated.drawCropline()
            # logger.logImage("crop-line", manipulated.croplineImage)

        if arguments.contours:
            manipulated.drawContours()

        # Everything in the image is classified, so decorate the image with distances
        manipulated.drawDistances()

        # Just a test of stitching. This needs some more thought
        # we can't stitch things where there is nothing in common between the two images
        # even if there is overlap. It may just be black background after the segmentation.
        # One possibility here is to put something in the known overlap area that can them be used
        # to align the images.
        # The alternative is to use the original images and use the soil as the element that is common between
        # the two.  The worry here is computational efficiency

        # if arguments.stitch:
        #     if previousImage is not None:
        #         manipulated.stitchTo(previousImage)
        #     else:
        #         previousImage = image

        # Write out the processed image
        # cv.imwrite("processed.jpg", manipulated.image)
        processedImage = logger.logImage(constants.FILENAME_PROCESSED + "-" + arguments.algorithm + "-" + mlApproach, manipulated.image)
        originalImage = logger.logImage(constants.FILENAME_ORIGINAL + "-" + arguments.algorithm, manipulated.original)
        # ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        #
        # D A T A B A S E
        #

        # If the image is not already in DB, put it there
        imageInDB  = RawImage.find(manipulated.hash, persistenceConnection)
        if imageInDB is None:
            url = "file://" + currentSessionName + "/" + rawImage
            lat = processed.latitude
            long = processed.longitude

            # If the altitude on the command line overrides the image
            if arguments.altitude > 0.0:
                agl = arguments.altitude
            else:
                agl = processed.altitude

            hash = manipulated.hash
            date_format = '%Y:%m:%d %H:%M:%S'
            acquired = datetime.strptime(processed.takenAt, date_format)
            processedURL = "file://" + currentSessionName + "/" + processedImage
            processedEntry = {arguments.algorithm + "-" + mlApproach: processedURL}
            segmentedURL = "file://" + currentSessionName + "/" + originalImage
            segmented = {arguments.algorithm: segmentedURL}

            # The image does not have a human-readable name yet
            imageName = constants.NAME_IMAGE + "-" + str(sequence)

            image = RawImage(imageName, url, lat, long, agl, acquired, arguments.crop, segmented, processedEntry, hash, None)
            image.save(persistenceConnection)
            # For blobs to point back to the parent
            parentId = ObjectId(image.id)
        else:
            log.debug(f"Image {manipulated.hash} is already in DB")
            processedURL = "file://" + currentSessionName + "/" + processedImage
            imageInDB.addProcessed(arguments.algorithm + "-" + mlApproach, processedURL)
            segmentedURL = "file://" + currentSessionName + "/" + originalImage
            imageInDB.addSegmented(arguments.algorithm, segmentedURL)
            imageInDB.save(persistenceConnection)
            parentId = ObjectId(imageInDB.id)

        # So at this point, the raw image, the segmented image, and the processed image should be in the DB
        # Construct the list of all attributes
        logger.reportIndex = False
        for blobName, blobAttributes in manipulated.blobs.items():
            log.debug(f"Inserting blob {blobName} of {originalImage} into DB")
            allFactors = Factors()

            allAttributes = {}
            columns = allFactors.getColumns([], [])
            for factor in columns:
                allAttributes[factor] = blobAttributes[factor]
            #log.debug(f"Attribute list: {allAttributes}")
            fileName = Path(originalImage)
            blobFilename = str(fileName.with_suffix('')) + "-" + blobName
            blobImage = logger.logImage(blobFilename, blobAttributes[constants.NAME_IMAGE])
            blobURL = "file://" + currentSessionName + "/" + blobImage
            # TODO: Not correct -- this should really be the location of the blob, not the image
            lat = processed.latitude
            long = processed.longitude
            # This could be obtained from looking at the parent's image, but just for convenience, put it here.
            # If the altitude on the command line overrides the image
            if arguments.altitude > 0.0:
                altitude = arguments.altitude
            else:
                altitude = processed.altitude
            technique = mlApproach
            hash = sha1(np.ascontiguousarray(blobAttributes[constants.NAME_IMAGE])).hexdigest()
            classifiedAs = int(blobAttributes[constants.NAME_TYPE])
            blobInDB = Blob(blobName, blobURL, lat, long, altitude, allAttributes, factors, technique, classifiedAs, parentId, hash, None)
            blobInDB.save(persistenceConnection)
        logger.reportIndex = True

        # Write out the crop images so we can use them later
        if arguments.xtract:
            manipulated.extractImages()
            for blobName, blobAttributes in manipulated.blobs.items():
                if blobAttributes[constants.NAME_TYPE] == constants.TYPE_DESIRED:
                    logger.logImage("crop", blobAttributes[constants.NAME_IMAGE])
                else:
                    logger.logImage("weed", blobAttributes[constants.NAME_IMAGE])


        reporting.addBlobs(sequence, blobs)
        sequence = sequence + 1

        if arguments.spray:
            performance.start()
            treatment = Treatment(manipulated.original, manipulated.binary)
            treatment.overlayTreatmentLanes()
            treatment.generatePlan(classifiedBlobs)
            # treatment.drawTreatmentLanes(classifiedBlobs)
            performance.stopAndRecord(constants.PERF_TREATMENT)
            logger.logImage("treatment", treatment.image)

        imageNumberBasler = imageNumberBasler + 1

    except IOError as e:
        print("There was a problem communicating with the camera")
        print(e)
        sys.exit(1)
    except EOFError:
        print("End of input")
        return constants.ProcessResult.EOF

    if arguments.histograms:
        reporting.showHistogram("Areas", 20, constants.NAME_AREA)
        reporting.showHistogram("Shape", 20, constants.NAME_SHAPE_INDEX)

    logger.increment()

    return constants.ProcessResult.OK


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

    # Write session data out as an INI file
    finished = os.path.join(logger.directory,
                            options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_FILENAME_FINISHED))
    log.debug("Writing session statistics to: {}".format(finished))
    sessionInfo = configparser.ConfigParser()
    sessionInfo[constants.PROPERTY_SECTION_GENERAL] = {'ACQUIRED': 'imageNumber'}
    try:
        with open(finished, 'w') as fp:
            sessionInfo.write(fp)
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
        log.error(
            "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

    systemRoom.sendMessage(systemMessage.formMessage())


def runDiagnostics(systemRoom: MUCCommunicator, camera: Camera):
    """
    Run diagnostics for this subsystem, collecting information about the camera connectivity.
    :param systemRoom: The room to send the results
    """
    systemMessage = SystemMessage()
    systemMessage.action = constants.Action.DIAG_REPORT.name
    systemMessage.diagnostics = camera.status.name
    systemMessage.gsdCamera = camera.gsd
    try:
        position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
        systemMessage.position = position
    except KeyError:
        log.error(
            "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

    # Include the status of system components in the report
    systemMessage.statusCamera = camera.status.name
    systemMessage.statusSystem = systemStatus.name
    systemMessage.statusOdometry = odometryStatus.name

    # TODO: Correct hard coded status
    systemMessage.statusSystem = constants.OperationalStatus.OK.name

    systemRoom.sendMessage(systemMessage.formMessage())


totalMovement = 0.0
keepAliveMessages = 0
movementSinceLastProcessing = 0.0
movementSinceLastProcessingForIntel = 0.0
distanceAtLastIntelProcessing = 0.0
distanceAtLastBaslerProcessing = 0.0
odometryMessageCount = 0

# This is the current sequence number of the odometry messages.
# Setting this to -1 means the first message will be processed
currentSequenceNumberForOdometry = -1


def messageIsCurrent(timestamp: int) -> bool:
    """
    Determine if a message is old or current
    :param timestamp: Timestamp the message was sent
    :return: True if message is current, False otherwise
    """
    timeDelta = (time.time() * 1000) - timestamp
    # log.debug("Time delta of message: {} ns".format(timeDelta))
    return timeDelta < constants.OLD_MESSAGE


#
# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageSystemCB(conn, msg: xmpp.protocol.Message):
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
            # log.debug("system message from {}".format(msg.getFrom()))
            systemMessage = SystemMessage(raw=msg.getBody())
            if messageIsCurrent(systemMessage.timestamp):
                # log.debug("Processing [{}]".format(msg.getBody()))
                if systemMessage.action == constants.Action.START.name:
                    processing = True
                    currentSessionName = systemMessage.name
                    currentOperation = systemMessage.operation
                    outputDirectory = arguments.output + "/" + currentSessionName
                    log.debug("Begin processing to: {}".format(outputDirectory))
                    logger = ImageLogger()
                    if not logger.connect(outputDirectory):
                        log.error("Unable to connect to logging. {} does not exist.".format(outputDirectory))
                if systemMessage.action == constants.Action.STOP.name:
                    log.debug("----- Stop weeding ------")
                    currentOperation = constants.Operation.QUIESCENT.name
                    postWeedingCleanup()
                if systemMessage.action == constants.Action.CURRENT.name:
                    sendCurrentOperation(roomSystem)
                if systemMessage.action == constants.Action.START_DIAG.name:
                    # log.debug("Request for diagnostics")
                    runDiagnostics(roomSystem, camera)
            else:
                log.info("Old message seen -- ignored")

    elif msg.getType() == "chat":
        log.info("private: " + str(msg.getFrom()) + ":" + str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))


def processOdometryMessage(message: str):
    """
    Process the odometry message
    :param message: A JSON string of the message
    """
    global totalMovement
    global keepAliveMessages
    global movementSinceLastProcessing
    global movementSinceLastProcessingForIntel
    global currentSequenceNumberForOdometry
    global distanceAtLastIntelProcessing
    global distanceAtLastBaslerProcessing
    global odometryMessageCount

    performance.start()
    # log.debug("RAW message: {}".format(message))
    odometryMessage = OdometryMessage(raw=message)

    # We are only concerned with distance messages here
    if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
        # This assumes we haven't missed any messages
        # totalMovement += odometryMessage.distance
        # movementSinceLastProcessing += odometryMessage.distance
        # movementSinceLastProcessingForIntel += odometryMessage.distance

        totalMovement = odometryMessage.totalDistance
        movementSinceLastProcessing = totalMovement - distanceAtLastBaslerProcessing
        movementSinceLastProcessingForIntel = totalMovement - distanceAtLastIntelProcessing

        if odometryMessage.sequence == -1:
            log.debug("Odometry sequence number restarted at -1. Not processing this message")
            currentSequenceNumberForOdometry = 0
            return
        # If the sequence number is the same as the last one, ignore
        elif odometryMessage.sequence == currentSequenceNumberForOdometry:
            # log.debug("Sequence number {} already processed.".format(currentSequenceNumberForOdometry))
            return
        # Missed a message -- update the current sequence and process
        elif odometryMessage.sequence != currentSequenceNumberForOdometry + 1:
            # Produce an error message if the movement is greater than the overlap of the GSD. Slightly exceeding the GSD is normal
            # If the movement is 410mm and the GSD is 400mm, that's OK, but if the movement was 420.5mm in the same
            # situation, we want to know
            gap = odometryMessage.sequence - (currentSequenceNumberForOdometry + 1)
            overlapFactor = float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))

            # When the system first comes up, the total distance moved makes it look like a message was missed

            if odometryMessageCount > 0:
                if movementSinceLastProcessing > (gsdBasler * (1 + overlapFactor)):
                    log.fatal("Missed movement exceeds GSD + overlap of Basler: {} Message gap {}".format(
                        movementSinceLastProcessing, gap))
                if movementSinceLastProcessingForIntel > (gsdIntel * (1 + overlapFactor)):
                    log.fatal("Missed movement exceeds GSD + overlap of Intel: {} Message gap {}".format(
                        movementSinceLastProcessingForIntel, gap))

            currentSequenceNumberForOdometry = odometryMessage.sequence
        # Normal case
        else:
            currentSequenceNumberForOdometry = odometryMessage.sequence
            # log.debug("Processing Odometry message: {}".format(message))

        odometryMessageCount += 1
        # The time of the observation
        timeRead = odometryMessage.timestamp
        # Determine how old the observation is
        # The version of python on the jetson does not support time_ns, so this a bit of a workaround until I
        # get that sorted out.  Just convert the reading to milliseconds for now
        # timeDelta = (time.time() * 1000) - (timeRead / 1000000)
        timeDelta = (time.time() * 1000) - timeRead

        # If the movement is equal to the size of the image, take a picture
        # We need to allow for some overlap so the images can be stitched together.
        # So reduce this by the overlap factor

        #
        # log.debug("Total movement: {} at time: {}. Movement: {} GSD [Basler: {} Intel: {}] Time now is {} delta from now {} ms Processing: {}".
        #           format(totalMovement, timeRead, movementSinceLastProcessing, gsdBasler, gsdIntel, time.time() * 1000, timeDelta, processing))

        # The Basler camera
        if movementSinceLastProcessing > gsdBasler:
            distanceAtLastBaslerProcessing = totalMovement
            log.debug("Acquiring image from Basler.  Movement since last processing {} GSD {}".format(
                movementSinceLastProcessing, gsdBasler))
            movementSinceLastProcessing = 0
            #
            # Record the context under which this photo was taken
            contextForImage = Context()
            contextForImage.latitude = odometryMessage.latitude
            contextForImage.longitude = odometryMessage.longitude
            # Convert to kilometers
            contextForImage.speed = odometryMessage.speed / 1e+6
            contextForImage.model = "2500"
            processor(contextForImage, constants.Capture.RGB, camera.position)

        # The intel RGB camera
        if movementSinceLastProcessingForIntel > gsdIntel:
            log.debug("Acquiring image from Intel Camera.  Movement since last processing {} GSD {}".format(
                movementSinceLastProcessingForIntel, gsdIntel))
            movementSinceLastProcessingForIntel = 0
            distanceAtLastIntelProcessing = totalMovement

            # Record the context under which this photo was taken
            contextForImage = Context()
            contextForImage.latitude = odometryMessage.latitude
            contextForImage.longitude = odometryMessage.longitude
            # Convert to kilometers
            contextForImage.speed = odometryMessage.speed / 1e+6
            # contextForPhoto.model
            processor(contextForImage, constants.Capture.DEPTH_RGB, camera.position)
    else:
        pass
        # log.debug("Message type is not distance. Ignored")

    # Too noisy -- less than 0.25 ms on the jetsons.  Good enough
    # log.debug("Processed odometry message: {} ms".format(performance.stop()))


#
# The callback for messages received in the odometry room.
# When the total distance is the width of the image, grab an image and process it.
#

# This is the MUC style of interaction

def messageOdometryCB(conn, msg: xmpp.protocol.Message):
    global totalMovement
    global keepAliveMessages
    global movementSinceLastProcessing
    global movementSinceLastProcessingForIntel

    # Record how long this take
    performance.start()

    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
        body = msg.getBody()
        # Check if this is a real message and not just an empty keep-alive message
        if body is not None:
            # log.debug("Distance message from {}".format(msg.getFrom()))
            odometryMessage = OdometryMessage(raw=body)

            if messageIsCurrent(odometryMessage.timestamp):
                # log.debug("Message: {}".format(odometryMessage.data))
                #
                # We are only concerned with distance messages here
                if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
                    pass
                else:
                    pass
                    # log.debug("Message type is not distance. Ignored")
            else:
                log.info("Old message seen -- ignored")
        else:
            # There's not much to do here for keepalive messages
            keepAliveMessages += 1
            # print("weeds: keepalive message from chatroom")
    elif msg.getType() == "chat":
        log.error("Private Message from {}: {} ".format(str(msg.getFrom()), str(msg.getBody())))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

    log.debug("Processed odometry message: {} ms".format(performance.stop()))


# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageTreatmentCB(conn, msg: xmpp.protocol.Message):
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
        body = msg.getBody()
        # Check if this is a real message and not just an empty keep-alive message
        # if body is not None:
        #     log.debug("treatment message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
    elif msg.getType() == "chat":
        print("private: " + str(msg.getFrom()) + ":" + str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))


def connectMQ(communicator: ClientMQCommunicator) -> bool:
    global odometryStatus
    serverResponding = False
    retriesRequired = False
    while not serverResponding:
        (serverResponding, response) = communicator.sendMessageAndWaitForResponse(constants.COMMAND_PING, 10000)
        if not serverResponding:
            log.error("Odometry server did not respond within 10 seconds. Will retry.")
            odometryStatus = constants.OperationalStatus.FAIL
            communicator.disconnect()
            retriesRequired = True
        else:
            if retriesRequired:
                log.info("Odometry server responded successfully after retry")
                odometryStatus = constants.OperationalStatus.OK
    return serverResponding


#
# Process the incoming MQ stream
#
def processMQ(communicator: ClientMQCommunicator):
    """
    Process the incoming MQ stream.
    :param communicator:
    """
    global odometryStatus
    global processingOdometry

    # Wait for tge initial connection before proceeding
    serverResponding = connectMQ(communicator)

    processingOdometry = True
    # Continue processing messages until shutdown
    while processingOdometry:
        (serverResponding, response) = communicator.sendMessageAndWaitForResponse(constants.COMMAND_ODOMETERY, 1000)
        # If the server responds, process the message, otherwise reconnect.
        if serverResponding:
            odometryStatus = constants.OperationalStatus.OK
            processOdometryMessage(response)
        else:
            odometryStatus = constants.OperationalStatus.FAIL
            log.error("The odometry server failed to respond during operation. Reconnecting")
            communicator.disconnect()
            connectMQ(communicator)


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
def takeRGBImages(camera: CameraBasler):
    cameraConnected = False

    if camera.ip == constants.IP_NONE:
        log.info("IP Address of post-emitter camera not found.  Using dummy capture loop.")
        while True:
            time.sleep(60)
            log.info("Dummy post-emitter capture loop")
        return

    # Connect to the camera and take an image
    log.debug("Connecting to camera")

    capturing = True
    # Temporary -- this effectively disables basler
    # while capturing:
    #    log.debug("Dummy basler capture loop")
    #    time.sleep(10)
    #
    # return

    cameraConnected = camera.connect()
    # camera.start()

    if cameraConnected:
        if isinstance(camera, CameraBasler):
            if camera.initializeCapture():
                try:
                    log.debug("Beginning Basler RGB Capture")
                    # The version using callbacks
                    camera.startCapturing()
                    # The version using a direct grabber
                    # camera.start()
                except IOError as io:
                    camera.log.error(io)
                rc = 0
            else:
                log.fatal("Unable to initialize camera")
                rc = -1
        else:
            log.debug("Not a physical camera")
            camera.startCapturing()
    else:
        log.error("Unable to connect to camera")


#
# Enrich the images on disk with the EXIF data
# This is to allow the processing engine to not be concerned with I/O
#
def enrichImages():
    """
    Enrich the images with various EXIF tags.
    """
    while True:
        # Get the next image from the queue, blocking if it is empty
        rawImage = rawImages.dequeue()
        outputDirectory = arguments.output + "/" + currentSessionName
        enricher = Enrich(outputDirectory)
        log.debug("Enriching image: {}".format(rawImage.filename))
        captureType = rawImage.captureType
        # enricher.writeImageAndEnrich(rawImage)

        # Intel RGB
        if rawImage.captureType == constants.Capture.DEPTH_RGB:
            log.debug("Saving Intel RGB data as numpy: {}".format(rawImage.filename + constants.EXTENSION_NPY))
            performance.start()
            np.save(rawImage.filename + constants.EXTENSION_NPY, rawImage.image)
            performance.stopAndRecord(constants.PERF_SAVE_INTEL_RGB_NPY)
            log.debug("Saving Intel RGB data as JPG: {}".format(rawImage.filename + constants.EXTENSION_IMAGE))

            # Save the image as JPG
            performance.start()
            image = Image.fromarray(rawImage.image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(rawImage.filename + constants.EXTENSION_IMAGE)
            performance.stopAndRecord(constants.PERF_SAVE_INTEL_RGB)
            enricher.addEXIFToImageAndWriteToDisk(rawImage)
            # This uses the opencv libs -- not sure why this doesn't work
            # TODO: Convert the image save routine back to use opencv
            # ImageManipulation.write(rgb.image, imagePath)

            # Send out a message to the treatment channel that an image has been taken and is available.
            performance.start()
            message = TreatmentMessage()
            message.plan = constants.Treatment.RAW_IMAGE
            message.source = constants.Capture.DEPTH_RGB
            message.name = "original"
            message.url = "http://" + platform.node() + "/" + rawImage.urlFilename + constants.EXTENSION_IMAGE
            message.timestamp = time.time() * 1000

            try:
                position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
                message.position = position
            except KeyError:
                log.error(
                    "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL,
                                                          constants.PROPERTY_POSITION))

            messageText = message.formMessage()
            log.debug("Sending: {}".format(messageText))
            messageID = roomTreatment.sendMessage(messageText)
            log.debug("Sent message with ID: {}".format(messageID))
            performance.stopAndRecord(constants.PERF_TREATMENT_MSG)

        elif rawImage.captureType == constants.Capture.DEPTH_DEPTH:
            log.debug("Saving Intel depth data as numpy: {}".format(rawImage.filename))
            performance.start()
            np.save(rawImage.filename, rawImage.image)
            performance.stopAndRecord(constants.PERF_SAVE_INTEL_DEPTH)

            # Send message to note the depth data
            message = TreatmentMessage()
            message.plan = constants.Treatment.RAW_IMAGE
            message.source = constants.Capture.DEPTH_DEPTH
            message.name = "original"
            message.url = "http://" + platform.node() + "/" + rawImage.urlFilename + constants.EXTENSION_NPY
            message.timestamp = time.time() * 1000

            try:
                position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
                message.position = position
            except KeyError:
                log.error(
                    "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL,
                                                          constants.PROPERTY_POSITION))

            messageText = message.formMessage()
            log.debug("Sending: {}".format(messageText))
            messageID = roomTreatment.sendMessage(messageText)
            log.debug("Sent message with ID: {}".format(messageID))
            performance.stopAndRecord(constants.PERF_TREATMENT_MSG)

        elif rawImage.captureType == constants.Capture.RGB:
            log.debug("Saving Basler RGB: {}".format(rawImage.filename + constants.EXTENSION_IMAGE))
            # ImageManipulation.show("Source",image)
            performance.start()
            imageName = rawImage.filename + constants.EXTENSION_IMAGE

            image = Image.fromarray(rawImage.image)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(rawImage.filename + constants.EXTENSION_IMAGE)

            # Use the basler routines to save the image
            # As PNG
            # rawImage.image.Save(pylon.ImageFileFormat_Png, rawImage.filename + constants.EXTENSION_PNG)
            # rawImage.image.Release()

            # As JPG. Bit of a pain here. JPG can't be saved on linux. Not sure why
            # ipo = pylon.ImagePersistenceOptions()
            # quality = 100
            # ipo.SetQuality(quality)
            # #
            # rawImage.image.Save(pylon.ImageFileFormat_Jpeg, rawImage.filename, ipo)

            # Use the opencv routines to save the image
            # veg.SetImage(rawImage.image)

            # manipulated = ImageManipulation(rawImage.image, imageNumberBasler, logger)
            # try:
            #     cv.imwrite(imageName, rawImage.image)
            # except Exception:
            #     log.error("Unable to save Basler RGB")
            # End of image save using opencv

            # fileName = logger.logImage(imageName, manipulated.image)

            # enricher.addEXIFToImageAndWriteToDisk(rawImage)
            performance.stopAndRecord(constants.PERF_SAVE_BASLER_RGB)

            # Send out a message to the treatment channel that an image has been taken and is available.
            message = TreatmentMessage()
            message.plan = constants.Treatment.RAW_IMAGE
            message.source = constants.Capture.RGB
            message.name = "original"
            message.url = "http://" + platform.node() + "/" + rawImage.urlFilename + constants.EXTENSION_IMAGE
            message.timestamp = time.time() * 1000

            try:
                position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
                message.position = position
            except KeyError:
                log.error(
                    "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL,
                                                          constants.PROPERTY_POSITION))

            messageText = message.formMessage()
            log.debug("Sending: {}".format(messageText))
            performance.start()
            messageID = roomTreatment.sendMessage(messageText)
            log.debug("Sent message with ID: {}".format(messageID))
            performance.stopAndRecord(constants.PERF_TREATMENT_MSG)

        else:
            log.error("Unknown capture type: {}".format(rawImage.captureType))


# Start up various subsystems
#

if not arguments.standalone:
    # Diagnostics will appear in the same directory structure as the output files
    systemDiagnostics = Diagnostics("../output", options.option(constants.PROPERTY_SECTION_GENERAL,
                                                                constants.PROPERTY_POSITION) + constants.EXTENSION_HTML)
    systemDiagnostics.writeHTML()

# Set the status of all components to failed initially
systemStatus = constants.OperationalStatus.FAIL
intel435Status = constants.OperationalStatus.FAIL
odometryStatus = constants.OperationalStatus.FAIL

#currentSessionName = ""
currentOperation = constants.Operation.QUIESCENT.name

# D A T A B A S E
# If any of these are specified, they all must be
if arguments.database:
    dbHost = options.option(constants.PROPERTY_SECTION_DATABASE, constants.PROPERTY_HOST) if arguments.host is None else arguments.host
    dbPort = int(options.option(constants.PROPERTY_SECTION_DATABASE, constants.PROPERTY_PORT)) if arguments.port is None else arguments.port
    dbName = options.option(constants.PROPERTY_SECTION_DATABASE, constants.PROPERTY_DB) if arguments.dbname is None else arguments.dbname
    if dbHost is None or dbPort is None or dbName is None:
        log.fatal("Specify host, port, and database to use a database on the command line or INI file")
        sys.exit(-1)

    persistenceConnection = Mongo()
    persistenceConnection.connect(dbHost, dbPort, "", "", dbName)
    if not persistenceConnection.connected:
        log.fatal("Unable to connect to database")
        sys.exit(-1)
else:
    persistenceConnection = Disk()


# Set up image processing features
try:
    factors = options.option(constants.PROPERTY_SECTION_IMAGE_PROCESSING, constants.PROPERTY_FACTORS).split(',')
    factors = [factor.strip() for factor in factors]
    log.debug(f"Extract and use factors: {factors}")
except KeyError:
    log.fatal(f"Unable to find factors {constants.PROPERTY_SECTION_IMAGE_PROCESSING}/{constants.PROPERTY_FACTORS}")
    sys.exit(-1)

camera = startupCamera(options)

# For analysing weeds, we collect depth images
gsdIntel = float('inf')

# What sort of capture is to be performed with the intel camera?

# By default, this is only depth data
# intelCaptureType = constants.IntelCapture.DEPTH
# captureType = options.option(constants.PROPERTY_SECTION_INTEL, constants.PROPERTY_CAPTURE)
# if captureType == constants.IntelCapture.DEPTH.name:
#     intelCaptureType = constants.IntelCapture.DEPTH
#     log.info("Intel capture limited to depth data only")
# elif captureType == constants.IntelCapture.RGBDEPTH.name:
#     intelCaptureType = constants.IntelCapture.RGBDEPTH
#     log.info("Intel capturing both RGB and depth data")

# This is the GSD of the image that takes into account overlap
gsdBasler = (1 - float(
    options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * camera.gsd
camera.gsdAdjusted = gsdBasler
log.debug("GSD Basler: {}/{}".format(gsdBasler, camera.gsd))

(roomOdometry, roomSystem, roomTreatment) = startupCommunications(options, messageOdometryCB, messageSystemCB,
                                                                  messageTreatmentCB)
log.debug("Communications started")

odometryMQ = startupMQCommunications(options, processOdometryMessage)

performance = startupPerformance()
log.debug("Performance started")

# Start the worker threads, putting them in a list
threads = list()

#
# There are two modes of operation here: standalone or part of a system.
#
# If this is part of a system, startup all the threads required, and have the odometry messages drive things
#
if not arguments.standalone:

    # Images before and after processing
    rawImages = Images()
    processedImages = Images()

    log.debug("Start camera image acquisition")
    acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeRGBImages, args=(camera,))
    acquire.daemon = True
    threads.append(acquire)
    acquire.start()

    # log.debug("Start post-emitter camera image acquisition")
    # acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE_POST, target=takeRGBImages, args=(None,))
    # acquire.daemon = True
    # threads.append(acquire)
    # acquire.start()

    log.debug("Starting odometry MQ receiver")
    odometryProcessor = threading.Thread(name=constants.THREAD_NAME_REQ_RSP, target=processMQ, args=(odometryMQ,))
    odometryProcessor.daemon = True
    threads.append(odometryProcessor)
    odometryProcessor.start()

    log.debug("Starting system receiver")
    # sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=processMessages, args=(roomSystem,))
    sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=roomSystem.processMessages, args=())
    sys.daemon = True
    threads.append(sys)
    sys.start()

    # Not needed for post-emitter assessment
    if arguments.weeds:
        log.debug("Starting treatment thread")
        # treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=processMessages, args=(roomTreatment,))
        treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=roomTreatment.processMessages, args=())
        treat.daemon = True
        threads.append(treat)
        treat.start()

    log.debug("Starting enrichment thread")
    enrich = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=enrichImages, args=())
    enrich.daemon = True
    threads.append(enrich)
    enrich.start()

    # Wait for the workers to finish
    threadsAreAlive = True
    while threadsAreAlive:
        time.sleep(5)
        for thread in threads:
            if not thread.is_alive():
                log.error("Thread {} exited. This is not normal.".format(thread.name))
                threadsAreAlive = False

else:  # if not arguments.standalone

    # Connect to the camera and process until an error is hit
    camera.connect()
    contextForImage = Context()
    processing = True
    while processing:
        result = processor(contextForImage)
        processing = (result == constants.ProcessResult.OK or result == constants.ProcessResult.NOT_PROCESSED)

performance.cleanup()

# Not quite right here to get the list of all blobs from the reporting module
# classifier.train(reporting.blobs)

result, reason = reporting.writeSummary()

# if not result:
#     print(reason)
#     sys.exit(1)
# else:
#     sys.exit(0)
