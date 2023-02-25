#
# W E E D S
#

import argparse
import glob
import platform
import sys
import threading
from typing import Callable

import configparser

try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    supportsPlotting = True
except ImportError:
    print("Unable to import plotting libraries.")
    supportsPlotting = False

import scipy.ndimage

import logging.config
import shutil

import xmpp
# from xmpp import protocol
from datetime import datetime

# This does not work
# from CameraFile import CameraFile, CameraBasler

from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier, LogisticRegressionClassifier, KNNClassifier, DecisionTree, RandomForest, GradientBoosting, SuppportVectorMachineClassifier
from OptionsFile import OptionsFile
from Reporting import Reporting
from Treatment import Treatment
from MUCCommunicator import MUCCommunicator
from MQCommunicator import ClientMQCommunicator
from Messages import OdometryMessage, SystemMessage, TreatmentMessage
from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure
from CameraDepth import CameraDepth
from RealSense import RealSense
from ProcessedImage import ProcessedImage, Images
from Enrich import Enrich
from Context import Context
from Diagnostics import Diagnostics
from Camera import Camera
from CameraFile import CameraFile
from CameraBasler import CameraBasler

#from Selection import Selection

#
# C A M E R A S
#
# TODO: Move to Camera.py file
# This is very sloppy work, and has completely defeated me, so I give up
# This works just fine in another file, but fails whenever it is imported,
# so I'm giving up and copying it here

import logging
import logging.config
import time
from collections import deque

import numpy as np
import os
import cv2 as cv
from abc import ABC, abstractmethod

import pypylon.pylon
from pypylon import _genicam

from PIL import Image

import constants
from Performance import Performance


# ##################
#
# B A S L E R  E V E N T  H A N D L E R S
#

# Handle various basler camera events

# class ConfigurationEventPrinter(pypylon.pylon.ConfigurationEventHandler):
#     def OnAttach(self, camera):
#         print("OnAttach event")
#
#     def OnAttached(self, camera):
#         print("OnAttached event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnOpen(self, camera):
#         print("OnOpen event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnOpened(self, camera):
#         print("OnOpened event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnGrabStart(self, camera):
#         print("OnGrabStart event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnGrabStarted(self, camera):
#         print("OnGrabStarted event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnGrabStop(self, camera):
#         print("OnGrabStop event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnGrabStopped(self, camera):
#         print("OnGrabStopped event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnClose(self, camera):
#         print("OnClose event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnClosed(self, camera):
#         print("OnClosed event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnDestroy(self, camera):
#         print("OnDestroy event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnDestroyed(self, camera):
#         print("OnDestroyed event")
#
#     def OnDetach(self, camera):
#         print("OnDetach event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnDetached(self, camera):
#         print("OnDetached event for device ", camera.GetDeviceInfo().GetModelName())
#
#     def OnGrabError(self, camera, errorMessage):
#         print("OnGrabError event for device ", camera.GetDeviceInfo().GetModelName())
#         print("Error Message: ", errorMessage)
#
#     def OnCameraDeviceRemoved(self, camera):
#         print("OnCameraDeviceRemoved event for device ", camera.GetDeviceInfo().GetModelName())
#
# # Handle image grab notifications
#
# class ImageEvents(pypylon.pylon.ImageEventHandler):
#     def OnImagesSkipped(self, camera, countOfSkippedImages):
#         print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
#         print(countOfSkippedImages, " images have been skipped.")
#         print()
#
#     def OnImageGrabbed(self, camera, grabResult):
#         """
#         Called when an image has been grabbed by the camera
#         :param camera:
#         :param grabResult:
#         """
#         #log.debug("OnImageGrabbed event for device: {}".format(camera.GetDeviceInfo().GetModelName()))
#
#         # Image grabbed successfully?
#         if grabResult.GrabSucceeded():
#             # Convert the image grabbed to something we like
#             image = _CameraBasler.convert(grabResult)
#             img = image.GetArray()
#             # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
#             # We will mark the images based on when we got them -- ideally, this should be:
#             # timestamped = ProcessedImage(img, grabResult.TimeStamp)
#             timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
#
#             cameraNumber = camera.GetCameraContext()
#             camera = _Camera.cameras[cameraNumber]
#             #log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
#             camera._images.append(timestamped)
#
#             # print("SizeX: ", grabResult.GetWidth())
#             # print("SizeY: ", grabResult.GetHeight())
#             # img = grabResult.GetArray()
#             # print("Gray values of first row: ", img[0])
#             # print()
#         else:
#             log.error("Image Grab error code: {} {}".format(grabResult.GetErrorCode(), grabResult.GetErrorDescription()))
#
# # Example of an image event handler.
# class SampleImageEventHandler(pypylon.pylon.ImageEventHandler):
#     def OnImageGrabbed(self, _camera, grabResult):
#         print("CSampleImageEventHandler::OnImageGrabbed called.")
#         print()
#         print()
#
# # I gave up on getting this to work, and just copied it here.
# # This is not the correct thing to do, but I renamed the class so it would not
# # conflict with the base class of the depth camera.  This is a mess.
#
# class _Camera(ABC):
#     cameras = list()
#     cameraCount = 0
#
#     def __init__(self, **kwargs):
#
#         # Register the camera on the global list so we can keep track of them
#         # Even though there will probably be only one
#         self.cameraID = _Camera.cameraCount
#         _Camera.cameraCount += 1
#         _Camera.cameras.append(self)
#
#         self._status = constants.OperationalStatus.UNKNOWN
#
#         self._gsd = 0
#         return
#
#     @property
#     def status(self) -> constants.OperationalStatus:
#         return self._status
#
#     @abstractmethod
#     def connect(self) -> bool:
#         raise NotImplementedError()
#         return True
#
#     @abstractmethod
#     def initialize(self):
#         return
#
#     @abstractmethod
#     def start(self):
#         return
#
#     @abstractmethod
#     def disconnect(self):
#         raise NotImplementedError()
#         return True
#
#     @abstractmethod
#     def diagnostics(self):
#         self._connected = False
#         return 0
#
#     @abstractmethod
#     def capture(self) -> ProcessedImage:
#         self._connected = False
#         return
#
#     @abstractmethod
#     def getResolution(self) -> ():
#         self._connected = False
#         return (0,0)
#
#     @abstractmethod
#     def getMMPerPixel(self) -> float:
#         return
#
#     @property
#     def gsd(self) -> int:
#         """
#         The ground sampling distance as specified in the options file. As we can't determine how high off the ground
#         the camera is, this is a pre-computed value
#         :return: width of the ground capture.
#         """
#         return self._gsd
#
#     @gsd.setter
#     def gsd(self, distance: int):
#         self._gsd = distance
#
# class _CameraFile(_Camera):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self._connected = False
#
#         self.log = logging.getLogger(__name__)
#         if constants.KEYWORD_DIRECTORY in kwargs:
#             self.directory = kwargs[constants.KEYWORD_DIRECTORY]
#         else:
#             self.log.fatal("The image directory name must be specified with the keyword {}".format(constants.KEYWORD_DIRECTORY))
#         if constants.KEYWORD_GSD in kwargs:
#             self._gsd = kwargs[constants.KEYWORD_GSD]
#         else:
#             self.log.warning("The GSD keyword is not specified for the image set with {}. Using default.".format(constants.KEYWORD_GSD))
#             self._gsd = 0.5
#
#         self._currentImage = 0
#         self._image = None
#         self._capturing = False
#         self._metadata = None
#         self._mmPerPixel = 0.0
#         self._flist = []
#         return
#
#     def connect(self) -> bool:
#         """
#         Connects to a directory and finds all images there. This method will not traverse subdirectories
#         :return:
#         """
#         self._connected = os.path.isdir(self.directory)
#         # Find all the files in the directory.
#         if self._connected:
#             self._flist = glob.glob(self.directory + '/*' + constants.EXTENSION_IMAGE)
#             #self._flist = [p for p in pathlib.Path(self.directory).iterdir() if p.is_file()]
#         else:
#             self.log.error("Unable to connect to directory: {}".format(self.directory))
#
#         metadataFile = glob.glob(self.directory + '/*' + constants.EXTENSION_META)
#
#         if len(metadataFile) == 1:
#             # Load the metadata for the imageset
#             self._metadata = OptionsFile(metadataFile[0])
#             if self._metadata.load():
#                 try:
#                     self._mmPerPixel = float(self._metadata.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_PIXELS_PER_MM))
#                     self.log.debug("Image set mm per pixel: {}".format(self._mmPerPixel))
#                 except KeyError as key:
#                     self.log.error("Could not find pixel to mm mapping in metadata")
#             else:
#                 self.log.error("Unable to load metadata file {}".format(metadataFile[0]))
#         else:
#             self.log.warning("Expected one metadata file. Found {}".format(len(metadataFile)))
#
#         return self._connected
#
#     def disconnect(self):
#         self._connected = False
#         return True
#
#     def diagnostics(self):
#         return True, "Camera diagnostics passed"
#
#     def initialize(self):
#         return
#
#     def start(self):
#         return
#
#     def capture(self) -> ProcessedImage:
#         """
#         Each time capture() is called, the next image in the directory is returned
#         :return:
#         The image as a numpy array.  Raises EOFError when no more images exist
#         """
#         if self._currentImage < len(self._flist):
#             imageName = str(self._flist[self._currentImage])
#             self._image = cv.imread(imageName, cv.IMREAD_COLOR)
#             self._currentImage = self._currentImage + 1
#             processed = ProcessedImage(constants.Capture.RGB, self._image, 0)
#             return processed
#         # Raise an EOFError  when we get through the sequence of images
#         else:
#             raise EOFError
#
#     def startCapturing(self):
#         """
#         Start capturing loop for files on disk. This is a no-op loop
#         """
#         self.log.debug("Dummy capture loop started")
#         self._capturing = True
#         while self._capturing:
#             time.sleep(10)
#
#     def getResolution(self) -> ():
#         # The camera resolution is the shape of the current image
#         self.log.debug("Getting resolution of current image")
#         return self._image.shape
#
#     def getMMPerPixel(self) -> float:
#         #
#         # TODO: The mm per pixel is something that should be read from the metadata for the image set
#         return self._mmPerPixel
#
# class _CameraPhysical(_Camera):
#     def __init__(self, **kwargs):
#      self._connected = False
#      self._currentImage = 0
#      self._cam = cv.VideoCapture(0)
#      super().__init__(**kwargs)
#      return
#
#     def connect(self):
#         """
#         Connects to the camera and sets it to highest resolution for capture.
#         :return:
#         True if connection was successful
#         """
#         # Read calibration information here
#         HIGH_VALUE = 10000
#         WIDTH = HIGH_VALUE
#         HEIGHT = HIGH_VALUE
#
#         # A bit a hack to set the camera to the highest resolution
#         self._cam.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
#         self._cam.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
#
#         return True
#
#     def disconnect(self):
#         self._cam.release()
#
#     def initialize(self):
#         return
#
#     def start(self):
#         return
#
#     def diagnostics(self) -> (bool, str):
#         """
#         Execute diagnostics on the camera.
#         :return:
#         Boolean result of the diagnostics and a string of the details
#         """
#         return True, "Camera diagnostics not provided"
#
#     def capture(self) -> ProcessedImage:
#         """
#         Capture a single image from the camera.
#         Requires calling the connect() method before this call.
#         :return:
#         The image as a numpy array
#         """
#         ret, frame = self._cam.read()
#         if not ret:
#             raise IOError("There was an error encountered communicating with the camera")
#         # cv.imwrite("camera.jpg", frame)
#         processed = ProcessedImage(constants.Capture.RGB, frame, 0)
#         return processed
#
#     def getResolution(self) -> ():
#         w = self._cam.get(cv.CAP_PROP_FRAME_WIDTH)
#         h = self._cam.get(cv.CAP_PROP_FRAME_HEIGHT)
#         return (w, h)
#
#     # This should be part of the calibration procedure
#     def getMMPerPixel(self) -> float:
#         return 0.0
#
# #
# # The Basler camera is accessed through the pylon API
# # Perhaps this can be through openCV, but this will do for now
# #
#
#
# from pypylon import pylon
# from pypylon import genicam
#
# class _CameraBasler(_Camera):
#     # Initialize the converter for images
#     # The images stream of in YUV color space.  An optimization here might be to make
#     # both formats available, as YUV is something we will use later
#
#     _converter = pylon.ImageFormatConverter()
#
#     # converting to opencv bgr format
#     _converter.OutputPixelFormat = pylon.PixelType_BGR8packed
#     _converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
#
#     def __init__(self, **kwargs):
#         """
#         Tha basler camera object.
#         :param kwargs: ip=<ip-address of camera>
#         """
#         self._connected = False
#         self._currentImage = 0
#         self._camera = None
#         self.log = logging.getLogger(__name__)
#         self._strategy = constants.STRATEGY_ASYNC
#         self._capturing = False
#         self._images = deque(maxlen=constants.IMAGE_QUEUE_LEN)
#         self._camera = pylon.InstantCamera()
#
#         if constants.KEYWORD_GSD in kwargs:
#             self._gsd = kwargs[constants.KEYWORD_GSD]
#         else:
#             self.log.info("The GSD keyword is not specified with {}. Calculated instead.".format( constants.KEYWORD_GSD))
#             # This is just a placeholder
#             self._gsd = 0.5
#
#         # Assume a GigE camera for now
#         if constants.KEYWORD_IP in kwargs:
#             self._ip = kwargs[constants.KEYWORD_IP]
#         else:
#             self.log.fatal("The IP address of the camera must be specified with the keyword {}".format(constants.KEYWORD_IP))
#
#         super().__init__(**kwargs)
#
#     @classmethod
#     def convert(cls, grabResult):
#         image = _CameraBasler._converter.Convert(grabResult)
#         return image
#
#
#     def connect(self) -> bool:
#         """
#         Connects to the camera with the specified IP address.
#         """
#         tl_factory = pylon.TlFactory.GetInstance()
#
#         self._connected = False
#
#         for dev_info in tl_factory.EnumerateDevices():
#             self.log.debug("Looking for {}. Current device is {}".format(self._ip, dev_info.GetIpAddress()))
#             if dev_info.GetIpAddress() == self._ip:
#                 try:
#                     self._camera = pylon.InstantCamera()
#                     self._camera.Attach(tl_factory.CreateDevice(dev_info))
#                 except Exception as e:
#                     log.fatal("Error encountered in attaching camera")
#                     log.fatal("{}".format(e))
#                     self._status = constants.OperationalStatus.FAIL
#                 #self._camera.MaxNumBuffer = 100
#                 try:
#                     self._camera.Open()
#                     self.log.info("Using device {} at {}".format(self._camera.GetDeviceInfo().GetModelName(), dev_info.GetIpAddress()))
#                     self._connected = True
#                     self._status = constants.OperationalStatus.OK
#                 except Exception as e:
#                     self.log.fatal("Error encountered in opening camera")
#                     log.fatal("{}".format(e))
#                     self._status = constants.OperationalStatus.FAIL
#                 # This shows how to get the list of what is available as attributes.  Not particularly useful for what
#                 # we need here
#                 # info = pylon.DeviceInfo()
#                 # info = self._camera.GetDeviceInfo()
#                 # tlc = pylon.GigETransportLayer()
#                 # tlc = self._camera.GetTLNodeMap()
#                 #
#                 # properties = info.GetPropertyNames()
#
#                 #self.log.debug("Current counter {}".format())
#                 break
#
#         if not self._connected:
#             self.log.error("Failed to connect to camera")
#             #raise EnvironmentError("No GigE device found")
#
#         return self._connected
#
#
#     def initializeCapture(self):
#
#         initialized = False
#         try:
#             # Register the standard configuration event handler for enabling software triggering.
#             # The software trigger configuration handler replaces the default configuration
#             # as all currently registered configuration handlers are removed by setting the registration mode to RegistrationMode_ReplaceAll.
#             self._camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(),
#                                                pylon.RegistrationMode_ReplaceAll,
#                                                pylon.Cleanup_Delete)
#
#             # For demonstration purposes only, add a sample configuration event handler to print out information
#             # about camera use.t
#             self._camera.RegisterConfiguration(ConfigurationEventPrinter(),
#                                                pylon.RegistrationMode_Append,
#                                                pylon.Cleanup_Delete)
#
#             # The image event printer serves as sample image processing.
#             # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
#             # results must be created and registered.
#             self._camera.RegisterImageEventHandler(ImageEvents(),
#                                                    pylon.RegistrationMode_Append,
#                                                    pylon.Cleanup_Delete)
#
#             # For demonstration purposes only, register another image event handler.
#             # self._camera.RegisterImageEventHandler(SampleImageEventHandler(),
#             #                                        pylon.RegistrationMode_Append,
#             #                                        pylon.Cleanup_Delete)
#
#             self._camera.SetCameraContext(self.cameraID)
#             initialized = True
#
#         except genicam.GenericException as e:
#             # Error handling.
#             self.log.fatal("Unable to initialize the capture", e.GetDescription())
#             initialized = False
#
#         return initialized
#
#     def initialize(self):
#         """
#         Set the camera parameters to reflect what we want them to be.
#         :return:
#         """
#
#         if not self._connected:
#             raise IOError("Camera is not connected.")
#
#         self.log.debug("Camera initialized")
#
#
#     def startCapturing(self):
#         # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
#         # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
#         # The GrabStrategy_OneByOne default grab strategy is used.
#         self.log.debug("Begin grab with OneByOne Strategy")
#         try:
#             #self.camera.Open()
#             self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
#         except _genicam.RuntimeException as e:
#             log.fatal("Failed to open the camera and start grabbing.")
#             log.fatal("{}".format(e))
#
#         # If we immediately start waiting for the trigger, we get an error
#         time.sleep(2)
#         self._capturing = True
#         while self._capturing:
#             try:
#                 if self.camera.WaitForFrameTriggerReady(400, pylon.TimeoutHandling_ThrowException):
#                     self.camera.ExecuteSoftwareTrigger()
#             except _genicam.TimeoutException as e:
#                 self.log.fatal("Timeout from camera")
#             except _genicam.RuntimeException as e:
#                 if not self._capturing:
#                     self.log.warning("Errors encountered in shutdown.  This is normal")
#                 else:
#                     self.log.error("Unexpected errors in capture")
#                     self.log.error("Device: {}".format(self._camera.GetDeviceInfo().GetModelName()))
#                     self.log.error("{}".format(e))
#             except Exception as e:
#                 self.log.error("Unable to execute wait for trigger")
#                 self.log.error(e)
#
#
#     def start(self):
#         """
#         Begin capturing images and store them in a queue for later retrieval.
#         """
#
#         if not self._connected:
#             raise IOError("Camera is not connected.")
#
#         # The scheme here is to get the images and store them for later consumption.
#         # The basler library does not have quite what is needed here, as we can't quite tell
#         # when an image is needed, as that is based on distance tranversed (let's say images every 10 cm to allow for
#         # overlap.
#
#         # Start grabbing images
#         self._camera.GevSCPSPacketSize = 8192
#         self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
#         self.log.debug("Started grabbing images")
#         # Fetch the images from the camera and store the results in a buffer
#         if self._strategy == constants.STRATEGY_ASYNC:
#             self.log.debug("Asynchronous capture")
#             self._capturing = True
#             while self._capturing:
#                 try:
#                     timestamped = self._grab()
#                     self._images.append(timestamped)
#                 except IOError as e:
#                     self.log.error(e)
#                 #self.log.debug("Image queue size: {}".format(len(self._images)))
#             self._camera.StopGrabbing()
#
#         # For synchronous capture, we don't do anything but retrieve the image on demand
#         else:
#             self.log.debug("Synchronous capture")
#
#
#
#     def stop(self):
#         """
#         Stop collecting from the current camera.
#         :return:
#         """
#
#         self.log.debug("Stopping image capture")
#
#         # Stop only if camera is connected.  This doesn't directly stop the collection, but clears the flag so the
#         # collection loop will stop
#         if self._connected:
#             self._capturing = False
#         # if self._strategy == constants.STRATEGY_ASYNC:
#         #     self._camera.StopGrabbing()
#
#         return
#
#     def disconnect(self):
#         """
#         Disconnected from the current camera and stop grabbing images
#         """
#
#         self.log.debug("Disconnecting from camera")
#         self.stop()
#         self._camera.Close()
#
#     def diagnostics(self) -> (bool, str):
#         """
#         Execute diagnostics on the camera.
#         :return:
#         Boolean result of the diagnostics and a string of the details
#         """
#         return True, "Camera diagnostics not provided"
#
#     def capture(self) -> ProcessedImage:
#         """
#         Capture a single image from the camera.
#         Requires calling the connect() method before this call.
#
#         If the image is in the queue, it will be served from there -- otherwise the method will retrieve if
#         synchronously from the camera.
#
#         :return:
#         The image as a numpy array
#         """
#
#         if not self._connected:
#             raise IOError("Camera is not connected")
#
#         # If there are no images in the queue, just wait for one.
#         while len(self._images) == 0:
#             self.log.error("Image queue is empty. Wait for a new image to appear")
#             time.sleep(0.1)
#
#         # The image we want is the one closest to the current time. The queue may contain a bunch of older images
#         processed = self._images.popleft()
#         img = processed.image
#         # The timestamp is in milliseconds
#         timestamp = processed.timestamp / 1000
#         self.log.debug("Image captured at UTC: {}".format(datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')))
#         return processed
#
#     def getResolution(self) -> ():
#         w = self._camera.get(cv.CAP_PROP_FRAME_WIDTH)
#         h = self._camera.get(cv.CAP_PROP_FRAME_HEIGHT)
#         return w, h
#
#     # This should be part of the calibration procedure
#     def getMMPerPixel(self) -> float:
#         return 0.0
#
#     @property
#     def camera(self) -> pylon.InstantCamera:
#         return self._camera
#
#     @camera.setter
#     def camera(self, openedCamera: pylon.InstantCamera):
#         self._camera = openedCamera
#
#
#     def _grabImage(self) -> ProcessedImage:
#
#         try:
#             grabResult = self._camera.RetrieveResult(200, pylon.TimeoutHandling_ThrowException)
#
#         except _genicam.RuntimeException as e:
#             self.log.fatal("Genicam runtime error encountered.")
#             self.log.fatal("{}".format(e))
#
#         if grabResult.GrabSucceeded():
#             # This is very noisy -- a bit more than we need here
#             self.log.debug("Image grab succeeded at timestamp " + str(grabResult.TimeStamp))
#         else:
#             raise IOError("Failed to grab image. Pylon error code: {}".format(grabResult.GetErrorCode()))
#
#         image = self._converter.Convert(grabResult)
#         img = image.GetArray()
#         # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
#         # We will mark the images based on when we got them -- ideally, this should be:
#         # timestamped = ProcessedImage(img, grabResult.TimeStamp)
#         timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
#         return timestamped
#
#     def _grab(self) -> ProcessedImage:
#         """
#         Grab the image from the camera
#         :return: ProcessedImag
#         """
#
#
#         try:
#             self.log.debug("Grab start")
#             grabResult = pypylon.pylon.GrabResult(self._camera.RetrieveResult(constants.TIMEOUT_CAMERA, pylon.TimeoutHandling_ThrowException))
#             self.log.debug("Grab complete")
#         except _genicam.RuntimeException as e:
#             self.log.fatal("Genicam runtime error encountered.")
#             self.log.fatal("{}".format(e))
#         # If the camera is close while we are capturing, this may be null.
#         if not grabResult.IsValid():
#             self.log.error("Image is not valid")
#             raise IOError("Image is not valid")
#
#         if grabResult.GrabSucceeded():
#             # This is very noisy -- a bit more than we need here
#             #self.log.debug("Image grab succeeded at timestamp " + str(grabResult.TimeStamp))
#             pass
#         else:
#             raise IOError("Failed to grab image. Pylon error code: {}".format(grabResult.GetErrorCode()))
#
#         image = self._converter.Convert(grabResult)
#         img = image.GetArray()
#         # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
#         # We will mark the images based on when we got them -- ideally, this should be:
#         #timestamped = ProcessedImage(img, grabResult.TimeStamp)
#         timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
#         return timestamped
#
#     def save(self, filename: str) -> bool:
#         """
#         Save the camera settings
#         :param filename: The file to contain the settings
#         :return: True on success
#         """
#         #self._camera.Open()
#         self.log.info("Saving camera configuration to: {}".format(filename))
#         pylon.FeaturePersistence.Save(filename, self._camera.GetNodeMap())
#         return True
#
#     def load(self, filename: str) -> bool:
#         """
#         Load the camera configuration from a file. Usually, this is the .pfs file saved from the pylon viewer
#         :param filename: The name of the file on disk
#         :return: True on success
#         """
#         loaded = False
#
#         #self._camera.Open()
#         # If the camera configuration exists, use that, otherwise warn
#         if os.path.isfile(filename):
#             self.log.info("Using saved camera configuration: {}".format(filename))
#             try:
#                 pylon.FeaturePersistence.Load(filename,self._camera.GetNodeMap(),True)
#             except _genicam.RuntimeException as geni:
#                 log.error("Unable to load configuration: {}".format(geni))
#
#             loaded = True
#         else:
#             self.log.warning("Unable to find configuration file: {}.  Camera configuration unchanged".format(filename))
#         return loaded
# #
# # E N D  C A M A R A S
# #
################################

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

captureTypeGroup = parser.add_mutually_exclusive_group()
captureTypeGroup.add_argument("-gr", "--grab", action="store_true", default=False, help="Just grab images. No processing")
captureTypeGroup.add_argument("-nu", "--null", action="store_true", default=False, help="Null processor. No processing")

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
parser.add_argument("-stand", "--standalone", action="store_true", default=False, help="Run standalone and just process the images")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")
parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Generate debugging data and text")
parser.add_argument("-x", "--xtract", action="store_true", default=False, help="Extract each crop plant into images")


arguments = parser.parse_args()

# This is just the root of the output directory, typically ../output.  Later, this will be ../output/<UUID> for
# a specific session

outputDirectory = arguments.output

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

#
# D E P T H  C A M E R A
#
def startupRGBDepthCamera(options: OptionsFile) -> CameraDepth:
    """
    Starts the attached depth camera
    :return: The depth camera instance or None if the camera cannot be found.
    """
    global intel435Status
    sensors = RealSense()
    sensors.query()
    markSensorAsFailed = False
    cameraForDepth = None

    if sensors.count() < 1:
        log.error("Detected {} depth/IMU sensors. Expected at least 1.".format(sensors.count()))
        log.error("No sensor will be used.")
        markSensorAsFailed = True
        intel435Status = constants.OperationalStatus.FAIL

    # Start the Depth Cameras
    try:
        cameraForDepth = CameraDepth(constants.Capture.DEPTH_RGB)
        if markSensorAsFailed:
            cameraForDepth.state.toMissing()
        else:
            cameraForDepth.state.toIdle()
            intel435Status = constants.OperationalStatus.OK
    except KeyError:
        log.error("Unable to find serial number for depth camera: {}/{} & {}".format(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_SERIAL_LEFT, constants.PROPERTY_SERIAL_RIGHT))
        cameraForDepth.state.toMissing()

    # log.debug("Using Intel camera: {}".format(cameraForDepth))
    return cameraForDepth

def startupCamera(options: OptionsFile) -> Camera:
    if arguments.input is not None:
        # Get the images from a directory
        theCamera = CameraFile(directory=arguments.input)
        theCamera.gsd = int(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_IMAGE_WIDTH))
    else:
        # Get the images from an actual camera
        cameraIP = options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP)
        # The version of the BaslerCamera defined here
        #theCamera = _CameraBasler(ip=cameraIP)
        theCamera = CameraBasler(ip=cameraIP, capture=constants.CAPTURE_STRATEGY_QUEUED)
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
        communicator = ClientMQCommunicator(SERVER=options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_SERVER),
                                            PORT=constants.PORT_ODOMETRY)
    except KeyError:
        log.error("Unable to find {}/{} in ini file".format(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_SERVER))
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

    return odometryRoom, systemRoom, treatmentRoom
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

def resample(index: np.ndarray, targetX: int, targetY: int) -> np.ndarray:
    # Hardcode this for now -- depth is 1280x720, and we want 1920x1080

    #z = (1920 / 1280, 1080 / 720)
    z = (targetY / 1920, targetX / 1080)

    transformed = scipy.ndimage.zoom(index, z, order=0)
    return transformed

# The plotly version
def plot3D(index: np.ndarray, title: str):

    if not supportsPlotting:
        log.error("Unable to produce plots on this platform")
        return

    # I can get plotly to work only with square arrays, not rectangular, so just take a subset
    subset = index[0:1500, 0:1500]
    log.debug("Index is {}".format(index.shape))
    log.debug("Subset is {}".format(subset.shape))
    xi = np.linspace(0, subset.shape[0], num=subset.shape[0])
    yi = np.linspace(0, subset.shape[1], num=subset.shape[1])

    fig = go.Figure(data=[go.Surface(x=xi, y=yi, z=subset)])

    # Can't get these to work
    #fig = go.Figure(data=[go.Mesh3d(x=xi, y=yi, z=subset, color='lightpink', opacity=0.50)])
    #fig = go.Figure(data=go.Isosurface(x=xi, y=yi,z=subset, isomin=-1, isomax=1))

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

    yLen,xLen = downsampled.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    log.debug("3D plot x: {} y: {}".format(x,y))

    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,10))
    axes = fig.gca(projection ='3d')
    plt.title(title)
    axes.scatter(x, y, downsampled, c=downsampled.flatten(), cmap='BrBG', s=0.25)
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

def nullProcessor(contextForImage: Context, captureType: constants.Capture) -> bool:
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

def storeImage(contextForImage: Context, captureType: constants.Capture) -> bool:
    global imageNumberBasler
    global imageNumberIntel

    if not processing:
        log.debug("Not collecting images (This is normal if the weeding has not started")
        return False

    # log.info("Storing image {} type {}".format(imageNumberBasler, captureType))

    start = time.time()
    if captureType == constants.Capture.DEPTH_RGB:
        # The RGB image from the Intel camera
        if rgbDepthCamera.connected:
            try:
                performance.start()
                # Capture the image from the Intel Camera
                rgbDepth = rgbDepthCamera.capture()
                #performance.stopAndRecord(constants.PERF_ACQUIRE_INTEL_RGB)
                # It may be that file writes are throwing times off
                performance.stop()

                # DEPTH data
                imageName = "{}-{}-{:05d}".format(constants.FILENAME_INTEL_DEPTH, options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION), imageNumberIntel)
                #depthPath = os.path.join(logger.directory, imageName)
                depthPath = os.path.join(arguments.output + "/" + currentSessionName + "/", imageName)

                processedImage = ProcessedImage(constants.Capture.DEPTH_DEPTH, rgbDepth.depth, 0)
                processedImage.filename = depthPath
                processedImage.make = "intel"
                processedImage.model = "435"
                processedImage.exposure = contextForImage.exposure
                processedImage.latitude = contextForImage.latitude
                processedImage.longitude = contextForImage.longitude
                # Put the image into the queue for further processing
                # log.debug("Adding Intel depth to queue")
                rawImages.enqueue(processedImage)

                # R G B  A S  N U M P Y
                imageName = "{}-intel-{}-{:05d}".format(constants.FILENAME_RAW, options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION), imageNumberIntel)
                #rgbPath = os.path.join(logger.directory, imageName)
                rgbPath = os.path.join(arguments.output + "/" + currentSessionName + "/", imageName)

                processedImage = ProcessedImage(constants.Capture.DEPTH_RGB, rgbDepth.rgb, 0)
                processedImage.urlFilename = currentSessionName + "/" + imageName
                processedImage.filename = rgbPath
                processedImage.make = "intel"
                processedImage.model = "435"
                processedImage.exposure = contextForImage.exposure
                processedImage.latitude = contextForImage.latitude
                processedImage.longitude = contextForImage.longitude
                # Put the image into the queue for further processing
                # log.debug("Adding Intel RGB to queue")
                rawImages.enqueue(processedImage)

                imageNumberIntel += 1
            except IOError as e:
                log.fatal("Cannot capture RGB data ({})".format(e))
        else:
            depthData = None

    elif captureType == constants.Capture.RGB:
        performance.start()
        try:
            processed = camera.capture()
            rawImage = processed.image
        except IOError as e:
            log.fatal("Cannot capture image. ({})".format(e))
            return False

        performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        # Set the context and enqueue the image
        imageName = "{}-basler-{}-{:05d}".format(constants.FILENAME_RAW, options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION), imageNumberBasler)
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

        imageNumberBasler += 1

    # log.debug("Image Processing time: {}".format(time.time() - start))

    return True

mmPerPixel = 0.01

def processImage(contextForImage: Context) -> bool:
    global imageNumberBasler
    global sequence
    global previousImage

    try:

        if arguments.verbose:
            print("Processing image " + str(imageNumberBasler))
        log.info("Processing image " + str(imageNumberBasler))
        performance.start()

        # Attempt to capture the image.
        try:
            processed = camera.capture()
            rawImage = processed.image
        except EOFError as eof:
            # This case is where we just hit the end of an image set from disk
            log.info("Encountered end of image set")
            return False
        except IOError as io:
            # This is the case where something went wrong with a grab from a camera
            log.error("Encountered I/O Error {}".format(io))
            return False

        performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        #ImageManipulation.show("Source",image)
        veg.SetImage(rawImage)

        # Attempt to capture the depth data if connected
        if rgbDepthCamera.connected:
            try:
                depthData = rgbDepthCamera.capture()
            except EOFError as eof:
                # This case is where we just hit the end of an image set from disk
                log.info("Encountered end of image set")
                return False
            except IOError as io:
                # This is the case where something went wrong with a grab from a camera
                log.error("Encountered I/O Error for depth camera: {}".format(io))
                return False
        else:
            log.warning("Depth camera is not connected")

        performance.stopAndRecord(constants.PERF_ACQUIRE_BASLER_RGB)

        manipulated = ImageManipulation(rawImage, imageNumberBasler, logger)
        logger.logImage(constants.FILENAME_RAW, manipulated.image)

        #manipulated.mmPerPixel = mmPerPixel
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        performance.start()
        index = veg.Index(arguments.algorithm)
        performance.stopAndRecord(constants.PERF_INDEX)

        #ImageManipulation.show("index", index)
        #cv.imwrite("index.jpg", index)
        if arguments.plot:   plot3D(index, arguments.algorithm)

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
            #veg.ShowImage("Thresholded", imageThresholded)
            logger.logImage("inverted", filledMaskInverted)
            #veg.ShowImage("Filled", filledMask)
            #veg.ShowImage("Inverted", filledMaskInverted)
            #veg.ShowImage("Final", finalMask)
            logger.logImage("final", finalMask)
            #plt.imshow(veg.imageMask, cmap='gray', vmin=0, vmax=1)
            #plt.imshow(finalImage)

        #print("X={}".format(x))            #plt.show()
            #logger.logImage("mask", veg.imageMask)
        #ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(finalImage, imageNumberBasler, logger)
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
            return False

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


        # Determine the distance from the object to the edge of the image given  the pixel size of the camera
        performance.start()
        manipulated.computeDistancesToImageEdge(camera.getMMPerPixel(), camera.getResolution())
        performance.stopAndRecord(constants.PERF_DISTANCE)

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

        # Everything in the image is classified, so decorate the image with distances
        manipulated.drawDistances()

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

        imageNumberBasler = imageNumberBasler + 1

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
if arguments.null:
    processor = nullProcessor
elif arguments.grab:
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
    finished = os.path.join(logger.directory, options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_FILENAME_FINISHED))
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
        log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

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
        log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

    # Include the status of system components in the report
    systemMessage.statusCamera = camera.status.name
    systemMessage.statusSystem = systemStatus.name
    systemMessage.statusIntel = intel435Status.name
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

    performance.start()
    odometryMessage = OdometryMessage(raw=message)

    # We are only concerned with distance messages here
    if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
        # This assumes we haven't missed any messages
        #totalMovement += odometryMessage.distance
        #movementSinceLastProcessing += odometryMessage.distance
        #movementSinceLastProcessingForIntel += odometryMessage.distance

        totalMovement = odometryMessage.totalDistance
        movementSinceLastProcessing = totalMovement - distanceAtLastBaslerProcessing
        movementSinceLastProcessingForIntel = totalMovement - distanceAtLastIntelProcessing

        if odometryMessage.sequence == -1:
            log.debug("Odometry sequence number restarted at -1. Not processing this message")
            currentSequenceNumberForOdometry = 0
            return
        # If the sequence number is the same as the last one, ignore
        elif odometryMessage.sequence == currentSequenceNumberForOdometry:
            #log.debug("Sequence number {} already processed.".format(currentSequenceNumberForOdometry))
            return
        # Missed a message -- update the current sequence and process
        elif odometryMessage.sequence != currentSequenceNumberForOdometry + 1:
            # Produce an error message if the movement is greater than the overlap of the GSD. Slightly exceeding the GSD is normal
            # If the movement is 410mm and the GSD is 400mm, that's OK, but if the movement was 420.5mm in the same
            # situation, we want to know
            gap = odometryMessage.sequence - (currentSequenceNumberForOdometry + 1)
            overlapFactor = float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))
            if movementSinceLastProcessing > gsdBasler * (1 + overlapFactor):
                log.fatal("Missed movement exceeds GSD + overlap of Basler: {} Message gap {}".format(movementSinceLastProcessing, gap))
            if movementSinceLastProcessingForIntel > gsdIntel * (1 + overlapFactor):
                log.fatal("Missed movement exceeds GSD + overlap of Intel: {} Message gap {}".format(movementSinceLastProcessingForIntel,gap))

            # log.error("Missed odometry sequence. Expected {} processed {} gap {}".format(currentSequenceNumberForOdometry + 1, odometryMessage.sequence, gap))
            currentSequenceNumberForOdometry = odometryMessage.sequence
        # Normal case
        else:
            currentSequenceNumberForOdometry = odometryMessage.sequence
            # log.debug("Processing Odometry message: {}".format(message))

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
            # Temporary -- don't get the basler image
            # log.debug("Acquiring image from Basler.  Movement since last processing {} GSD {}".format(movementSinceLastProcessing, gsdBasler))
            # movementSinceLastProcessing = 0
            #
            # # Record the context under which this photo was taken
            # contextForImage = Context()
            # contextForImage.latitude = odometryMessage.latitude
            # contextForImage.longitude = odometryMessage.longitude
            # # Convert to kilometers
            # contextForImage.speed = odometryMessage.speed / 1e+6
            # contextForPhoto.model
            # processor(contextForImage, constants.Capture.RGB)

        # The intel RGB camera
        if movementSinceLastProcessingForIntel > gsdIntel:
            # log.debug("Acquiring image from Intel Camera.  Movement since last processing {} GSD {}".format(movementSinceLastProcessingForIntel, gsdIntel))
            movementSinceLastProcessingForIntel = 0
            distanceAtLastIntelProcessing = totalMovement

            # Record the context under which this photo was taken
            contextForImage = Context()
            contextForImage.latitude = odometryMessage.latitude
            contextForImage.longitude = odometryMessage.longitude
            # Convert to kilometers
            contextForImage.speed = odometryMessage.speed / 1e+6
            # contextForPhoto.model
            processor(contextForImage, constants.Capture.DEPTH_RGB)
    else:
        pass
        # log.debug("Message type is not distance. Ignored")


    # Too noisy -- less than 0.25 ms on the jetsons.  Good enough
    #log.debug("Processed odometry message: {} ms".format(performance.stop()))

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
                #     totalMovement += odometryMessage.distance
                #     movementSinceLastProcessing += odometryMessage.distance
                #     movementSinceLastProcessingForIntel += odometryMessage.distance
                #     # The time of the observation
                #     timeRead = odometryMessage.timestamp
                #     # Determine how old the observation is
                #     # The version of python on the jetson does not support time_ns, so this a bit of a workaround until I
                #     # get that sorted out.  Just convert the reading to milliseconds for now
                #     #timeDelta = (time.time() * 1000) - (timeRead / 1000000)
                #     timeDelta = (time.time() * 1000) - timeRead
                #
                #     # If the movement is equal to the size of the image, take a picture
                #     # We need to allow for some overlap so the images can be stitched together.
                #     # So reduce this by the overlap factor
                #     # TODO: Optimize by moving this calculation out
                #     gsdBasler = (1 - float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * camera.gsd
                #     gsdIntel = (1 - float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * rgbDepthCamera.gsd
                #     #
                #     # log.debug("Total movement: {} at time: {}. Movement: {} GSD [Basler: {} Intel: {}] Time now is {} delta from now {} ms".
                #     #           format(totalMovement, timeRead, movementSinceLastProcessing, gsdBasler, gsdIntel, time.time() * 1000, timeDelta))
                #
                #     # The Basler camera
                #     if movementSinceLastProcessing > gsdBasler:
                #         log.debug("Acquiring image from Basler.  Movement since last processing {} GSD {}".format(movementSinceLastProcessing, gsdBasler))
                #         movementSinceLastProcessing = 0
                #
                #         # Record the context under which this photo was taken
                #         contextForImage = Context()
                #         contextForImage.latitude = odometryMessage.latitude
                #         contextForImage.longitude = odometryMessage.longitude
                #         # Convert to kilometers
                #         contextForImage.speed = odometryMessage.speed / 1e+6
                #         # contextForPhoto.model
                #         processor(contextForImage, constants.Capture.RGB)
                #     # The intel RGB camera
                #     elif movementSinceLastProcessingForIntel > gsdIntel:
                #         log.debug("Acquiring image from Intel Camera.  Movement since last processing {} GSD {}".format(movementSinceLastProcessingForIntel, gsdIntel))
                #         movementSinceLastProcessingForIntel = 0
                #
                #         # Record the context under which this photo was taken
                #         contextForImage = Context()
                #         contextForImage.latitude = odometryMessage.latitude
                #         contextForImage.longitude = odometryMessage.longitude
                #         # Convert to kilometers
                #         contextForImage.speed = odometryMessage.speed / 1e+6
                #         # contextForPhoto.model
                #         processor(contextForImage, constants.Capture.DEPTH_RGB)
                else:
                    pass
                    # log.debug("Message type is not distance. Ignored")
            else:
                log.info("Old message seen -- ignored")
        else:
            # There's not much to do here for keepalive messages
            keepAliveMessages += 1
            #print("weeds: keepalive message from chatroom")
    elif msg.getType() == "chat":
        log.error("Private Message from {}: {} ".format(str(msg.getFrom()), str(msg.getBody())))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

    log.debug("Processed odometry message: {} ms".format(performance.stop()))

# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageTreatmentCB(conn,msg: xmpp.protocol.Message):
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
        body = msg.getBody()
        # Check if this is a real message and not just an empty keep-alive message
        # if body is not None:
        #     log.debug("treatment message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
    elif msg.getType() == "chat":
            print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
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

def takeDepthImages(camera: CameraDepth):
    """
    Take depth images from the Intel 435 camera
    :param camera: CameraDepth instance
    :return:
    """
    global intel435Status
    cameraConnected = False

    if camera is None:
        log.error("Depth camera is not created")
        return cameraConnected

    # Connect to the camera and take an image
    log.debug("Connecting to depth camera")
    cameraConnected = camera.connect()

    if cameraConnected:
        if isinstance(camera, CameraDepth):
            camera._state.toClaim()
            camera.initialize()
            camera.start()

            if camera.initializeCapture():
                try:
                    camera.startCapturing()
                except IOError as io:
                    camera.log.error(io)
                rc = 0
            else:
                log.fatal("Unable to initialize depth camera")
                rc = -1
        else:
            log.error("Not a depth camera")
            intel435Status = constants.OperationalStatus.FAIL
    else:
        log.error("Unable to connect to depth camera")
        intel435Status = constants.OperationalStatus.FAIL

#
# Take the images -- this method will not return, only add new images to the queue
#
def takeRGBImages(camera: CameraBasler):

    cameraConnected = False

    # Connect to the camera and take an image
    log.debug("Connecting to camera")

    capturing = True
    while capturing:
        log.debug("Dummy basler capture loop")
        time.sleep(10)

    return

    camera.connect()
    camera.start()

    if cameraConnected:
        if isinstance(camera, CameraBasler):
            # The camera settings are stored in files like aca-2500-gc.pfs
            # This will be used for call capture parameters
            # Get the specific settings for location and position <location>-<position>.pfs
            try:
                location = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_LOCATION)
                position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
                filename = "{}-{}.pfs".format(location, position)
            except KeyError as key:
                log.error("Unable to find location/position in ini file. Expected {}/{} and {}/{}"
                          .format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_LOCATION,
                                  constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))
                filename = camera.camera.GetDeviceInfo().GetModelName() + ".pfs"
                log.error("Using camera defaults from {}".format(filename))

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
                    log.debug("Beginning Basler RGB Capture")
                    camera.startCapturing()
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
                    "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

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

        elif rawImage.captureType == constants.Capture.RGB:
            log.debug("Saving Basler RGB: {}".format(rawImage.filename + constants.EXTENSION_IMAGE))
            # ImageManipulation.show("Source",image)
            performance.start()
            veg.SetImage(rawImage.image)

            manipulated = ImageManipulation(rawImage.image, imageNumberBasler, logger)
            imageName = rawImage.filename + constants.EXTENSION_IMAGE
            try:
                cv.imwrite(imageName, rawImage.image)
            except Exception:
                log.error("Unable to save Basler RGB")

            #fileName = logger.logImage(imageName, manipulated.image)

            enricher.addEXIFToImageAndWriteToDisk(rawImage)
            performance.stopAndRecord(constants.PERF_SAVE_BASLER_RGB)

            # Send out a message to the treatment channel that an image has been taken and is available.
            message = TreatmentMessage()
            message.plan = constants.Treatment.RAW_IMAGE
            message.source = constants.Capture.RGB
            message.name = "original"
            message.url = "http://" + platform.node() + "/" + currentSessionName + "/" + imageName
            message.timestamp = time.time() * 1000

            try:
                position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
                message.position = position
            except KeyError:
                log.error(
                    "Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

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
options = readINI()

# Diagnostics will appear in the same directory structure as the output files
systemDiagnostics = Diagnostics("../output", options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION) + constants.EXTENSION_HTML)
systemDiagnostics.writeHTML()

# Set the status of all components to failed initially
systemStatus = constants.OperationalStatus.FAIL
intel435Status = constants.OperationalStatus.FAIL
odometryStatus = constants.OperationalStatus.FAIL

currentSessionName = ""
currentOperation = constants.Operation.QUIESCENT.name

(logger, log) = startupLogger(arguments.output)
#log = logging.getLogger(__name__)

camera = startupCamera(options)
log.debug("RGB camera started")

# This is confusing -- there are TWO rgb streams available: Basler and Intel.
# This is for the latter
rgbDepthCamera = startupRGBDepthCamera(options)

# This is the GSD of the image that takes into account overlap
gsdBasler = (1 - float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * camera.gsd
gsdIntel = (1 - float(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_OVERLAP_FACTOR))) * rgbDepthCamera.gsd
log.debug("GSD Basler: {}/{} GSD Intel: {}/{}".format(gsdBasler, camera.gsd, gsdIntel, rgbDepthCamera.gsd))

(roomOdometry, roomSystem, roomTreatment) = startupCommunications(options, messageOdometryCB, messageSystemCB, messageTreatmentCB)
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

    log.debug("Start RGB image acquisition")
    acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeRGBImages, args=(camera,))
    acquire.daemon = True
    threads.append(acquire)
    acquire.start()

    # Unfortunately: looks like I can't create two cameras for depth and RGB streams -- just one.
    # log.debug("Start depth data acquisition")
    # acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeDepthImages, args=(depthCamera,))
    # threads.append(acquire)
    # acquire.start()

    log.debug("Start Intel RGB image acquisition")
    acquireRGB = threading.Thread(name=constants.THREAD_NAME_ACQUIRE_RGB, target=takeDepthImages, args=(rgbDepthCamera,))
    acquireRGB.daemon = True
    threads.append(acquireRGB)
    acquireRGB.start()

    # TODO: This thread is no longer needed once MQ commmunications is debugged
    # log.debug("Starting odometry MUC receiver")
    # #generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=processMessages, args=(roomOdometry,))
    # generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=roomOdometry.processMessages, args=())
    # generator.daemon = True
    # threads.append(generator)
    # generator.start()

    log.debug("Starting odometry MQ receiver")
    odometryProcessor = threading.Thread(name=constants.THREAD_NAME_REQ_RSP, target=processMQ, args=(odometryMQ,))
    odometryProcessor.daemon = True
    threads.append(odometryProcessor)
    odometryProcessor.start()

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
    while processor(contextForImage):
        pass


performance.cleanup()

# Not quite right here to get the list of all blobs from the reporting module
#classifier.train(reporting.blobs)

result, reason = reporting.writeSummary()

# if not result:
#     print(reason)
#     sys.exit(1)
# else:
#     sys.exit(0)
