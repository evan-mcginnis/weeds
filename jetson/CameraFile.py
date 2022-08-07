#
# C A M E R A F I L E
#
import datetime
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
import yaml

import constants
from Performance import Performance

cameras = list()
cameraCount = 0

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
        log.debug("OnImageGrabbed event for device: {}".format(camera.GetDeviceInfo().GetModelName()))

        log.debug("Context is {}".format(camera.GetCameraContext()))

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
            log.debug("Camera context is {}".format(cameraNumber))
            camera = cameras[cameraNumber]
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

    def __init__(self, **kwargs):
        global cameraCount

        # Register the camera on the global list so we can keep track of them
        # Even though there will probably be only one
        self.cameraID = cameraCount
        cameraCount += 1
        cameras.append(self)
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
                self._camera = pylon.InstantCamera(tl_factory.CreateDevice(dev_info))
                #self._camera.MaxNumBuffer = 100
                try:
                    # The sample code does _not_ call Open(), so for debug purposes, this is commented out here.
                    #self._camera.Open()
                    self.log.info("Using device {} at {}".format(self._camera.GetDeviceInfo().GetModelName(), dev_info.GetIpAddress()))
                    self._connected = True
                except Exception as e:
                    self.log.fatal("Error encountered in opening camera")
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
        camera.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)

        self._capturing = True
        while self._capturing:
            try:
                if camera.camera.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
                    camera.camera.ExecuteSoftwareTrigger();
            except _genicam.TimeoutException as e:
                self.log.fatal("Timeout from camera")

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
                self.log.debug("Image queue size: {}".format(len(self._images)))
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


    def getkey():
        return input("Enter \"t\" to trigger the camera or \"e\" to exit and press enter? (t/e) ")

    def takeImages(camera: CameraBasler):

        # Connect to the camera and take an image
        log.debug("Connecting to camera")
        camera.connect()
        if camera.initializeCapture():
            try:
                camera.startCapturing()
            except IOError as io:
                camera.log.error(io)
            rc = 0
        else:
            rc = -1


    parser = argparse.ArgumentParser("Basler Camera Utility")

    parser.add_argument('-s', '--single', action="store", required=True, help="Take a single picture")
    parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Log file configuration")
    parser.add_argument('-p', '--performance', action="store", required=False, default="camera.csv", help="Performance file")
    parser.add_argument('-o', '--options', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
    parser.add_argument('-a', '--asynch', action="store_true", required=False, default=False, help="Use asynchronous image acquisition")
    arguments = parser.parse_args()

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("jetson")

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

    # BEGIN WORKS
    # camera.connect()
    # camera.initializeCapture()
    # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
    # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
    # The GrabStrategy_OneByOne default grab strategy is used.
    #camera.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)

    # Wait for user input to trigger the camera or exit the program.
    # The grabbing is stopped, the device is closed and destroyed automatically when the camera object goes out of scope.
    # while True:
    #     time.sleep(0.05)
    #     key = getkey()
    #     print(key)
    #     if (key == 't' or key == 'T'):
    #         # Execute the software trigger. Wait up to 100 ms for the camera to be ready for trigger.
    #         if camera.camera.WaitForFrameTriggerReady(100, pylon.TimeoutHandling_ThrowException):
    #             camera.camera.ExecuteSoftwareTrigger();
    #     if (key == 'e') or (key == 'E'):
    #         break

    # for i in range(100):
    #     #time.sleep(0.5)
    #     if camera.camera.WaitForFrameTriggerReady(200, pylon.TimeoutHandling_ThrowException):
    #         camera.camera.ExecuteSoftwareTrigger();

    #sys.exit(1)
    # END WORKS

    # Start the thread that will begin acquiring images
    acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeImages, args=(camera,))
    acquire.start()

    acquire.join()

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

