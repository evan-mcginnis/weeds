# W A R N I N G
#
# The Basler logic is just copied to the weeds.py file
# This works just fine here, but not when that file imports the class
#

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
        log.debug("OnImageGrabbed event for device: {}".format(camera.GetDeviceInfo().GetModelName()))

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            log.debug("Image grabbed successfully")
            start = time.time()
            # Convert the image grabbed to something we like
            # image = CameraBasler(grabResult)
            # self.log.debug(f"Basler image converted time: {time.time() - start} s")
            image = DebugCameraBasler.convert(grabResult)
            img = image.GetArray()
            # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
            # We will mark the images based on when we got them -- ideally, this should be:
            # timestamped = ProcessedImage(img, grabResult.TimeStamp)
            timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))

            cameraNumber = camera.GetCameraContext()
            camera = Camera.cameras[cameraNumber]
            #log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
            camera._images.append(timestamped)

            # print("SizeX: ", grabResult.GetWidth())
            # print("SizeY: ", grabResult.GetHeight())
            # img = grabResult.GetArray()
            # print("Gray values of first row: ", img[0])
            # print()
            grabResult.Release()
        else:
            log.error("Image Grab error code: {} {}".format(grabResult.GetErrorCode(), grabResult.GetErrorDescription()))

# Example of an image event handler.
class SampleImageEventHandler(pypylon.pylon.ImageEventHandler):
    def OnImageGrabbed(self, _camera, grabResult):
        print("CSampleImageEventHandler::OnImageGrabbed called.")
        print()
        print()

class DebugCameraBasler(Camera):
    # Initialize the converter for images
    # The images stream of in YUV color space.  An optimization here might be to make
    # both formats available, as YUV is something we will use later

    _converter = pylon.ImageFormatConverter()

    # converting to opencv bgr format
    _converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    # Temporary -- the setting is already in place on the camera
    #_converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

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

        # if constants.KEYWORD_CONFIGURATION_EVENTS in kwargs:
        #     self.log.debug("Using supplied configuration event printer")
        #     self._configurationEvents = kwargs[constants.KEYWORD_CONFIGURATION_EVENTS]
        # else:
        #     self.log.debug("Creating configuration event printer")
        #     self._configurationEvents = ConfigurationEventPrinter()
        #
        # if constants.KEYWORD_IMAGE_EVENTS in kwargs:
        #     self.log.debug("Using supplied image event printer")
        #     self._imageEvents = kwargs[constants.KEYWORD_IMAGE_EVENTS]
        # else:
        #     self.log.debug("Creating Image Event printer")
        #     self._imageEvents = ImageEvents()

        super().__init__(**kwargs)

    @classmethod
    def convert(cls, grabResult):
        """
        Converts the grab result into the format expected by the rest of the system
        :param grabResult: A grab from the basler camera
        :return:
        """
        start = time.time()
        image = DebugCameraBasler._converter.Convert(grabResult)
        print(f"Image conversion took: {time.time() - start} seconds")
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
                self._camera.MaxNumBuffer = 100
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

    def initializeCapture(self, configCallbacks, imageCallbacks):

        initialized = False
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

            self._camera.RegisterConfiguration(configCallbacks,
                                               pylon.RegistrationMode_Append,
                                               pylon.Cleanup_Delete)

            # The image event printer serves as sample image processing.
            # When using the grab loop thread provided by the Instant Camera object, an image event handler processing the grab
            # results must be created and registered.
            # Originally
            # self._camera.RegisterImageEventHandler(ImageEvents(),
            #                                        pylon.RegistrationMode_Append,
            #                                        pylon.Cleanup_Delete)

            self._camera.RegisterImageEventHandler(imageCallbacks,
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


        self.log.debug("Camera initialized")

    def startCapturing(self):
        # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
        # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
        # The GrabStrategy_OneByOne default grab strategy is used.
        try:
            #self.camera.Open()
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
            self.log.debug("Start Capturing with OneByOne Strategy")
        except _genicam.RuntimeException as e:
            self.log.fatal("Failed to open the camera and start grabbing.")
            self.log.fatal("{}".format(e))

        # If we immediately start waiting for the trigger, we get an error
        time.sleep(3)
        self._capturing = True

        # This is for a dummy capture
        # while self._capturing:
        #     time.sleep(10)
        #     self.log.debug("Dummy capture of Basler RGB")

        while self._capturing:
            try:
                for i in range(5):
                    if self.camera.WaitForFrameTriggerReady(2000, pylon.TimeoutHandling_ThrowException):
                        self.camera.ExecuteSoftwareTrigger()
            except _genicam.TimeoutException as e:
                self.log.fatal("Timeout from camera in WaitForFrameTrigger ready")
                time.sleep(1)
                self.log.fatal(e)
            except _genicam.RuntimeException as e:
                if not self._capturing:
                    self.log.warning("Errors encountered in shutdown.  This is normal")
                else:
                    self.log.error("Unexpected errors in capture")
                    self.log.error("Device: {}".format(self._camera.GetDeviceInfo().GetModelName()))
                    self.log.error("{}".format(e))
            except Exception as e:
                self.log.error("Unable to execute wait for trigger")
                self.log.error(e)

    def start(self):
        """
            Begin capturing images and store them in a queue for later retrieval.
            """

        # Use the startCapturing() method for the loop
        return

        #######################
        # This is based ob a basler example
        #######################

        self._camera.Open()
        self._camera.MaxNumBuffer = 40
        self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        i = 0
        self.log.debug('Starting to acquire')
        while self._camera.IsGrabbing():
            for i in range(3):
                self.log.debug("Waiting for FrameTriggerReady")
                if self._camera.WaitForFrameTriggerReady(2000, pylon.TimeoutHandling_ThrowException):
                    self._camera.ExecuteSoftwareTrigger()
            time.sleep(0.2)
            # Check that grab results are waiting.

            if self._camera.GetGrabResultWaitObject().Wait(0):
                self.log.debug("Grab results wait in the output queue.")
            else:
                self.log.error("No results are waiting")

            # All triggered images are still waiting in the output queue
            # and are now retrieved.
            # The grabbing continues in the background, e.g. when using hardware trigger mode,
            # as long as the grab engine does not run out of buffers.
            start = time.time()
            grab = self._camera.RetrieveResult(0, pylon.TimeoutHandling_Return)
            img = grab.Array
            grab.Release()
            # timestamped = ProcessedImage(constants.Capture.RGB, grab, round(time.time() * 1000))
            # Temporary -- attach the image, not the grab
            timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
            timestamped.type = constants.ImageType.BASLER_RAW

            cameraNumber = self._camera.GetCameraContext()
            # self._camera = Camera.cameras[cameraNumber]
            # log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
            self.log.debug(f"Grabbed image processed and enqueued in {time.time() - start:.8f} seconds")
            self._images.append(timestamped)

        ###############################
        # This is the original logic
        ###############################

        self._camera.Open()
        self._camera.MaxNumBuffer = 40
        self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
        i = 0
        self.log.debug('Starting to acquire')
        while self._camera.IsGrabbing():
            try:
                t0 = time.time()
                grab = self._camera.RetrieveResult(900, pylon.TimeoutHandling_ThrowException)
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

                #image = CameraBasler.convert(grab)
                #self.log.debug(f"Basler image converted time: {time.time() - start} s")
                #img = image.GetArray()

                # The 1920 and 2500 cameras do not support PTP, so the timestamp is just the ticks since startup.
                # We will mark the images based on when we got them -- ideally, this should be:
                # timestamped = ProcessedImage(img, grabResult.TimeStamp)
                # Orignally
                #timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
                # Create a processed image that has not yet been converted
                # Use the pylon methods for everything
                # img = pylon.PylonImage()
                # img.AttachGrabResultBuffer(grab)
                img = grab.Array
                grab.Release()
                #timestamped = ProcessedImage(constants.Capture.RGB, grab, round(time.time() * 1000))
                # Temporary -- attach the image, not the grab
                timestamped = ProcessedImage(constants.Capture.RGB, img, round(time.time() * 1000))
                timestamped.type = constants.ImageType.BASLER_RAW

                cameraNumber = self._camera.GetCameraContext()
                #self._camera = Camera.cameras[cameraNumber]
                #log.debug("Camera context is {} Queue is {}".format(cameraNumber, len(camera._images)))
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
            self.log.debug("Image captured at UTC: {}".format(datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')))

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

if __name__ == "__main__":
    import argparse
    import threading
    from PIL import Image

    parser = argparse.ArgumentParser("Basler Camera Utility")

    parser.add_argument('-s', '--single', action="store", required=False, default="single.jpg", help="Take a single picture")
    parser.add_argument('-c', '--camera', action="store", required=True, help="IP Address of camera")
    parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Log file configuration")
    parser.add_argument('-p', '--performance', action="store", required=False, default="camera.csv", help="Performance file")
    parser.add_argument('-o', '--options', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
    parser.add_argument('-type', '--type', action="store", required=False, default='live', choices=['live', 'queued'], help="Get a live or queued image")
    arguments = parser.parse_args()

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("iron-chef")

    def takeRGBImages(cam: DebugCameraBasler):
        log.debug("Taking images")
        cam.connect()
        cam.initializeCapture(configurationEvents, imageEvents)
        cam.startCapturing()

    configurationEvents = ConfigurationEventPrinter()
    imageEvents = ImageEvents()
    camera = DebugCameraBasler(ip=arguments.camera, capture=constants.CAPTURE_STRATEGY_QUEUED, configuration=configurationEvents, image=imageEvents)
    #camera = CameraBasler(ip=arguments.camera, capture=arguments.type)
    if arguments.type == 'live':
        camera.connect()
        start = time.time()
        processed = camera.capture()
        print(f'Captured image in {time.time() - start} s')
    else:
        acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeRGBImages, args=(camera,))
        acquire.daemon = True
        acquire.start()

        log.debug("Sleeping to allow queue buildup")
        time.sleep(20)
        start = time.time()
        processed = camera.capture()
        log.debug(f'Captured image in {time.time() - start} s')
        image = Image.fromarray(processed.image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(arguments.single)
