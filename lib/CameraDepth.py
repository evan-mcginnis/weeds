#
# C A M E R A D E P T H
#
import pathlib
import logging
import logging.config
import time
from collections import deque
import signal
from io import TextIOWrapper

import numpy as np
import os
from abc import ABC, abstractmethod

from statemachine.exceptions import TransitionNotAllowed

import constants
from Performance import Performance

from Camera import Camera
from WeedExceptions import DepthUnknownStream
from TimestampedImage import TimestampedImage
from DepthImage import DepthImage

import pyrealsense2 as rs
import numpy as np

from RealSense import RealSense

#
# The Basler camera is accessed through the pylon API
# Perhaps this can be through openCV, but this will do for now
#

class CameraDepth(Camera):

    def __init__(self, captureType: constants.Capture, **kwargs):
        """
        The depth object.
        :param kwargs: gyro=<name of gryo log file> acceleration=<name of acceleration log file>
        """
        self._accelerationLogFile = TextIOWrapper
        self._gyroLogFile = TextIOWrapper
        self._connected = False
        self._currentImage = 0
        self._camera = None
        self.log = logging.getLogger(__name__)
        self._capturing = False
        self._images = deque(maxlen=constants.DEPTH_QUEUE_LEN)

        self._serial = None
        self._pipeline = None
        self._config = None

        self._currentGryo = np.zeros(3)
        self._currentAcceleration = np.zeros(3)
        self._depth = np.empty([constants.DEPTH_MAX_HORIZONTAL, constants.DEPTH_MAX_VERTICAL])
        self._imageNumber = 0

        self._pipelineIMU = None
        self._pipelineDepthRGB = None
        self._pipelineRGB = None

        self._configIMU = None
        self._configDepthRGB = None
        self._configRGB = None

        self._initialized = False

        # The GSD -- calculated to 620.0
        self._gsd = 620.0

        # Indicates that the capturing is complete
        self._capturingComplete = False

        self._captureType = captureType

        # For the IMU capture, the names of the gyro and acceleration file must be supplied
        if captureType == constants.Capture.IMU:
            try:
                self._gyro_log = kwargs[constants.KEYWORD_FILE_GYRO]
                self._acceleration_log = kwargs[constants.KEYWORD_FILE_ACCELERATION]
            except KeyError as key:
                raise ValueError("These keywords are required: {} {}".format(constants.KEYWORD_FILE_GYRO, constants.KEYWORD_FILE_ACCELERATION))

        # Cameras are identified by the serial number -- this represents the one we want
        try:
            self._serial = str(kwargs[constants.KEYWORD_SERIAL])
        except KeyError as key:
            self.log.info("The serial number of the device is not specified by keyword: {}  Using the first device".format(constants.KEYWORD_SERIAL))

        # The configuration of the depth camera
        try:
            self._config = kwargs[constants.KEYWORD_CONFIGURATION]
        except KeyError as key:
            self.log.warning("The configuration of the device is not specified with keyword: {}. Using defaults.".format(constants.KEYWORD_CONFIGURATION))

        super().__init__(**kwargs)

        return

    @property
    def gsd(self) -> float:
        return self._gsd

    @property
    def imageNumber(self) -> int:
        return self._imageNumber

    @imageNumber.setter
    def imageNumber(self, number: int):
        self._imageNumber = number

    @property
    def gyro(self) -> np.ndarray:
        return self._currentGryo

    @property
    def acceleration(self) -> np.ndarray:
        return self._currentAcceleration

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:

        # Enable the camera and apply the config
        # rs.enable_device(self._config, self._serial)


        self._connected = True

        return self._connected


    def initializeCapture(self):

        self._initialized = True
        if self._captureType == constants.Capture.IMU:
            # Capturing both the IMU data and the depth data in one stream seems to be problematic, separate these
            self._pipelineIMU = rs.pipeline()
            self._configIMU = rs.config()

            if self._serial is not None:
                # Choose the device based on serial number
                self.log.debug("Enable IMU device with serial number: {}".format(self._serial))
                self._configIMU.enable_device(self._serial)

            # Enable the gyroscopic and accelerometer streams
            self._configIMU.enable_stream(rs.stream.accel)
            self._configIMU.enable_stream(rs.stream.gyro)

        # A combined depth/RGB capture
        elif self._captureType == constants.Capture.DEPTH_RGB:
            self.log.debug("Initialize RGB/Depth capture")
            # Enable depth stream to capture 1280x720, 6 FPS
            self._pipelineDepthRGB = rs.pipeline()
            self._configDepthRGB = rs.config()

            if self._serial is not None:
                # Choose the device based on serial number
                self.log.debug("Enable depth device with serial number: {}".format(self._serial))
                self._configDepthRGB.enable_device(self._serial)

            self._configDepthRGB.enable_stream(rs.stream.depth, constants.DEPTH_MAX_HORIZONTAL, constants.DEPTH_MAX_VERTICAL, rs.format.z16, constants.DEPTH_MAX_FRAMES)
            self._configDepthRGB.enable_stream(rs.stream.color, constants.INTEL_RGB_MAX_HORIZONTAL, constants.INTEL_RGB_MAX_VERTICAL, rs.format.rgb8, constants.INTEL_RGB_MAX_FRAMES)
            # self._configDepthRGB.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
            # self._configDepthRGB.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

        elif self._captureType == constants.Capture.RGB:
            self._pipelineRGB = rs.pipeline()
            self._configRGB = rs.config()

            if self._serial is not None:
                # Choose the device based on serial number
                self.log.debug("Enable RGB device with serial number: {}".format(self._serial))
                self._configRGB.enable_device(self._serial)

            # Enable bgr stream to capture 1920x1080, 8 FPS
            self._configRGB.enable_stream(rs.stream.color, constants.INTEL_RGB_MAX_HORIZONTAL, constants.INTEL_RGB_MAX_VERTICAL, rs.format.bgr8, constants.INTEL_RGB_MAX_FRAMES)

        else:
            self._initialized = False
            raise DepthUnknownStream("Unknown stream: {}".format(self._captureType))


        return self._initialized

    def initialize(self):
        """
        Set the camera parameters to reflect what we want them to be.
        :return:
        """

        if not self._connected:
            raise IOError("Depth Camera is not connected.")

        if self._captureType == constants.Capture.IMU:
            try:
                self._gyroLogFile = open(self._gyro_log, "w")
            except OSError:
                raise IOError("Cannot access gyro file: {}".format(self._gyro_log))
            except ResourceWarning:
                self.log.error("Unable to open {} for writing.".format(self._gyro_log))

            try:
                self._accelerationLogFile = open(self._acceleration_log, "w")
            except OSError:
                raise IOError("Cannot access acceleration file: {}".format(self._acceleration_log))
            except ResourceWarning:
                self.log.error("Unable to open {} for writing.".format(self._acceleration_log))

        self.log.debug("Depth Camera initialized")

        # try:
        #     self.state.toIdle()
        # except TransitionNotAllowed as transition:
        #     self.log.critical("Unable to transition camera to idle state")

        return

    def _gyro(self, gyro) -> np.ndarray:
        return np.asarray([gyro.x, gyro.y, gyro.z])


    def _acceleration(self, acceleration) -> np.ndarray:
        return np.asarray([acceleration.x, acceleration.y, acceleration.z])

    @property
    def captureType(self) -> constants.Capture:
        return self._captureType

    def startCapturing(self):
        self.log.debug("Beginning capture of type {}".format(self._captureType.name))

        # This is a bit of a dummy loop
        if self.state.is_failed or self.state.is_missing:
            self.log.critical("Camera is marked as failed or missing. No capture")
            self._capturing = True
            while self._capturing:
                time.sleep(60)
            self.log.debug("Null capture complete")
            return

        # The IMU capture loop
        if self._captureType == constants.Capture.IMU:
            self.log.debug("Begin grab of IMU stream")
            try:
                self._pipelineIMU.start(self._configIMU)
                self._capturing = True
                try:
                    self._state.toCapture()
                except TransitionNotAllowed as transition:
                    self.log.critical("Unable to transition camera to capturing")
                    self.log.critical(transition)
                    self._capturing = False
            except Exception as e:
                self.log.fatal("Failed to open the depth camera {} and start grabbing IMU.".format(self._serial))
                self.log.fatal("RAW: {}".format(e))
                self._capturing = False
                self.state.toFailed()

            self.log.debug("Capturing IMU data")
            while self._capturing:
                f = self._pipelineIMU.wait_for_frames()
                self._currentAcceleration = self._acceleration(f[0].as_motion_frame().get_motion_data())
                self._currentGryo = self._gyro(f[1].as_motion_frame().get_motion_data())
                #self._currentDepth = f.get_depth_frame()


                try:
                    self._gyroLogFile.write("{},{},{}\n".format(self._currentGryo[0], self._currentGryo[1], self._currentGryo[2]))
                    self._accelerationLogFile.write("{},{},{}\n".format(self._currentAcceleration[0], self._currentAcceleration[1], self._currentAcceleration[2]))
                    #epthFile = constants.FILE_DEPTH.format(1)
                    #np.save(depthFile, self._depth)
                # This is the case where an operation is stopped while data is available.  A bit of a corner-case, but one that can
                # be addressed with a semaphore.  This is certainly not a clean way to go about this.
                except ValueError as value:
                    self.log.warning("Sloppiness detected.  Tried to write to a closed file")
                except Exception as ex:
                    self.log.error("Unexpected exception hit in write of gyro and acceleration data")
                    self.log.error("Raw: {}".format(ex))
                #self.log.debug("Current gyro {}".format(self._currentGryo))

            # This is a handshake so the stop method knows we have stopped capturing
            # Ideally, this would be a state machine, but that's overkill for what we need.

            self.log.debug("IMU Capture complete")

        # The depth capture loop
        elif self._captureType == constants.Capture.DEPTH_RGB:
            self.log.debug("Begin grab of depth/RGB stream")
            try:
                self._pipelineDepthRGB.start(self._configDepthRGB)
                self._capturing = True
                try:
                    self._state.toCapture()
                except TransitionNotAllowed as transition:
                    self.log.critical("Unable to transition depth camera to capturing")
                    self.log.critical(transition)
                    self._capturing = False
            except Exception as e:
                self.log.fatal("Failed to open the depth camera {} and start grabbing depth data.".format(self._serial))
                self.log.fatal("{}".format(e))
                self.state.toFailed()
                self._capturing = False


            self.log.debug("Capturing DEPTH/RGB data")
            while self._capturing:
                try:
                    f = self._pipelineDepthRGB.wait_for_frames()
                    _currentDepth = f.get_depth_frame()
                    _currentRGB = f.get_color_frame()
                    if not _currentRGB or not _currentDepth:
                        self.log.error("Failed to capture both RGB and depth")
                        break
                    # Convert the depth data to a numpy array.
                    self._depth = np.asanyarray(_currentDepth.get_data())
                    self._RGB = np.asanyarray(_currentRGB.get_data())

                    # Put the image and the depth in the queue
                    # self.log.debug("Appending depth data to queue")
                    processed = TimestampedImage(self._RGB, self._depth, time.time())
                    self._images.append(processed)
                except Exception as e:
                    self.log.fatal("Failed to capture depth frame")
                    self.log.fatal(e)



            # This is a handshake so the stop method knows we have stopped capturing
            # Ideally, this would be a state machine, but that's overkill for what we need.

            self.log.debug("Depth Capture complete")

        # # The RGB capture loop
        # elif self._captureType == constants.Capture.RGB:
        #     self.log.debug("Begin grab of RGB stream")
        #     try:
        #         self._pipelineRGB.start(self._configRGB)
        #         self._capturing = True
        #         try:
        #             self._state.toCapture()
        #         except TransitionNotAllowed as transition:
        #             self.log.critical("Unable to transition depth camera to capturing")
        #             self.log.critical(transition)
        #             self._capturing = False
        #     except Exception as e:
        #         self.log.fatal("Failed to open the depth camera {} and start grabbing RGB data.".format(self._serial))
        #         self.log.fatal("{}".format(e))
        #         self.state.toFailed()
        #         self._capturing = False
        #
        #
        #     self.log.debug("Capturing RGB data")
        #     while self._capturing:
        #         try:
        #             f = self._pipelineRGB.wait_for_frames()
        #             _currentRGB = f.get_color_frame()
        #         except Exception as e:
        #             self.log.fatal("Failed to capture RGB frame")
        #             self.log.fatal(e)
        #             return
        #
        #         # Convert the RGB data to a numpy array.  Probably overkill
        #         self._RGB = np.asanyarray(_currentRGB.get_data())
        #
        #         # Put the image in the queue
        #         #self.log.debug("Appending depth data to queue")
        #         processed = TimestampedImage(self._RGB, None, time.time())
        #         self._images.append(processed)

                # try:
                #     depthFile = constants.FILE_DEPTH.format(1)
                #     np.save(depthFile, self._depth)!
                # except Exception as ex:
                #     self.log.error("Unexpected exception hit in write of depth data")
                #     self.log.error("Raw: {}".format(ex))

            # This is a handshake so the stop method knows we have stopped capturing
            # Ideally, this would be a state machine, but that's overkill for what we need.

            self.log.debug("Depth Capture complete")


        self.log.debug("Capture complete")
        self._capturingComplete = True

    def start(self):
        """
        Begin capturing images and store them in a queue for later retrieval.
        """

        if not self._connected:
            raise IOError("Camera is not connected.")
        # try:
        #     self.state.toCapture()
        # except TransitionNotAllowed as transition:
        #     self.log.critical("Unable to transition camera to capturing")


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

            self._state.toStop()

            # Wait for the capturing to finish
            while not self._capturingComplete:
                self.log.debug("Waiting for capture to complete")

            # Close the log files
            if self._captureType == constants.Capture.IMU:
                self._gyroLogFile.close()
                self._accelerationLogFile.close()

        return

    def disconnect(self):
        """
        Disconnected from the current camera and stop grabbing images
        """

        self.log.debug("Disconnecting from camera")
        #self.stop()

    def diagnostics(self) -> (bool, str):
        """
        Execute diagnostics on the camera.
        :return:
        Boolean result of the diagnostics and a string of the details
        """
        return True, "Camera diagnostics not provided"

    def capture(self) -> TimestampedImage:
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

        if len(self._images) == 0:
            self.log.error("Image queue is empty.")
            processed = TimestampedImage(np.empty([constants.DEPTH_MAX_HORIZONTAL, constants.DEPTH_MAX_VERTICAL]),
                                         np.empty([constants.DEPTH_MAX_HORIZONTAL, constants.DEPTH_MAX_VERTICAL]),
                                         time.time())
        else:
            self.log.debug("Serving image from queue")
            processed = self._images.popleft()
            #img = processed.image
            #timestamp = processed.timestamp
            #self.log.debug("Image captured at " + str(timestamp))
        return processed


    def getResolution(self) -> ():
        w = 0
        h = 0
        return (w, h)

    # This should be part of the calibration procedure
    def getMMPerPixel(self) -> float:
        return 0.0

    @property
    def camera(self):
        return self._camera

    # @camera.setter
    # def camera(self, openedCamera: pylon.InstantCamera):
    #     self._camera = openedCamera




    def save(self, filename: str) -> bool:
        """
        Save the camera settings
        :param filename: The file to contain the settings
        :return: True on success
        """
        #self._camera.Open()
        self.log.info("Saving camera configuration to: {}".format(filename))
        return True

    def load(self, filename: str) -> bool:
        """
        Load the camera configuration from a file. Usually, this is the .pfs file saved from the pylon viewer
        :param filename: The name of the file on disk
        :return: True on success
        """
        loaded = False

        if os.path.isfile(filename):
            self.log.info("Using saved camera configuration: {}".format(filename))

            loaded = True
        else:
            self.log.warning("Unable to find configuration file: {}.  Camera configuration unchanged".format(filename))
        return loaded

#
# The PhysicalCamera class as a utility
#
if __name__ == "__main__":

    import argparse
    import sys
    from OptionsFile import OptionsFile
    from PIL import Image
    #import matplotlib.pyplot as plt
    #import matplotlib.image

    import threading


    def getkey():
        return input("Enter \"t\" to trigger the camera or \"e\" to exit and press enter? (t/e) ")

    def takeImages(camera: CameraDepth):

        # Connect to the camera and take an image
        log.debug("Connecting to camera")
        camera.connect()
        camera.initialize()
        camera.start()

        if camera.initializeCapture():
            try:
                camera.startCapturing()
            except IOError as io:
                camera.log.error(io)
            rc = 0
        else:
            rc = -1


    parser = argparse.ArgumentParser("Depth Camera Utility")

    parser.add_argument('-s', '--single', action="store", required=True, help="Take a single picture")
    parser.add_argument('-c', '--camera', action="store", required=False, help="Serial number of target device")
    parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Log file configuration")
    parser.add_argument('-p', '--performance', action="store", required=False, default="camera.csv", help="Performance file")
    parser.add_argument('-o', '--options', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
    parser.add_argument('-i', '--info', action="store_true", required=False, default=False)
    parser.add_argument('-t', '--type', action="store", required=True, choices=[constants.Capture.DEPTH_RGB.name.lower(), constants.Capture.IMU.name.lower(), constants.Capture.RGB.name.lower()])
    arguments = parser.parse_args()

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("iron-chef")

    # Check that the format of the lines is what we expect
    #evalutionText, lines = checkLineNames(arguments.emitter)
    performance = Performance(arguments.performance)
    (performanceOK, performanceDiagnostics) = performance.initialize()

    # Parse the options file
    options = OptionsFile(arguments.options)
    if not options.load():
        print("Error encountered with option load for: {}".format(arguments.options))
        sys.exit(1)


    realsense = RealSense()

    devices = realsense.query()
    device = realsense.device()

    if arguments.type == constants.Capture.IMU.name.lower():
        camera = CameraDepth(constants.Capture.IMU, gyro=constants.PARAM_FILE_GYRO, acceleration=constants.PARAM_FILE_ACCELERATION)
    elif arguments.type == constants.Capture.DEPTH_RGB.name.lower():
        #camera = CameraDepth(constants.Capture.DEPTH, serial=arguments.camera)
        camera = CameraDepth(constants.Capture.DEPTH_RGB)
    elif arguments.type == constants.Capture.RGB.name.lower():
        camera = CameraDepth(constants.Capture.RGB)

    camera._state.toIdle()
    camera._state.toClaim()
    camera.connect()
    #camera.initializeCapture()
    # Start the thread that will begin acquiring images
    acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeImages, args=(camera,))
    acquire.start()

    # Wait for items in the queue to appear
    time.sleep(10)
    #acquire.join()


    timenow = time.time()
    logging.debug("Image needed from {}".format(timenow))
    try:
        performance.start()
        processed = camera.capture()
        performance.stopAndRecord(constants.PERF_ACQUIRE)
        if camera.captureType == constants.Capture.DEPTH_RGB:
            np.save(arguments.single + ".npy", processed.depth)
            image = Image.fromarray(processed.rgb)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(arguments.single + ".jpg")
        elif camera.captureType == constants.Capture.RGB:
            image = Image.fromarray(processed.rgb)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(arguments.single)
    except IOError as io:
        camera.log.error("Failed to capture image: {0}".format(io))
    rc = 0

    # Stop the camera, and this should stop the thread as well
    camera.log.debug("Stopping camera")
    camera.stop()
    time.sleep(2)
    camera.log.debug("Disconnecting")
    camera.disconnect()

    sys.exit(rc)

