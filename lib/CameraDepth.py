#
# C A M E R A F I L E
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
#import cv2 as cv
from abc import ABC, abstractmethod

from statemachine.exceptions import TransitionNotAllowed

import constants
from Performance import Performance

from Camera import Camera




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

import pyrealsense2 as rs
import numpy as np


class CameraDepth(Camera):

    def __init__(self, **kwargs):
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
        self._images = deque(maxlen=constants.IMAGE_QUEUE_LEN)

        self._pipeline = None
        self._config = None

        self._currentGryo = np.zeros(3)
        self._currentAcceleration = np.zeros(3)

        # Indicates that the capturing is complete
        self._capturingComplete = False

        try:
            self._gyro_log = kwargs[constants.KEYWORD_FILE_GYRO]
            self._acceleration_log = kwargs[constants.KEYWORD_FILE_ACCELERATION]
        except KeyError as key:
            raise ValueError("These keywords are required: {} {}".format(constants.KEYWORD_FILE_GYRO, constants.KEYWORD_FILE_ACCELERATION))

        super().__init__(**kwargs)

        return

    @property
    def gyro(self):
        return self._currentGryo

    @property
    def acceleration(self):
        return self._currentAcceleration

    def connect(self) -> bool:

        self._connected = True

        return self._connected


    def initializeCapture(self):

        self._pipeline = rs.pipeline()
        self._config = rs.config()

        # Enable the gyroscopic and accelerometer streams
        self._config.enable_stream(rs.stream.accel)
        self._config.enable_stream(rs.stream.gyro)

        self._initialized = True

        return(self._initialized)

    def initialize(self):
        """
        Set the camera parameters to reflect what we want them to be.
        :return:
        """

        if not self._connected:
            raise IOError("Depth Camera is not connected.")

        try:
            self._gyroLogFile = open(self._gyro_log, "w")
        except OSError:
            raise IOError("Cannot access gyro file: {}".format(self._gyro_log))

        try:
            self._accelerationLogFile = open(self._acceleration_log, "w")
        except OSError:
            raise IOError("Cannot access acceleration file: {}".format(self._acceleration_log))

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

    def startCapturing(self):

        self.log.debug("Begin grab of IMU stream")
        try:
            self._pipeline.start(self._config)
            self._capturing = True
            try:
                self.state.toCapture()
            except TransitionNotAllowed as transition:
                self.log.critical("Unable to transition camera to capturing")
                self.log.critical(transition)
                self._capturing = False
        except Exception as e:
            self.log.fatal("Failed to open the depth camera and start grabbing.")
            self.log.fatal("{}".format(e))
            self._capturing = False

        self.log.debug("Capturing IMU data")
        while self._capturing:
            f = self._pipeline.wait_for_frames()
            self._currentAcceleration = self._acceleration(f[0].as_motion_frame().get_motion_data())
            self._currentGryo = self._gyro(f[1].as_motion_frame().get_motion_data())
            try:
                self._gyroLogFile.write("{},{},{}\n".format(self._currentGryo[0], self._currentGryo[1], self._currentGryo[2]))
                self._accelerationLogFile.write("{},{},{}\n".format(self._currentAcceleration[0], self._currentAcceleration[1], self._currentAcceleration[2]))
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

            self.state.toStop()

        # Wait for the capturing to finish
        while not self._capturingComplete:
            self.log.debug("Waiting for capture to complete")

        # Close the log files
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
    parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Log file configuration")
    parser.add_argument('-p', '--performance', action="store", required=False, default="camera.csv", help="Performance file")
    parser.add_argument('-o', '--options', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
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

    #cameraIP = options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_CAMERA_IP)
    camera = CameraDepth(gyro=constants.PARAM_FILE_GYRO, acceleration=constants.PARAM_FILE_ACCELERATION)



    # Start the thread that will begin acquiring images
    acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeImages, args=(camera,))
    acquire.start()

    # Wait for items in the queue to appear
    time.sleep(10)
    #acquire.join()


    # timenow = time.time()
    # logging.debug("Image needed from {}".format(timenow))
    # try:
    #     performance.start()
    #     img = camera.capture()
    #     performance.stopAndRecord(constants.PERF_ACQUIRE)
    #     cv.imwrite(arguments.single, img)
    # except IOError as io:
    #     camera.log.error("Failed to capture image: {0}".format(io))
    rc = 0

    # Stop the camera, and this should stop the thread as well
    camera.stop()
    time.sleep(2)
    camera.disconnect()

    sys.exit(rc)

