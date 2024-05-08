#
# I N T E L  4 3 5  C A P T U R E
#

import argparse
import sys
import threading
import time
import os
import glob
from exif import Image

# from GPSUtilities import GPSUtilities
# from OptionsFile import OptionsFile

import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QDialogButtonBox
# from PyQt5 import QtGui, QtCore

from OptionsFile import OptionsFile
import logging
import logging.config
import constants

from playsound import playsound
from PIL import Image, ImageQt

from PyQt5 import Qt
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *

from UI435 import Ui_MainWindow

from CameraDepth import CameraDepth

import os

OUTPUT = "output"
DEFAULT_OUTPUT = "."
DEPTH_JPG = ".depth.jpg"
RGB_JPG = ".rgb.jpg"
IR_JPG = ".ir.jpg"

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, camera: CameraDepth, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self._currentImageNumber = 0
        self._camera = camera

        self.takePictureButton.clicked.connect(self.takePicture)

        self._outputDirectory = DEFAULT_OUTPUT
        self._currentRGBFileName = None
        self._currentDepthFileName = None
        self._currentIRFileName = None

    @property
    def filenameRGB(self) -> str:
        return self._currentRGBFileName

    @property
    def filenameDepth(self) -> str:
        return self._currentDepthFileName

    @property
    def filenameIR(self) -> str:
        return self._currentIRFileName

    @property
    def outputDirectory(self) -> str:
        return self._outputDirectory

    @outputDirectory.setter
    def outputDirectory(self, theDirectory: str):
        self._outputDirectory = theDirectory

    def initialize(self):

        width = self.takePictureButton.sizeHint().width() + 20
        self.takePictureButton.setFixedSize(width, width)

        # myPixmap = QtGui.QPixmap(_fromUtf8('image.jpg'))
        # myScaledPixmap = myPixmap.scaled(self.label.size(), Qt.KeepAspectRatio)
        # self.label.setPixmap(myScaledPixmap)

    def incrementPictureCount(self):

        self._currentImageNumber += 1
        self.imageCount.display(self._currentImageNumber)

    def displayPicture(self, captureType: constants.Capture):

        if captureType == constants.Capture.RGB:
            if self._currentRGBFileName is None:
                return
            else:
                pix = QPixmap(self._currentRGBFileName)
                self.rgbImage.setPixmap(pix)
        if captureType == constants.Capture.DEPTH_DEPTH:
            depth = np.load(self._currentDepthFileName)
            qpixmap = QPixmap(DEPTH_JPG)
            self.depthImage.setPixmap(qpixmap)
        if captureType == constants.Capture.IR:
            qpixmap = QPixmap(IR_JPG)
            self.depthImage.setPixmap(qpixmap)

    def displayDistance(self, distance: float):

        self.distanceToGround.display(distance)

    def takePicture(self):
        print(f"Take picture: {self._currentImageNumber}")

        # Grab the image
        processedImage = self._camera.capture()

        # Play a sound when a picture is taken -- there is a bit of a delay, so this is really not great
        playsound('shutter.mp3')

        rgbImage = processedImage.rgb
        depthImage = processedImage.depth
        irImage = processedImage.ir

        log.debug("Saving image")
        # DEPTH
        if depthImage is not None:
            depthFileName = f"{constants.PREFIX_IMAGE}{self._currentImageNumber:03}{constants.EXTENSION_NPY}"
            depthFQN = os.path.join(self._outputDirectory, depthFileName)
            np.save(depthFQN, depthImage)
            # Save a JPG version of the depth data just so it can be shown
            plt.imsave(DEPTH_JPG, depthImage, vmin=250, vmax=340)

            self._currentDepthFileName = depthFQN
            self.displayPicture(constants.Capture.DEPTH_DEPTH)
            self.displayDistance(self._camera.agl)
        else:
            self._currentDepthFileName = None

        # RGB
        if rgbImage is not None:

            image = Image.fromarray(rgbImage)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            rgbFileName = f"{constants.PREFIX_IMAGE}{self._currentImageNumber:03}{constants.EXTENSION_IMAGE}"
            rgbFQN = os.path.join(self._outputDirectory, rgbFileName)
            image.save(rgbFQN)
            self._currentRGBFileName = rgbFQN
            self.displayPicture(constants.Capture.RGB)

        # IR
        if irImage is not None:
            # Save a numpy version and a JPG
            irFileName = f"{constants.PREFIX_IMAGE}{self._currentImageNumber:03}-ir{constants.EXTENSION_NPY}"
            irFQN = os.path.join(self._outputDirectory, irFileName)
            np.save(irFQN, depthImage)

            image = Image.fromarray(irImage)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            irFileName = f"{constants.PREFIX_IMAGE}{self._currentImageNumber:03}-ir{constants.EXTENSION_IMAGE}"
            irFQN = os.path.join(self._outputDirectory, irFileName)
            image.save(irFQN)
            image.save(IR_JPG)
            self._currentIRFileName = irFQN
            self.displayPicture(constants.Capture.IR)

        self.incrementPictureCount()



    def exitHandler(self):
        # Stop the camera, and this should stop the thread as well
        print("In exit handler")
        self._camera.log.debug("Stopping camera")
        self._camera.stop()
        time.sleep(2)
        self._camera.log.debug("Disconnecting")
        self._camera.disconnect()

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

def updateStatus(camera: CameraDepth, window: MainWindow):

    while not camera.initialized:
        print("Waiting for camera initialization")
        time.sleep(5)
    while not camera.capturing:
        print("Wait for capturing to begin")
        time.sleep(.25)
    while camera.capturing:
        # Show the distance to the target
        window.displayDistance(camera.agl)
        if camera.captureType == constants.Capture.DEPTH_RGB:
            if camera.currentRGB is not None:
                image = Image.fromarray(camera.currentRGB)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.save(RGB_JPG)
                window._currentRGBFileName = RGB_JPG
                window.displayPicture(constants.Capture.RGB)
            else:
                log.error("Current RGB is not defined")

        time.sleep(3)
    log.debug("Capture stopped")




parser = argparse.ArgumentParser("Intel 435 Capture")

parser.add_argument('-o', '--output', action="store", required=True, help="Output directory ")
parser.add_argument('-l', '--logging', action="store", required=True, help="Log file configuration")
parser.add_argument('-t', '--type', action="store", required=False, default=constants.Capture.DEPTH_RGB.name,
                    choices=[constants.Capture.DEPTH_RGB.name.lower(), constants.Capture.IR.name.lower()], help="Type of capture")
parser.add_argument('-c', '--config', action="store", required=False, help="Configuration file for camera")
arguments = parser.parse_args()

logging.config.fileConfig(arguments.logging)
log = logging.getLogger("capture")

camera = CameraDepth(constants.Capture[arguments.type.upper()], config=arguments.config)
camera._state.toIdle()
camera._state.toClaim()
camera.connect()
# camera.initializeCapture()
# Start the thread that will begin acquiring images
acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeImages, args=(camera,))
acquire.start()

time.sleep(2)
if not acquire.is_alive():
    print("Error encountered with camera")
    sys.exit(-1)

app = QtWidgets.QApplication(sys.argv)

window = MainWindow(camera)
window.outputDirectory = arguments.output
window.initialize()

app.aboutToQuit.connect(window.exitHandler)

update = threading.Thread(name=constants.THREAD_NAME_UPDATE, target=updateStatus, args=(camera, window))
update.start()


#window.setStatus()

window.show()
sys.exit(app.exec_())

