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

import numpy as np
# import pandas as pd
# from PyQt5.QtCore import Qt
# from PyQt5.QtWidgets import QDialogButtonBox
# from PyQt5 import QtGui, QtCore

from OptionsFile import OptionsFile
import logging
import logging.config
import constants


from PIL import Image, ImageQt

from PyQt5 import QtWidgets
# from PyQt5.QtGui import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *

from UI435 import Ui_MainWindow

from CameraDepth import CameraDepth

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, camera: CameraDepth, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self._currentImageNumber = 0
        self._camera = camera

        self.takePictureButton.clicked.connect(self.takePicture)

    def takePicture(self):
        self._currentImageNumber += 1
        print(f"Take picture: {self._currentImageNumber}")

        # Grab the image
        processedImage = self._camera.capture()
        rgbImage = processedImage.rgb
        depthImage = processedImage.depth

        log.debug("Saving image")
        np.save("depth" + constants.DELIMETER + str(self._currentImageNumber) + constants.EXTENSION_NPY, depthImage)

        image = Image.fromarray(rgbImage)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save("rgb" + constants.DELIMETER + str(self._currentImageNumber) + constants.EXTENSION_IMAGE)

        # convert data to QImage using PIL

        # img = Image.fromarray(rgbImage, mode='RGB')
        # qt_img = ImageQt.ImageQt(img)
        # self.depthImage.setPixmap(QtGui.QPixmap.fromImage(qt_img))



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



parser = argparse.ArgumentParser("Intel 435 Capture")

parser.add_argument('-o', '--output', action="store", required=True, help="Output directory ")
parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Log file configuration")
arguments = parser.parse_args()

logging.config.fileConfig(arguments.logging)
log = logging.getLogger("capture")

camera = CameraDepth(constants.Capture.DEPTH_RGB)
camera._state.toIdle()
camera._state.toClaim()
camera.connect()
# camera.initializeCapture()
# Start the thread that will begin acquiring images
acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeImages, args=(camera,))
acquire.start()

app = QtWidgets.QApplication(sys.argv)

window = MainWindow(camera)

app.aboutToQuit.connect(window.exitHandler)


#window.setStatus()

window.show()
sys.exit(app.exec_())

