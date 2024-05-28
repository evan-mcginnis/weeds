import datetime
import re
import sys
from enum import Enum
from time import sleep
import argparse
import sys
import os
import glob
from exif import Image
import cv2 as cv

from GPSUtilities import GPSUtilities
from OptionsFile import OptionsFile

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5 import QtGui, QtCore

from OptionsFile import OptionsFile
import logging
import logging.config
import xmpp
import constants


from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from color_ui import Ui_MainWindow

FILE_PROGRESS = "progress.ini"
ATTRIBUTE_CURRENT = "CURRENT"
SECTION_EDIT = "EDIT"

UNKNOWN_LONG = -999
NOT_SET = "-----"
UNKNOWN_STR = NOT_SET

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Images
        self._attributes = None
        self._fileNames = []

        self.setupUi(self)
        self.setMouseTracking(True)
        # Attributes
        #self._attributes = np.empty()

        #self.actionExit.triggered.connect(self.safeExit)
        self._scaledWidth = 0
        self._scaledHeight = 0
        self._scaleRatioWidth = 1
        self._scaleRatioHeight = 1

        self._imageFileName = ""

        self.image.mousePressEvent = self.getPos

    def setup(self):
        palette = self.rgbRed.palette()
        palette.color(palette.light(), QColor)
    def getPos(self, event):
        x = event.pos().x()
        y = event.pos().y()
        scaledX, scaledY = self.scaledPositionToOriginal(event.pos().x(), event.pos().y())
        print(f"Position: ({x},{y}) Scaled ({scaledX},{scaledY})")
        self.pointX.setStyleSheet("color: black; background-color: yellow")
        self.pointY.setStyleSheet("color: black; background-color: yellow")
        self.pointX.display(x)
        self.pointY.display(y)

        # RGB
        # CVT is BGR, not RGB
        self.rgbRed.display(self._imgAsRGB[y, x, 2])
        self.rgbGreen.display(self._imgAsRGB[y, x, 1])
        self.rgbBlue.display(self._imgAsRGB[y, x, 2])

        # HSV
        self.hsvHue.display(self._imgAsHSV[y, x, 0])
        self.hsvSaturation.display(self._imgAsHSV[y, x, 1])
        self.hsvValue.display(self._imgAsHSV[y, x, 2])

        # YIQ

        # CIELAB
        self.cielabL.display(self._imgAsCIELAB[y, x, 0])
        self.cielabA.display(self._imgAsCIELAB[y, x, 1])
        self.cielabB.display(self._imgAsCIELAB[y, x, 2])

        # YCBCR
        self.ycbcrY.display(self._imgAsYCBCR[y, x, 0])
        self.ycbcrCb.display(self._imgAsYCBCR[y, x, 1])
        self.ycbcrCr.display(self._imgAsYCBCR[y, x, 2])

    def mouseMoveEvent(self, event):
        scaledX, scaledY = self.scaledPositionToOriginal(event.x(), event.y())
        print(f'Mouse coords: ({event.x()}, {event.y()}) Scaled ({scaledX}, {scaledY})')

    def scaledPositionToOriginal(self, x: int, y: int) -> (int, int):
        scaledX = int(self._scaleRatioWidth * x)
        scaledY = int(self._scaleRatioHeight * y)

        return scaledX, scaledY

    def updateInformationForCurrentImage(self):
        """
        Updates the display for the current image using the EXIF data contained within it
        """

        with open(self._fileNames[self._currentFileNumber], 'rb') as rawImage:
            theImage = Image(rawImage)
            # Get the current EXIF
            if theImage.has_exif:
                exif = theImage.list_all()
                #print("EXIF Parameters: {}".format(theImage.list_all()))
            else:
                print("Warning: Image contains no EXIF data")
                exif = []


    def _convertToColorSpaces(self):
        """
        Convert the RGB image to various color spaces
        """
        self._imgAsRGB = cv.imread(self._imageFileName, cv.IMREAD_COLOR)
        self._imgAsHSV = cv.cvtColor(self._imgAsRGB.astype(np.uint8), cv.COLOR_BGR2HSV)
        self._imgAsYCBCR = cv.cvtColor(self._imgAsRGB.astype(np.uint8), cv.COLOR_BGR2YCR_CB)
        self._imgAsCIELAB = cv.cvtColor(self._imgAsRGB.astype(np.uint8), cv.COLOR_BGR2Lab)

    def loadImage(self, imageFileName: str):
        self._imageFileName = imageFileName
        self._convertToColorSpaces()
        self._displayImage(imageFileName)

    def _displayImage(self, imageName):
        """
        Display the current image.
        :param imageName:
        """
        width = self.image.width()
        height = self.image.height()
        print("Image area is {}x{}".format(width, height))
        #self.image.setGeometry()
        pixmap = QPixmap(imageName)
        originalWidth = pixmap.width()
        originalHeight = pixmap.height()
        print(f"Original image is {pixmap.width()} x {pixmap.height()}")
        #scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio)
        #scaled = pixmap.scaled(self.image.size(), Qt.KeepAspectRatio)
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaledWidth = scaled.width()
        self._scaledHeight = scaled.height()
        self._scaleRatioWidth = originalWidth / self._scaledWidth
        self._scaleRatioHeight = originalHeight / self._scaledHeight
        print(f"Target image scaled is ({scaled.width()} x {scaled.height()}) scaling factors: {self._scaleRatioWidth} x {self._scaleRatioHeight}")
        self.image.setPixmap(scaled)
        self.image.resize(self._scaledWidth, self._scaledHeight)
        #self.image.setMinimumSize(self._scaledWidth, self._scaledHeight)
        self.image.setFixedSize(self._scaledWidth, self._scaledHeight)


    def setup(self):
        pass

    def safeExit(self):
        app.quit()

    def exitHandler(self):
        pass







parser = argparse.ArgumentParser("Image Color Information")

parser.add_argument('-i', '--input', action="store", required=True,  help="Input Image")

arguments = parser.parse_args()

#logging.config.fileConfig(arguments.log)
log = logging.getLogger("review")

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()

window.setWindowTitle("University of Arizona")


app.aboutToQuit.connect(window.exitHandler)


if arguments.input is not None:
    window.loadImage(arguments.input)

window.setup()
window.show()

sys.exit(app.exec_())
