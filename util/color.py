import datetime
import re
import sys
import time
from enum import Enum
from time import sleep
import argparse
import sys
import os
import glob
from exif import Image
import cv2 as cv
from skimage.color import rgb2yiq

from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger

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

class Reading(Enum):
    VEGETATION = 0
    GROUND = 1

MAX_ZOOM = 5

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    spaces = ["BGR", "RGB", "YIQ", "YUV", "HSI", "HSV", "YCBCR", "CIELAB"]
    def __init__(self, *args, **kwargs):

        # Create columns for dataframe like hsv_1, hsv_2, hsv_3
        columns = []

        if "colorspace" in kwargs:
            columns.append("type")
            columns.append(kwargs["colorspace"] + "_0")
            columns.append(kwargs["colorspace"] + "_1")
            columns.append(kwargs["colorspace"] + "_2")
            self._colorSpace = kwargs["colorspace"]
            _ = kwargs.pop("colorspace")
        else:
            columns.append("type")
            columns.append("colorspace" + "_0")
            columns.append("colorspace" + "_1")
            columns.append("colorspace" + "_2")
            self._colorSpace = ""

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
        self._logger = None
        self._manipulated: ImageManipulation
        self._manipulated = None

        self._mode = "color"
        self._n = 0
        self._vegetationPoints = 0
        self._groundPoints = 0

        self._samples = pd.DataFrame(columns=columns)

        self._output = None

        self._imgAsBGR = None
        self._imgAsRGB = None
        self._imgAsYIQ = None
        self._imgAsYUV = None
        self._imgAsHSI = None
        self._imgAsHSV = None
        self._imgAsYCBCR = None
        self._imgAsCIELAB = None

        self._currentSampleSource = None
        self._currentLocation = None

        self._files = []
        self._filesToProcess = 0
        self._currentImageName = None

        self._currentZoom = 0
        self._scale = 1

        self._pixmap = None

        # All the bands processed
        self._allBands = {
            "BGR": {"readings": self._imgAsBGR},
            "RGB": {"readings": self._imgAsRGB},
            "YIQ": {"readings": self._imgAsYIQ},
            "YUV": {"readings": self._imgAsYUV},
            "HSI": {"readings": self._imgAsHSI},
            "HSV": {"readings": self._imgAsHSV},
            "YCBCR": {"readings": self._imgAsYCBCR},
            "CIELAB": {"readings": self._imgAsCIELAB}
        }

        # Dataframe columns
        columns = ["type"]
        types = {"type": int}

        for bandName, bandData in self._allBands.items():
            columns.append(f"{bandName}-band-0")
            types[f"{bandName}-band-0"] = float
            columns.append(f"{bandName}-band-1")
            types[f"{bandName}-band-1"] = float
            columns.append(f"{bandName}-band-2")
            types[f"{bandName}-band-2"] = float

        # Dataframe creation
        self._samples = pd.DataFrame(columns=columns)
        self._samples = self._samples.astype(dtype=types)

    @property
    def files(self) -> []:
        return self._files

    @files.setter
    def files(self, theFiles: []):
        self._files = theFiles
        self._filesToProcess = len(theFiles)

    @property
    def location(self) -> Reading:
        return self._currentLocation

    @location.setter
    def location(self, theLocation: Reading):
        self._currentLocation = theLocation
    #
    # @property
    # def source(self) -> str:
    #     return self._currentSampleSource
    #
    # @source.setter
    # def source(self, theSource: str):
    #     """
    #     The source color space used
    #     :param theSource:
    #     """
    #     assert theSource in self.spaces
    #     self._currentSampleSource = theSource

    @property
    def output(self) -> str:
        return self._output

    @output.setter
    def output(self, theOutput: str):
        self._output = theOutput

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, theMode: str):
        self._mode = theMode

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, theN):
        self._n = theN
        self._vegetationPoints = theN
        self._groundPoints = theN


    def scaleCurrentImage(self, currentX, currentY: int):
        subset = (MAX_ZOOM - self._currentZoom) / MAX_ZOOM

        # The new position within the scaled image
        scaledX = int(subset * currentX)
        scaledY = int(subset * currentY)
        newScaledWidth = int(subset * self._scaledWidth)
        newScaledHeight = int(subset * self._scaledHeight)

        print(f"Sample {subset} of image. New size {newScaledWidth} x {newScaledHeight}. Scaled position: ({scaledX},{scaledY})")

        #self._scaledImage = self._i
        # Compute what is to be sampled
        # To maintain position, the width is from the Y position to the edges
        #widthStarts = newScaledWidth - scaledX

        size = self._pixmap.size()
        scaled_pixmap = self._pixmap.scaled(newScaledWidth, newScaledHeight, Qt.KeepAspectRatio)
        self.image.setPixmap(scaled_pixmap)

        self._scale *= 2

        size = self._pixmap.size()

        print(f"Current size is {size}")
        scaled_pixmap = self._pixmap.scaledToHeight(newScaledHeight)

        self.image.setPixmap(scaled_pixmap)

    def wheelEvent(self, event: QWheelEvent):
        """
        React to the scroll of the middle wheel.
        :param event:
        """
        log.debug(f"Mouse scroll detected {event.angleDelta().y()}. Current Zoom: {self._currentZoom}")

        # Zoom in
        if event.angleDelta().y() > 0:
            if self._currentZoom + 1 in range(0, MAX_ZOOM + 1):
                self._currentZoom += 1
                self.scaleCurrentImage(event.x(), event.y())
        # Zoom out
        else:
            if self._currentZoom - 1 in range(0, MAX_ZOOM + 1):
                self._currentZoom -= 1
                self.scaleCurrentImage(event.x(), event.y())



    def checkpoint(self):
        """
        Write out the data if the output file is defined
        """
        if self._output is not None:
            # If this isn't converted explicitly, it writes the type out as a float instead, so 1.0 instead of 1
            self._samples['type'] = self._samples['type'].astype(int)
            self._samples.to_csv(self._output, index=False)

    def finished(self):
        """
        Write out the data if the output file is defined
        """
        self.checkpoint()

    def addReading(self, source: Reading, x: int, y: int):
        """
        Add the reading of the point to the sample list
        :param y:
        :param x:
        :param source: Ground or Vegetation
        """
        entry = [int(source.value)]
        for bandName, bandData in self._allBands.items():
            readings = bandData["readings"]
            entry.append(readings[y, x, 0])
            entry.append(readings[y, x, 1])
            entry.append(readings[y, x, 2])

        self._samples.loc[len(self._samples)] = entry


    def setup(self):
        # palette = self.rgbRed.palette()
        # palette.color(palette.light(), QColor)
        self._logger = ImageLogger()


    def getPos(self, event):
        band = None
        x = event.pos().x()
        y = event.pos().y()
        scaledX, scaledY = self.scaledPositionToOriginal(event.pos().x(), event.pos().y())
        print(f"Position: ({x},{y}) Scaled ({scaledX},{scaledY})")
        self.pointX.setStyleSheet("color: black; background-color: yellow")
        self.pointY.setStyleSheet("color: black; background-color: yellow")
        self.pointX.display(x)
        self.pointY.display(y)

        # RGB
        self.rgbRed.display(self._imgAsRGB[scaledY, scaledX, 0])
        self.rgbGreen.display(self._imgAsRGB[scaledY, scaledX, 1])
        self.rgbBlue.display(self._imgAsRGB[scaledY, scaledX, 2])

        # HSV
        self.hsvHue.display(self._imgAsHSV[scaledY, scaledX, 0])
        self.hsvSaturation.display(self._imgAsHSV[scaledY, scaledX, 1])
        self.hsvValue.display(self._imgAsHSV[scaledY, scaledX, 2])

        # YIQ
        self.yiqY.display(self._imgAsYIQ[scaledY, scaledX, 0])
        self.yiqI.display(self._imgAsYIQ[scaledY, scaledX, 1])
        self.yiqQ.display(self._imgAsYIQ[scaledY, scaledX, 2])

        # CIELAB
        self.cielabL.display(self._imgAsCIELAB[scaledY, scaledX, 0])
        self.cielabA.display(self._imgAsCIELAB[scaledY, scaledX, 1])
        self.cielabB.display(self._imgAsCIELAB[scaledY, scaledX, 2])

        # YCBCR
        self.ycbcrY.display(self._imgAsYCBCR[scaledY, scaledX, 0])
        self.ycbcrCb.display(self._imgAsYCBCR[scaledY, scaledX, 1])
        self.ycbcrCr.display(self._imgAsYCBCR[scaledY, scaledX, 2])

        # YUV
        self.yuvY.display(self._imgAsYUV[scaledY, scaledX, 0])
        self.yuvU.display(self._imgAsYUV[scaledY, scaledX, 1])
        self.yuvV.display(self._imgAsYUV[scaledY, scaledX, 2])

        # HSI
        self.hsiH.display(self._imgAsHSI[scaledY, scaledX, 0])
        self.hsiS.display(self._imgAsHSI[scaledY, scaledX, 1])
        self.hsiI.display(self._imgAsHSI[scaledY, scaledX, 2])

        self.displayMode(1)

        # If all the vegetation and ground points have been captured, load the next image
        if self._groundPoints == 0 and self._vegetationPoints == 0:
            self.displayLoadingMessage()
            self.checkpoint()
            self.loadNextImage()


        # Get the band information for the colorspace of interest
        # This is only really needed for the sample mode
        # bandInfo = self._allBands[self._currentSampleSource]
        # readings = bandInfo["readings"]
        #
        # band0 = readings[scaledY, scaledX, 0]
        # band1 = readings[scaledY, scaledX, 1]
        # band2 = readings[scaledY, scaledX, 2]

        self.addReading(self._currentLocation, scaledX, scaledY)

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
        # OpenCV treats images as BGR, not RGB
        self._imgAsRGB = cv.cvtColor(self._imgAsBGR.astype(np.uint8), cv.COLOR_BGR2RGB)

        self._imgAsHSV = cv.cvtColor(self._imgAsBGR.astype(np.uint8), cv.COLOR_BGR2HSV)
        self._imgAsYCBCR = cv.cvtColor(self._imgAsBGR.astype(np.uint8), cv.COLOR_BGR2YCR_CB)
        self._imgAsCIELAB = cv.cvtColor(self._imgAsBGR.astype(np.uint8), cv.COLOR_BGR2Lab)
        self._imgAsYUV = cv.cvtColor(self._imgAsBGR.astype(np.uint8), cv.COLOR_BGR2YUV)

        # This method takes RGB as input, not BGR
        self._imgAsYIQ = rgb2yiq(self._imgAsRGB)

        self._imgAsHSI = self._manipulated.toHSI()

        self._allBands = {
            "BGR": {"readings": self._imgAsBGR},
            "RGB": {"readings": self._imgAsRGB},
            "YIQ": {"readings": self._imgAsYIQ},
            "YUV": {"readings": self._imgAsYUV},
            "HSI": { "readings": self._imgAsHSI},
            "HSV": {"readings": self._imgAsHSV},
            "YCBCR": {"readings": self._imgAsYCBCR},
            "CIELAB": {"readings": self._imgAsCIELAB}
        }


    def displayLoadingMessage(self):
        self.pointX.display(0)
        self.pointY.display(0)

        # RGB
        self.rgbRed.display(0)
        self.rgbGreen.display(0)
        self.rgbBlue.display(0)

        # HSV
        self.hsvHue.display(0)
        self.hsvSaturation.display(0)
        self.hsvValue.display(0)

        # YIQ
        self.yiqY.display(0)
        self.yiqI.display(0)
        self.yiqQ.display(0)

        # CIELAB
        self.cielabL.display(0)
        self.cielabA.display(0)
        self.cielabB.display(0)

        # YCBCR
        self.ycbcrY.display(0)
        self.ycbcrCb.display(0)
        self.ycbcrCr.display(0)

        # YUV
        self.yuvY.display(0)
        self.yuvU.display(0)
        self.yuvV.display(0)

        # HSI
        self.hsiH.display(0)
        self.hsiS.display(0)
        self.hsiI.display(0)

        self._displayImage("loading.jpg")
        self.operatingInstructions.setText("Processing")

        # Curious -- the loading image is not displayed until a redraw is forced
        qApp.processEvents()

    def loadNextImage(self) -> bool:

        loaded = False
        if len(self._files) >= 1:
            self._imageFileName = self._files.pop()
            self._imgAsBGR = cv.imread(self._imageFileName, cv.IMREAD_COLOR)
            self._manipulated = ImageManipulation(self._imgAsBGR, 0, self._logger)
            self._convertToColorSpaces()
            self._displayImage(self._imageFileName)

            # Reset the number of points to capture
            self._groundPoints = self._n
            self._vegetationPoints = self._n
            self.displayMode(0)
            self.displayCount()

            loaded = True

        return loaded

    def displayCount(self):
        countText = f"Image {self._filesToProcess - len(self._files)} of {self._filesToProcess}"
        self.imageCount.setText(countText)
    def displayMode(self, decrementBy: int):
        if self.mode == "color":
            operatingText = "Color selection"
        elif self.mode == "sample":
            if self._currentLocation == Reading.GROUND:
                self._groundPoints -= decrementBy
            elif self._currentLocation == Reading.VEGETATION:
                self._vegetationPoints -= decrementBy

            if self._groundPoints > 0:
                operatingText = f"Click on {self._groundPoints} ground points"
                self._currentLocation = Reading.GROUND
            elif self._vegetationPoints > 0:
                operatingText = f"Click on {self._vegetationPoints} vegetation points"
                self._currentLocation = Reading.VEGETATION
            else:
                operatingText = f"All points sampled"
        else:
            operatingText = f"Unknown mode: {self.mode}"

        self.operatingInstructions.setText(operatingText)

    def _displayImage(self, imageName):
        """
        Display the current image.
        :param imageName:
        """
        width = self.image.width()
        height = self.image.height()
        print(f"Image area is {width}x{height}")
        #self.image.setGeometry()
        self._pixmap = QPixmap(imageName)
        originalWidth = self._pixmap.width()
        originalHeight = self._pixmap.height()
        print(f"Original image of {imageName} is {self._pixmap.width()} x {self._pixmap.height()}")
        #scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio)
        #scaled = pixmap.scaled(self.image.size(), Qt.KeepAspectRatio)
        scaled = self._pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaledWidth = scaled.width()
        self._scaledHeight = scaled.height()
        self._scaleRatioWidth = originalWidth / self._scaledWidth
        self._scaleRatioHeight = originalHeight / self._scaledHeight
        print(f"Target image scaled is ({scaled.width()} x {scaled.height()}) scaling factors: {self._scaleRatioWidth} x {self._scaleRatioHeight}")
        self.image.setPixmap(scaled)
        self.image.resize(self._scaledWidth, self._scaledHeight)
        #self.image.setMinimumSize(self._scaledWidth, self._scaledHeight)
        self.image.setFixedSize(self._scaledWidth, self._scaledHeight)




    def safeExit(self):
        self.finished()
        app.quit()

    def exitHandler(self):
        self.finished()
        pass







parser = argparse.ArgumentParser("Image Color Information")

parser.add_argument('-i', '--input', action="store", required=True,  help="Input Image")
parser.add_argument('-m', '--mode', action="store", required=False, default="color", choices=["color", "sample"], help="Operating Mode")
parser.add_argument('-n', '--n', action="store", required=False, type=int, default=20, help="Number of points")
parser.add_argument('-o', '--output', action="store", required=False, help="Output file name")
parser.add_argument('-c', '--color', action="store", required=False, choices=MainWindow.spaces, help="Take samples from this color space")

arguments = parser.parse_args()

#logging.config.fileConfig(arguments.log)
log = logging.getLogger("review")

app = QtWidgets.QApplication(sys.argv)

if arguments.color is None:
    window = MainWindow()
else:
    window = MainWindow(colorspace=arguments.color)

window.setWindowTitle("Color Selection")
window.mode = arguments.mode
window.n = arguments.n

app.aboutToQuit.connect(window.exitHandler)

if os.path.isdir(arguments.input):
    files = glob.glob(arguments.input + "/" + "*.jpg")
elif os.path.isfile(arguments.input):
    files = [arguments.input]
else:
    print(f"Unable to access {arguments.input} as file or directory")
    exit(-1)

window.files = files

window.displayLoadingMessage()
window.loadNextImage()

if arguments.output is not None:
    if os.path.isfile(arguments.output):
        print(f"Output file {arguments.output} exists. Will not overwrite.")
        exit(-1)
    window.output = arguments.output

if arguments.mode == "sample":
    if arguments.output is None:
        print(f"Output file must be specified for sample mode")
        exit(-1)
    else:
        window.output = arguments.output
    # if arguments.color is None:
    #     print(f"Colorspace must be specified for sample mode")
    #     exit(-1)
    # else:
    #     window.source = arguments.color

window.displayMode(0)

window.setup()
window.show()

sys.exit(app.exec_())
