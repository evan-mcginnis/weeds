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
from enum import Enum

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

from review_ui import Ui_MainWindow

class VegetationType(Enum):
    CROP = 0
    WEED = 1
    MISTAKE = 2

    def __str__(self):
        return str(self.value)

FILE_PROGRESS = "progress.ini"
ATTRIBUTE_CURRENT = "CURRENT"
SECTION_EDIT = "EDIT"

UNKNOWN_LONG = -999
NOT_SET = "-----"
UNKNOWN_STR = NOT_SET

MAX_ZOOM = 5

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Images
        self._attributes = None
        self._fileNames = []
        self._currentFileNumber = 0
        self._maxFileNumber = 0
        self._currentFilename = ""

        self._processedFileName = None
        self._lastFileReviewed = 0
        # Attributes
        #self._attributes = np.empty()

        self._sourceDirectory = "."

        self.setupUi(self)

        self._correctType = False

        # Wire up the buttons
        self.button_next.clicked.connect(self.nextImage)
        self.button_previous.clicked.connect(self.previousImage)
        self.weed.clicked.connect(lambda: self.setBlobTypes(VegetationType.WEED))
        self.crop.clicked.connect(lambda: self.setBlobTypes(VegetationType.CROP))
        self.mistake.clicked.connect(lambda: self.setBlobTypes(VegetationType.MISTAKE))

        # Set the initial button states
        self.button_next.setEnabled(True)
        self.button_previous.setEnabled(False)

        self.actionExit.triggered.connect(self.safeExit)
        self.actionSave.triggered.connect(self.saveProgress)
        self.actionMark_for_review.triggered.connect(self.markCurrentImageForReview)

        self._scale = 1
        self._scaledWidth = 0
        self._scaledHeight = 0
        self._scaleRatioWidth = 1
        self._scaleRatioHeight = 1
        self._currentZoom = 1

        # Set up the shortcuts n for next, p for previous
        shortcut = QKeySequence(Qt.Key_N)
        self._shortcutNext = QShortcut(shortcut, self)
        self._shortcutNext.activated.connect(self.nextImage)

        shortcut = QKeySequence(Qt.Key_P)
        self._shortcutPrevious = QShortcut(shortcut, self)
        self._shortcutPrevious.activated.connect(self.previousImage)

        # Shortcuts for the buttons for fast assignment to all
        shortcut = QKeySequence(Qt.Key_W)
        self._shortcutWeed = QShortcut(shortcut, self)
        self._shortcutWeed.activated.connect(lambda: self.setBlobTypes(VegetationType.WEED))
        shortcut = QKeySequence(Qt.Key_C)
        self._shortcutCrop = QShortcut(shortcut, self)
        self._shortcutCrop.activated.connect(lambda: self.setBlobTypes(VegetationType.CROP))
        shortcut = QKeySequence(Qt.Key_M)
        self._shortcutMistake = QShortcut(shortcut, self)
        self._shortcutMistake.activated.connect(lambda: self.setBlobTypes(VegetationType.MISTAKE))

    @property
    def correctType(self) -> bool:
        return self._correctType

    @correctType.setter
    def correctType(self, correct: bool):
        self._correctType = correct

    @property
    def processed(self) -> str:
        return self._processedFileName

    @processed.setter
    def processed(self, filename: str):
        self._processedFileName = filename

    @property
    def lastFileReviewed(self):
        return self._lastFileReviewed

    @property
    def currentFileNumber(self):
        return self._currentFileNumber

    @currentFileNumber.setter
    def currentFileNumber(self, i: int):
        self._currentFileNumber = i

    def markCurrentImageForReview(self):
        pass

    def toggleView(self):
        pass

    def selectImageSet(self):
        pass

    def confirmResumeReview(self, text) -> bool:
        """
        Confirm the operation with a yes or no
        :param text: The text displayed to the user
        :return: True if operation is confirmed
        """
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(20)
        qm = QMessageBox()
        qm.setText(text)
        qm.setFont(font)
        qm.setWindowTitle("Resume")
        qm.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirmed = False
        try:
            qm.exec_()
            if qm.standardButton(qm.clickedButton()) == QMessageBox.Yes:
                confirmed = True
            else:
                confirmed = False
        except Exception as e:
            log.fatal(e)

        return confirmed

    def saveProgress(self):
        """
        Save the current image number to the INI file
        """
        # Write out a dataframe, including only crop and weed
        finalDF = self._attributes[(self._attributes.actual == constants.TYPE_WEED) | (self._attributes.actual == constants.TYPE_CROP)]
        # If we are just correcting the type, drop the ACTUAL column
        if self._correctType:
            finalDF.drop([constants.NAME_ACTUAL], axis=1, inplace=True)
        finalDF.to_csv(self._processedFileName, encoding="UTF-8", index=False)

        finalDF = self._attributes[self._attributes.actual == constants.TYPE_MISTAKE]
        finalDF.to_csv("discarded.csv", encoding="UTF-8", index=False)

        if self._currentFileNumber == self._maxFileNumber:
            if os.path.exists(FILE_PROGRESS):
                try:
                    os.remove(FILE_PROGRESS)
                except FileNotFoundError:
                    print("Unable to remove file: {}".format(FILE_PROGRESS))
                finally:
                    print("Unknown error encountered while removing file: {}".format(FILE_PROGRESS))
        else:
            with open(FILE_PROGRESS, "w") as f:
                f.write("[{}]\n".format(SECTION_EDIT))
                f.write("# Progress of image review\n")
                f.write("")
                f.write("{} =  {}".format(ATTRIBUTE_CURRENT, self._currentFileNumber))

    def loadProgress(self) -> bool:
        """
        Load the current image number from the INI file
        :return: True on successful read
        """
        rc = True
        progressINI = OptionsFile(FILE_PROGRESS)
        progressINI.load()
        try:
            self._lastFileReviewed = int(progressINI.option(SECTION_EDIT, ATTRIBUTE_CURRENT))
        except KeyError:
            # TODO: This should be a popup
            print("Unable to find last reviewed image number in progress file")
            rc = False
        return rc


    def updateFileNumber(self):
        """
        Update the current file number on display
        """
        self.image_number.setText(str(self._currentFileNumber) + '/' + str(self._maxFileNumber))


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

            # if "datetime" in exif:
            #     self.updateCaptureDate(theImage.datetime)

            # Image may not have the exif information

            # if "gps_latitude" in exif:
            #     reference = theImage.gps_latitude_ref
            #     self.updateLatitude(GPSUtilities.dms2decdeg(theImage.gps_latitude, reference))
            # else:
            #     self.updateLatitude(UNKNOWN_LONG)
            #
            # if "gps_longitude" in exif:
            #     reference = theImage.gps_longitude_ref
            #     self.updateLongitude(GPSUtilities.dms2decdeg(theImage.gps_longitude, reference))
            # else:
            #     self.updateLongitude(UNKNOWN_LONG)
            #
            # if "gps_speed" in exif:
            #     self.updateSpeed(theImage.gps_speed)
            # else:
            #     self.updateSpeed(UNKNOWN_LONG)

        self.updateFileNumber()

    def _displayImage(self, imageName):
        """
        Display the current image.
        :param imageName:
        """
        width = self.image.width()
        height = self.image.height()
        self._currentFilename = imageName
        print(f"{self._currentFilename} is {width} x {height}")
        #self.image.setGeometry()
        pixmap = QPixmap(imageName)
        #scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio)
        #scaled = pixmap.scaled(self.image.size(), Qt.KeepAspectRatio)
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scaledWidth = scaled.width()
        self._scaledHeight = scaled.height()
        self.image.setPixmap(scaled)


    def setup(self):

        if len(self._fileNames) > 0:
            self._displayImage(self._fileNames[0])
        self.populateBlobTypes(0)

    def setCurrentImage(self, imageNumber):
        self._currentFileNumber = imageNumber
        self._displayImage(self._fileNames[self._currentFileNumber])
        self.updateInformationForCurrentImage()

    def nextImage(self):
        """
        Processes the next image in the sequence
        """
        self.saveProgress()
        self._currentFileNumber += 1

        if self._currentFileNumber == self._maxFileNumber:
            self.button_next.setEnabled(False)
            self._shortcutNext.setEnabled(False)
        self.button_previous.setEnabled(True)

        self._displayImage(self._fileNames[self._currentFileNumber])
        self.populateBlobTypes(self._currentFileNumber)
        self.updateInformationForCurrentImage()

    def previousImage(self):
        self._currentFileNumber -= 1

        if self._currentFileNumber == 0:
            self.button_previous.setEnabled(False)

        self._displayImage(self._fileNames[self._currentFileNumber])
        self.populateBlobTypes(self._currentFileNumber)
        self.updateInformationForCurrentImage()

    def safeExit(self):
        app.quit()

    def exitHandler(self):
        log.debug("Saving progress")
        self.saveProgress()

    def parseImageName(self, name: str) -> (int, int):
        # Images names are of the form image-M-blob-N
        imageNumber, blobNumber = name.split(constants.DASH)

        return imageNumber, blobNumber

    def loadAttributesFromCSV(self, attributes: str) -> bool:
        """
        Load feature attributes and classifications into a pandas frame
        :param attributes: Path to the CSV
        :return: True on success
        """
        rc = True

        try:
            self._attributes = pd.read_csv(attributes)
        except FileNotFoundError as file:
            print("Unable to find file: {}".format(attributes))
            rc = False

        # If there is no "actual" column, insert one, setting the actual type to the predicted
        if constants.NAME_ACTUAL not in self._attributes.columns:
            self._attributes[constants.NAME_ACTUAL] = self._attributes[constants.NAME_TYPE]

        print(f"Loaded from csv")
        return rc

    def _selected(self, value):

        # image-M-blob-N is <crop | weed | mistake>
        text = value.split(' ')

        if text[2].lower() == constants.NAME_CROP.lower():
            value = constants.TYPE_CROP
        elif text[2].lower() == constants.NAME_WEED.lower():
            value = constants.TYPE_WEED
        elif text[2].lower() == constants.NAME_MISTAKE.lower():
            value = constants.TYPE_MISTAKE
        else:
            raise ValueError(f"Unable to determine type for: {text[2]}")

        typeLocation = self._attributes.columns.get_loc(constants.NAME_TYPE)
        actualLocation = self._attributes.columns.get_loc(constants.NAME_ACTUAL)
        assert typeLocation >= 0
        assert actualLocation >= 0

        # There should be only one row
        rows = self._attributes.loc[self._attributes[constants.NAME_NAME] == text[0]]
        for index, row in rows.iterrows():
            self._attributes.iloc[index, typeLocation] = value
            self._attributes.iloc[index, actualLocation] = value

        return



    def populateBlobTypes(self, imageNumber: int):
        names = ["Crop", "Weed"]

        # Delete all the combo boxes added for the previous view
        for i in reversed(range(self.blobType.count())):
            self.blobType.itemAt(i).widget().setParent(None)

        # The name of the image -- the trailing dash is included so we don't confuse image-1 with image-11
        imageName = constants.NAME_IMAGE + constants.DASH + str(imageNumber) + constants.DASH
        df = self._attributes
        blobsInImage = df[df[constants.NAME_NAME].str.contains(imageName)]

        for index, row in blobsInImage.iterrows():
            # creating a combo box widget
            comboBox = QComboBox(self)
            comboBox.activated[str].connect(self._selected)
            # setting geometry of combo box
            #comboBox.setGeometry(200, 150, 120, 40)

            # Put all choices in the combo-box, but the selected one will be the one predicted
            for type in [t.name for t in VegetationType]:
                comboBox.addItem(f"{row[constants.NAME_NAME]} is {type}", row[constants.NAME_NAME])
            comboBox.setCurrentIndex(int(row[constants.NAME_TYPE]))

            # Originally
            # comboBox.addItem(f"{row[constants.NAME_NAME]} is {names[int(row[constants.NAME_TYPE])]}", row[constants.NAME_NAME])
            # comboBox.addItem(f"{row[constants.NAME_NAME]} is {names[not int(row[constants.NAME_TYPE])]}", row[constants.NAME_NAME])
            # comboBox.addItem(f"{row[constants.NAME_NAME]} is {constants.NAME_MISTAKE}", row[constants.NAME_NAME])
            self.blobType.addWidget(comboBox)
        return

    def setBlobTypes(self, vegType: VegetationType):
        """
        Set all the types in the combo boxes based on the name of the sender button
        """
        # rbt = self.sender()
        # #print(f"Button name is: {rbt.text()}")
        # vegType = VegetationType[rbt.text().upper()]
        # Iterate over all the combo boxes
        for i in range(self.blobType.count()):
            myWidget = self.blobType.itemAt(i).widget()
            if isinstance(myWidget, QComboBox):
                #print(f"Setting combobox {i} type to: {vegType}")
                myWidget.setCurrentIndex(vegType.value)
                self._selected(myWidget.currentText())



    def selectDirectory(self) -> str:
        return("C:\\tmp\\output\\tucson-2023-02-12-03-24-31-e65r6xv3ni1eeo56o9zv4i3f3")

    def loadImagesFromDirectory(self, directory: str, pattern: str) -> bool:
        """
        Finds the images in the specified directory and sorts them by modification time
        :param pattern:
        :param directory:
        :return: True on success
        """

        self._sourceDirectory = directory

        rc = True
        try:
            self._fileNames = sorted(glob.glob(os.path.join(directory, pattern)), key=os.path.getmtime)
            print(f"Found {len(self._fileNames)} files")
            self._maxFileNumber = len(self._fileNames) - 1
            rc = len(self._fileNames) > 0
        except Exception as e:
            log.error("Unable to access files in directory: {}".format(self._sourceDirectory))
            rc = False

        return rc

    def updateClassifications(self):
        pass

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
        # pixmap = QPixmap(imageName)
        # #scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio)
        # #scaled = pixmap.scaled(self.image.size(), Qt.KeepAspectRatio)
        # scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # self._scaledWidth = scaled.width()
        # self._scaledHeight = scaled.height()
        # self.image.setPixmap(scaled)

        size = self.image.size()
        pixmap = QPixmap(self._currentFilename)
        scaled_pixmap = pixmap.scaled(newScaledWidth, newScaledHeight, Qt.KeepAspectRatio)
        self.image.setPixmap(scaled_pixmap)

        self._scale *= 2

        size = self.image.size()

        print(f"Current size is {size}")
        scaled_pixmap = self.image.scaledToHeight(newScaledHeight)

        self.image.setPixmap(scaled_pixmap)

    def wheelEvent(self, event):
        pass
        # Debug this later
        # log.debug(f"Mouse scroll detected {event.angleDelta().y()}. Current Zoom: {self._currentZoom}")
        #
        # # Zoom in
        # if event.angleDelta().y() > 0:
        #     if self._currentZoom + 1 in range(0, MAX_ZOOM + 1):
        #         self._currentZoom += 1
        #         self.scaleCurrentImage(event.x(), event.y())
        # # Zoom out
        # else:
        #     if self._currentZoom - 1 in range(0, MAX_ZOOM + 1):
        #         self._currentZoom -= 1
        #         self.scaleCurrentImage(event.x(), event.y())
        # log.debug(f"Mouse scroll detected {event.angleDelta().y()}. Current Zoom: {self._currentZoom}")
        #
        # # Zoom in
        # if event.angleDelta().y() > 0:
        #     if self._currentZoom + 1 in range(0, MAX_ZOOM + 1):
        #         self._currentZoom += 1
        #         self.scaleCurrentImage(event.x(), event.y())
        # # Zoom out
        # else:
        #     if self._currentZoom - 1 in range(0, MAX_ZOOM + 1):
        #         self._currentZoom -= 1
        #         self.scaleCurrentImage(event.x(), event.y())

parser = argparse.ArgumentParser("Image Reviewer")

parser.add_argument('-i', '--input', action="store", required=False,  help="Input directory")
parser.add_argument('-p', '--pattern', action="store", required=False,  default="*.jpg", help="Pattern, i.e., processed-*.jpg")
parser.add_argument('-o', '--output', action="store", required=False,  default="classifications.csv", help="Output CSV")
parser.add_argument('-a', '--attributes', action="store", required=False,  default="results.csv", help="Classification CSV")
parser.add_argument('-t', '--type', action="store_true", required=False, default=False, help="Correct the type and don't write actual column")
parser.add_argument('-l', '--logging', action="store", required=False, default='../jetson/logging.ini', help="Logging configuration")

arguments = parser.parse_args()

if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(-1)

if os.path.isfile(arguments.output):
    print(f"Output file exists: {arguments.output}")
    sys.exit(-1)

# Initialize logging
logging.config.fileConfig(arguments.logging)
log = logging.getLogger("review")

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()

window.setWindowTitle("University of Arizona")


app.aboutToQuit.connect(window.exitHandler)


if arguments.input is not None:
    log.debug(f"Load images from {arguments.input}")
    if not window.loadImagesFromDirectory(arguments.input, arguments.pattern):
        print(f"Unable to load files")
        sys.exit(-1)

window.loadAttributesFromCSV(arguments.attributes)

window.processed = arguments.output
window.correctType = arguments.type

window.setup()
window.show()

sys.exit(app.exec_())
