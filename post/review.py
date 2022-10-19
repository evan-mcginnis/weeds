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

FILE_PROGRESS = "progress.ini"
ATTRIBUTE_CURRENT = "CURRENT"
SECTION_EDIT = "EDIT"

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Images
        self._fileNames = []
        #self._attributes = np.empty(
        self._currentFileNumber = 0
        self._maxFileNumber = 0

        self._lastFileReviewed = 0
        # Attributes
        #self._attributes = np.empty()

        self._sourceDirectory = "."

        self.setupUi(self)

        # Wire up the buttons
        self.button_next.clicked.connect(self.nextImage)
        self.button_previous.clicked.connect(self.previousImage)

        # Set the initial button states
        self.button_next.setEnabled(True)
        self.button_previous.setEnabled(False)

        self.actionExit.triggered.connect(self.safeExit)
        self.actionSave.triggered.connect(self.saveProgress)
        self.actionMark_for_review.triggered.connect(self.markCurrentImageForReview)

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
        if self._currentFileNumber == self._maxFileNumber:
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

        rc = True
        progressINI = OptionsFile(FILE_PROGRESS)
        progressINI.load()
        try:
            self._lastFileReviewed = int(progressINI.option(SECTION_EDIT, ATTRIBUTE_CURRENT))
        except KeyError:
            print("Unable to find last reviewed image number in progress file")
            rc = False
        return rc

    def updateFileNumber(self):
        """
        Update the current file number on display
        """
        self.image_number.setText(str(self._currentFileNumber) + '/' + str(self._maxFileNumber))

    def updateCaptureDate(self, date: str):
        self.image_acquired.setText(date)

    def updateLatitude(self, latitude: str):
        self.image_latitude.setText(latitude)

    def updateLongitude(self, longitude: str):
        self.image_latitude.setText(longitude)

    def updateInformationForCurrentImage(self):

        with open(self._fileNames[self._currentFileNumber], 'rb') as rawImage:
            theImage = Image(rawImage)
            if theImage.has_exif:
                exif = theImage.list_all()
                #print("EXIF Parameters: {}".format(theImage.list_all()))
            else:
                print("Warning: Image contains no EXIF data")
                exif = []

            # if "datetime" in exif:
            #     self.updateCaptureDate(theImage.datetime)

            # Image may not have lat/long information
            if "gps_latitude" in exif:
                self.updateLatitude(theImage.gps_latitude)
            if "gps_longitude" in exif:
                self.updateLongitude(theImage.gps_longitude)

        self.updateFileNumber()

    def _displayImage(self, imageName):
        """
        Display the current image.
        :param imageName:
        """
        width = self.image.width()
        height = self.image.height()
        pixmap = QPixmap(imageName)
        scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(pixmap)


    def setup(self):

        if len(self._fileNames) > 0:
            self._displayImage(self._fileNames[0])

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
        self.button_previous.setEnabled(True)

        self._displayImage(self._fileNames[self._currentFileNumber])
        self.updateInformationForCurrentImage()

    def previousImage(self):
        self._currentFileNumber -= 1

        if self._currentFileNumber == 0:
            self.button_previous.setEnabled(False)

        self._displayImage(self._fileNames[self._currentFileNumber])
        self.updateInformationForCurrentImage()

    def safeExit(self):
        app.quit()

    def exitHandler(self):
        pass

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
        return rc

    def loadImagesFromDirectory(self, directory: str) -> bool:
        """
        Finds the images in the specified directory and sorts them by modification time
        :param directory:
        :return: True on success
        """

        self._sourceDirectory = directory

        rc = True
        try:
            self._fileNames = sorted(glob.glob(directory + "/" + '*.jpg'), key=os.path.getmtime)
            self._maxFileNumber = len(self._fileNames) - 1
            rc = True
        except Exception as e:
            log.error("Unable to access files in directory: {}".format(self._sourceDirectory))
            rc = False

        return rc

    def updateClassifications(self):
        pass


parser = argparse.ArgumentParser("Image Reviewer")

parser.add_argument('-i', '--input', action="store", required=True,  help="Input directory")
parser.add_argument('-a', '--attributes', action="store", required=True,  help="Attributes")
parser.add_argument('-o', '--output', action="store", required=False,  default="classifications.csv", help="Output CSV")

arguments = parser.parse_args()

#logging.config.fileConfig(arguments.log)
log = logging.getLogger("review")

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()

window.setWindowTitle("University of Arizona")


app.aboutToQuit.connect(window.exitHandler)

window.loadImagesFromDirectory(arguments.input)
window.loadAttributesFromCSV(arguments.attributes)
window.setup()

window.show()
if window.loadProgress():
    if window.confirmResumeReview("Resume editing at image {}?".format(window.lastFileReviewed)):
        window.setCurrentImage(window.lastFileReviewed)
        window.currentFileNumber = window.lastFileReviewed
        window.updateInformationForCurrentImage()
    else:
        window.setCurrentImage(0)

sys.exit(app.exec_())
