#
# U I
#
import datetime
import re
import sys
from enum import Enum
from time import sleep
import argparse
import threading
import time
import sys
import urllib.request
import urllib.error
from threading import Semaphore

from typing import Callable

import dns.resolver
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5 import QtGui, QtCore

from OptionsFile import OptionsFile
import logging
import logging.config
import xmpp
import constants

from MUCCommunicator import MUCCommunicator
from Messages import MUCMessage, OdometryMessage, SystemMessage, TreatmentMessage
from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure

import shortuuid

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from console import Ui_MainWindow
from dialog_init import Ui_initProgress

shuttingDown = False

messageNumber = 0
treatments = 0

class Presence(Enum):
    JOINED = 0
    LEFT = 1

class Status(Enum):
    OK = 0
    ERROR = 1

class WeedsSignals(QObject):
    xmppStatus = pyqtSignal(str, name="xmppStatus")

class InitializationSignals(WeedsSignals):
    finished = pyqtSignal(name="finished")
    result = pyqtSignal(str, name="result")
    progress = pyqtSignal(int)

class OdometrySignals(WeedsSignals):
    distance = pyqtSignal(str, float, name="distance")
    speed = pyqtSignal(str, float, name="speed")
    latitude = pyqtSignal(float, name="latitude")
    longitude = pyqtSignal(float, name="longitude")
    progress = pyqtSignal(float, name="progress")
    virtual = pyqtSignal()

class SystemSignals(WeedsSignals):
    diagnostics = pyqtSignal(str, str, name="diagnostics")
    camera = pyqtSignal(str, str, name="camera")
    operation = pyqtSignal(str, str, name="operation")
    occupant = pyqtSignal(str, str, str, name="occupant")

class TreatmentSignals(WeedsSignals):
    plan = pyqtSignal(int, name="plan")
    image = pyqtSignal(str, str, name="image")


class Housekeeping(QRunnable):
    def __init__(self, initializing: Semaphore, signals, systemRoom : MUCCommunicator, odometryRoom : MUCCommunicator, treatmentRoom: MUCCommunicator):
        super().__init__()
        self._initializing = initializing
        self._signals = signals

        self._systemRoom = systemRoom
        self._odometryRoom = odometryRoom
        self._treatmentRoom = treatmentRoom

    def run(self):

        log.debug("Housekeeping Worker thread")

        # Confirm the connectivity first
        for chatroom in [self._systemRoom, self._odometryRoom, self._treatmentRoom]:
            self._signals.result.emit("Connecting to {}".format(chatroom.muc))
            while not chatroom.connected:
                log.debug("Waiting for {} room connection.".format(chatroom.muc))
                time.sleep(0.5)
            # Slow things down a bit so we can read the messages.  Not really needed
            time.sleep(1)
            log.debug("Connected to {}".format(chatroom.muc))
            chatroom.getOccupants()
            self._signals.progress.emit(100)

        # Have everyone run diagnostics
        systemMessage = SystemMessage()
        systemMessage.action = constants.Action.START_DIAG.name
        self._systemRoom.sendMessage(systemMessage.formMessage())

        # Indicate to waiting threads that we are good to go
        self._initializing.release()

        # Signal that we are done
        self._signals.finished.emit()

class Worker(QRunnable):
    def __init__(self, room):
        super().__init__()

        self._room = room


class WorkerSystem(Worker):
    def __init__(self, room):
        super().__init__(room)
        self._signals = SystemSignals()

        self._room = room

    @property
    def signals(self) -> SystemSignals:
        return self._signals

    def run(self):
        processMessagesSync(self._room)


class WorkerOdometry(Worker):
    def __init__(self, room):
        super().__init__(room)
        self._signals = OdometrySignals()

    @property
    def signals(self) -> OdometrySignals:
        return self._signals

    def run(self):
        processMessagesSync(self._room)

    def process(self, conn, msg: xmpp.protocol.Message):
        if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP,
                                                         constants.PROPERTY_ROOM_ODOMETRY):
            odometryMessage = OdometryMessage(raw=msg.getBody())
            self._signals.distance.emit(odometryMessage.source, float(odometryMessage.speed))
            # window.setSpeed(odometryMessage.speed)
            self._signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
            # window.setDistance(odometryMessage.totalDistance)
        else:
            log.error("Processed message that was not for odometry")


class WorkerTreatment(Worker):
    def __init__(self, room):
        super().__init__(room)
        self._signals = TreatmentSignals()

    @property
    def signals(self) -> TreatmentSignals:
        return self._signals

    def run(self):
        processMessagesSync(self._room)

    def process(self, conn, msg: xmpp.protocol.Message):
        if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP,
                                                         constants.PROPERTY_ROOM_ODOMETRY):
            odometryMessage = OdometryMessage(raw=msg.getBody())
        else:
            log.error("Processed message that was not for odometry")

class DialogInit(QtWidgets.QDialog, Ui_initProgress):
    def __init__(self, steps, signals, *args, obj=None, **kwargs):
        super(DialogInit, self).__init__(*args, **kwargs)
        self.ui = Ui_initProgress()
        self.setupUi(self)
        self._percentComplete = 0

        # The number of steps for the initialization
        self._stepsTotal = steps
        self._stepsComplete = 0
        self._signals = signals

        self._signals.result.connect(self.currentStep)
        self._signals.progress.connect(self.updateProgress)
        self._signals.finished.connect(self.finished)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

    def updateProgress(self, percentComplete: int):
        log.debug("Update progress: {}".format(percentComplete))
        self._stepsComplete += 1
        self._percentComplete = int((self._stepsComplete * (100 / self._stepsTotal)))
        self.initiializationProgress.setValue(self._percentComplete)

    def finished(self):
        #window.button_start.setEnabled(True)
        window.button_start_imaging.setEnabled(True)
        window.reset_kph.setEnabled(True)
        window.reset_distance.setEnabled(True)
        window.reset_images_taken.setEnabled(True)
        window.status_current_operation.setText(constants.UI_OPERATION_NONE)
        dialogInit.currentStep("Everything is good to go")
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

    def currentStep(self, task: str):
        self.initializationTask.setText(task)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, initializing: Semaphore, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self._initializing = initializing
        self._treatmentSignals = None
        self._taskOdometry = None
        self._taskSystem = None
        self._taskHousekeeping = None
        self._taskTreatment = None
        self._distanceOverCapturedLength = 0.0
        self.setupUi(self)

        # Wire up the buttons
        self.button_start.clicked.connect(self.startWeeding)
        self.button_start_imaging.clicked.connect(self.startImaging)
        self.button_stop.clicked.connect(self.stopOperation)

        self.reset_kph.clicked.connect(self.resetKPH)
        self.reset_distance.clicked.connect(self.resetDistance)
        self.reset_images_taken.clicked.connect(self.resetImageCount)


        # Set the initial button states
        self.button_start.setEnabled(False)
        self.button_start_imaging.setEnabled(False)
        self.button_stop.setEnabled(False)

        self.reset_kph.setEnabled(False)
        self.reset_distance.setEnabled(False)
        self.reset_images_taken.setEnabled(False)

        self._odometryRoom = None
        self._systemRoom = None
        self._treatmentRoom = None

        self._OKtoImage = False

        self.currentDistance = 0.0
        self.absoluteDistance = 0.0

        self._requiredOccupants = list()

        self._intializationSignals = InitializationSignals()
        self._odometrySignals = OdometrySignals()
        self._systemSignals = SystemSignals()
        self._treatmentSignals = TreatmentSignals()

        self.statusTable.setUpdatesEnabled(True)


    @property
    def OKtoImage(self):
        return self._OKtoImage

    @OKtoImage.setter
    def OKtoImage(self, ok: bool):
        self._OKtoImage = ok

    @property
    def taskHousekeeping(self):
        return self._taskHousekeeping

    @property
    def taskOdometry(self) -> WorkerOdometry:
        return self._taskOdometry

    @property
    def taskSystem(self) -> WorkerSystem:
        return self._taskSystem

    @property
    def taskTreatment(self) -> WorkerTreatment:
        return self._taskTreatment

    @property
    def initializationSignals(self) -> pyqtSignal:
        return self._intializationSignals

    @property
    def odometrySignals(self) -> pyqtSignal:
        return self._odometrySignals

    @property
    def systemSignals(self) -> pyqtSignal:
        return self._systemSignals

    @property
    def treatmentSignals(self) -> pyqtSignal:
        return self._treatmentSignals

    def updateLatitude(self, latitude: float):
        if latitude != 0.0:
            self.latitude.display(latitude)
        else:
            self.latitude.display("------------")

    def updateLongitude(self, longitude: float):
        if longitude != 0.0:
            self.longitude.display(longitude)
        else:
            self.longitude.display("------------")

    def updateProgress(self, distance: float):
        self._distanceOverCapturedLength += distance

        try:
            percentage = int((self._distanceOverCapturedLength / 310) * 100)
        except ZeroDivisionError:
            percentage = 0

        log.debug("Advanced {}/{} or {}% of total".format(distance, self._distanceOverCapturedLength, percentage))

        if percentage > 100:
            percentage = 100
            self._distanceOverCapturedLength = 0
        self.tractor_progress.setValue(percentage)

    def updateCurrentSpeed(self, source, speed: float):
        log.debug("Update current {} speed to {}".format(source, speed))
        if source == constants.SOURCE_VIRTUAL:
            self.average_kph.setStyleSheet("color: black; background-color: yellow")
        else:
            self.average_kph.setStyleSheet("color: black; background-color: white")
        self.setSpeed(speed)

    def updateCurrentDistance(self, source: str, distance: float):
        log.debug("Update current distance")
        self.currentDistance = distance

        # The distance in the message is in mm -- display is in cm
        self.count_distance.display((distance - self.absoluteDistance) / 10)
        if source == constants.SOURCE_VIRTUAL:
            self.count_distance.setStyleSheet("color: black; background-color: yellow")
        else:
            self.count_distance.setStyleSheet("color: black; background-color: white")

    def updateOperation(self, operation: str, session: str):
        log.debug("Updating operation {}".format(operation))
        if operation == constants.Operation.IMAGING.name:
            # Disable the start button and enable the stop
            self.button_start.setEnabled(False)
            self.button_start_imaging.setEnabled(False)
            self.button_stop.setEnabled(True)

            self.reset_kph.setEnabled(True)
            self.reset_distance.setEnabled(True)
            self.reset_images_taken.setEnabled(True)

            self.tabWidget.setIconSize(QtCore.QSize(32, 32))
            self.tabWidget.setTabIcon(0, QtGui.QIcon('camera.png'))
            self.status_current_operation.setText(constants.UI_OPERATION_IMAGING)
        elif operation == constants.Operation.QUIESCENT.name:
            self.button_start.setEnabled(False)
            self.button_start_imaging.setEnabled(True)
            self.button_stop.setEnabled(False)

            self.reset_kph.setEnabled(True)
            self.reset_distance.setEnabled(True)
            self.reset_images_taken.setEnabled(True)
            self.tabWidget.setTabIcon(0, QtGui.QIcon())

            self.status_current_operation.setText(constants.UI_OPERATION_NONE)

    def updateDiagnostics(self, position: str, result: str):
        self.diagnostic_rio.setText(result)

    def updateCamera(self, position: str, result: str):
        """
        Update the camera status object
        :param position: LEFT or RIGHT
        :param result: OK, NOT_OK, or UNKNOWN
        :return:
        """
        log.debug("Update camera status: {}/{}".format(position,result))
        if position.lower() == constants.Position.LEFT.name.lower():
            statusItem = self.status_camera_left
        elif position.lower() == constants.Position.RIGHT.name.lower():
            statusItem = self.status_camera_right
        else:
            log.error("Received status update for unknown camera: {}".format(position))
            return

        if result.lower() == constants.OperationalStatus.OK.name.lower():
            statusItem.setText(constants.UI_STATUS_OK)
            statusItem.setStyleSheet("color: white; background-color: green; font-size: 20pt")
        elif result.lower() == constants.OperationalStatus.FAIL.name.lower():
            statusItem.setText(constants.UI_STATUS_NOT_OK)
            statusItem.setStyleSheet("color: white; background-color: red; font-size: 20pt")
        elif result.lower() == constants.OperationalStatus.UNKNOWN.name.lower():
            statusItem.setText(constants.UI_STATUS_UNKNOWN)
            statusItem.setStyleSheet("color: white; background-color: gray; font-size: 20pt")
        else:
            log.error("Received unknown status: {}/{}".format(position, result))
        statusItem.update()

    def setupWindow(self):
        # Adjust the table headers.  Can't seem to set this in designer
        header = self.statusTable.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        # This seems like a bug in designer
        self.statusTable.horizontalHeader().setVisible(True)
        self.statusTable.verticalHeader().setVisible(True)

        nNumRows = 6
        nRowHeight = self.statusTable.rowHeight(0)
        nTableHeight = (nNumRows * nRowHeight) + self.statusTable.horizontalHeader().height() + 2 * self.statusTable.frameWidth();
        self.statusTable.setMinimumHeight(nTableHeight)
        self.statusTable.setMaximumHeight(nTableHeight)

    def setupRooms(self, odometryRoom: MUCCommunicator, systemRoom: MUCCommunicator, treatmentRoom: MUCCommunicator):
        """
        Set up up the rooms and initialize the list of required occupants
        :param odometryRoom:
        :param systemRoom:
        :param treatmentRoom:
        """
        self._systemRoom = systemRoom
        self._odometryRoom = odometryRoom
        self._treatmentRoom = treatmentRoom

        self._requiredOccupants = [
            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": [0,1]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": [1,1]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": [2,1]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": [0,2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": [1,2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": [2,2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD_LEFT),
             "status": [3,2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD_MIDDLE),
             "status": [4,2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD_RIGHT),
             "status": [5,2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": [0,0]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": [1,0]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": [2,0]}
        ]
        self._regularExpression = re.compile(r'\.|@|/')

    def breakdownMUCJID(self, room: str) -> ():

        components = self._regularExpression.split(room)
        return (components[0], components[1], components[2], components[3], components[4])

    def setInitialState(self):
        """
        Initialize values on the screen
        """

        self._initializing.acquire()

        window.setSpeed(0)
        stylesheet = "QHeaderView::section{Background-color:rgb(211,211,211); border - radius: 14px;}"
        self.statusTable.setStyleSheet(stylesheet)
        # Mark every occupant as missing
        requiredOccupantCount = len(self._requiredOccupants)
        for occupant in self._requiredOccupants:
            x = occupant.get("status")[0]
            y = occupant.get("status")[1]
            self.statusTable.setItem(x, y, QtWidgets.QTableWidgetItem(constants.UI_STATUS_NOT_OK))

        # This will split up a JID of the form <room-name>@<conference-name>.<domain>.<domain>/<nickname>
        log.debug("System room has {} occupants".format(len(self._systemRoom.occupants)))
        log.debug("Odometry room has {} occupants".format(len(self._odometryRoom.occupants)))
        log.debug("Treatment room has {} occupants".format(len(self._treatmentRoom.occupants)))
        allOccupants = self._systemRoom.occupants + self._odometryRoom.occupants + self._treatmentRoom.occupants
        currentOccupantCount = len(allOccupants)
        for occupant in allOccupants:
            #for occupant in roomOccupants:
                # The room name will be in the form name@<roomname>.conference.<domain>/<nickname>
                room = occupant.get("jid")
                #components = regularExpression.split(room)
                (roomName, conferenceName, machineName, domainName, nickName) = self.breakdownMUCJID(room)
                #roomName = components[0] + "@" + components[1] + "." + components[2] + "." +components[3]
                fullRoomName = roomName + "@" + conferenceName + "." + machineName + "." + domainName
                log.debug("Initial state for occupant: {}".format(occupant.get("name")))
                self.setStatus(occupant.get("name"), fullRoomName, Presence.JOINED)

        #self.updateCamera(constants.Position.LEFT.name, constants.OperationalStatus.UNKNOWN.name)
        #self.updateCamera(constants.Position.RIGHT.name, constants.OperationalStatus.UNKNOWN.name)

        self.tractor_progress.setStyleSheet("color: white; background-color: green")
        self.tractor_progress.setValue(0)

        self.noteMissingEntities()

        self.getCurrentOperation()

        # Indicate that initialization is complete
        self._initializing.release()


    def getCurrentOperation(self):
        """
        Ask the room what the current operation is.
        """
        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.CURRENT.name
        self._systemRoom.sendMessage(systemMessage.formMessage())

    def noteMissingEntities(self):
        """
        Note missing entities by color coding them in the status table and putting an attention icon in the tab
        """
        # Iterate over the table and see if any entity is not there
        missingEntities = 0
        for row in range(self.statusTable.rowCount()):
            for column in range(self.statusTable.columnCount()):
                #self.statusTable.setItem(row, column, QtGui.)
                _item = self.statusTable.item(row, column)
                if _item:
                    text = self.statusTable.item(row, column).text()
                    if text == constants.UI_STATUS_NOT_OK:
                        # Highlight the missing entity
                        _item.setBackground(QtGui.QColor("red"))
                        _item.setForeground(QtGui.QColor("white"))
                        missingEntities += 1

        # Warn if all the occupants are not present
        if missingEntities:
            log.error("All occupants are not in the rooms")
            self.tabWidget.setIconSize(QtCore.QSize(32, 32))
            self.tabWidget.setTabIcon(1, QtGui.QIcon('exclamation.png'))
            # Indicate if the user is to be warned about starting the imaging process
            self.OKtoImage = False
        else:
            self.tabWidget.setIconSize(QtCore.QSize(32, 32))
            self.tabWidget.setTabIcon(1, QtGui.QIcon('checkbox.png'))
            self.OKtoImage = True


    @property
    def odometryRoom(self) -> MUCCommunicator:
        return self._odometryRoom

    @odometryRoom.setter
    def odometryRoom(self, room: MUCCommunicator):
        self._odometryRoom = room

    @property
    def systemRoom(self) -> MUCCommunicator:
        return self._systemRoom

    @systemRoom.setter
    def systemRoom(self, room: MUCCommunicator):
        self._systemRoom = room

    @property
    def treatmentRoom(self) -> MUCCommunicator:
        return self._treatmentRoom

    @treatmentRoom.setter
    def treatmentRoom(self, room: MUCCommunicator):
        self._treatmentRoom = room

    def setSpeed(self, speed: float):
        self.average_kph.display(round(speed,1))

    def setDistance(self, distance: float):
        self.currentDistance = distance
        # If the user has reset the distance to 0, use the offset
        self.count_distance.display(distance - self.absoluteDistance)

    def setTreatments(self, treatments: int):
        self.count_images.display(treatments)

    def setImage(self, position: str, url: str):
        try:
            request = urllib.request.urlopen(url)
            data = request.read()
            #pixmap = QPixmap()
            width = self.image_camera_left.width()
            height = self.image_camera_left.height()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            scaled = pixmap.scaled(width,height,Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if position == constants.Position.LEFT.name.lower():
                #self.image_camera_left.setMaximumSize(pixmap.size())
                self.image_camera_left.setPixmap(pixmap)
                #self.image_camera_left.setMaximumSize(QtCore.QSize(3990,3000))
            elif position == constants.Position.RIGHT.name.lower():
                self.image_camera_right.setPixmap(pixmap)

        except urllib.error.HTTPError as httperror:
            log.error("Unable to fetch from URL: {}".format(url))
        except urllib.error.URLError as urlerror:
            log.error("Unable to fetch from URL: {}".format(url))

    def setStatus(self, occupant: str, roomName: str, presence: str): #presence: Presence):
        """
        Sets the status for an occupant based on the list of required occupants.
        Only those occupants required to be in the room will have status updated.
        :param occupant: The name of the occupant without the room name
        :param presence: Presence.JOINED or Presence.LEFT
        """
        log.debug("Set status for {} in room {}".format(occupant, roomName))
        # Walk through the list of required occupants
        for requiredOccupant in self._requiredOccupants:
            # If an occupant is supposed to be there, update the status
            log.debug("Checking {} in room {} against {} {}".format(occupant, roomName, requiredOccupant.get("name"), requiredOccupant.get("room")))
            if requiredOccupant.get("name") == occupant and requiredOccupant.get("room") == roomName:
                # The occupant left or joined
                x = requiredOccupant.get("status")[0]
                y = requiredOccupant.get("status")[1]
                if presence == Presence.LEFT.name:
                    self.statusTable.setItem(x,y,QtWidgets.QTableWidgetItem(constants.UI_STATUS_NOT_OK))
                    item = self.statusTable.item(x,y)
                    item.setBackground(QtGui.QColor("red"))
                    item.setForeground(QtGui.QColor("white"))
                    #requiredOccupant.get("status").setText(constants.UI_STATUS_NOT_OK)
                    #requiredOccupant.get("status").setStyleSheet("color: white; background-color: red")
                else:
                    self.statusTable.setItem(x,y,QtWidgets.QTableWidgetItem(constants.UI_STATUS_OK))
                    item = self.statusTable.item(x,y)
                    if item is not None:
                        item.setBackground(QtGui.QColor("green"))
                        item.setForeground(QtGui.QColor("black"))
                    #requiredOccupant.get("status").setText(constants.UI_STATUS_OK)
                    #requiredOccupant.get("status").setStyleSheet("color: white; background-color: green")

        self.statusTable.update()

        # Curious -- this is the only way to get the table to update from NOT OK to OK.  The other way works just fine.
        self.statusTable.viewport().update()
        #self.statusTable.repaint()

        # Note in the tabs if someone is missing who should be there
        self.noteMissingEntities()


    def addImage(self):
        return

    def startOperation(self, operation: str, operationDescription: str):
        # Disable the start button and enable the stop
        try:
            self.button_start.setEnabled(False)
            self.button_start_imaging.setEnabled(False)
            self.button_stop.setEnabled(True)

            self.reset_kph.setEnabled(True)
            self.reset_distance.setEnabled(True)
            self.reset_images_taken.setEnabled(True)

            self.status_current_operation.setText(operationDescription)
        except Exception as e:
            log.fatal(e)

        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.START.name
        now = datetime.datetime.now()

        # Construct the name for this session that is legal for AWS
        timeStamp = now.strftime('%Y-%m-%d-%H-%M-%S-')
        sessionName = timeStamp + shortuuid.uuid()
        systemMessage.name = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + sessionName
        systemMessage.operation = operation
        # TODO: move this to time_ns
        systemMessage.timestamp = time.time() * 1000
        self._systemRoom.sendMessage(systemMessage.formMessage())

    def startImaging(self):
        if self.OKtoImage:
            text = constants.UI_CONFIRM_IMAGING_ALL_OK
        else:
            text = constants.UI_CONFIRM_IMAGING_WITH_ERRORS

        if self.confirmOperation(text):
            self.startOperation(constants.Operation.IMAGING.name, constants.UI_OPERATION_IMAGING)
            try:
                self.tabWidget.setIconSize(QtCore.QSize(32, 32))
                self.tabWidget.setTabIcon(0, QtGui.QIcon('camera.png'))
            except Exception as e:
                log.fatal(e)


    def resetKPH(self):
        self.setSpeed(0.0)

    def resetImageCount(self):
        global treatments
        treatments = 0
        self.setTreatments(0)

    def resetDistance(self):
        self.absoluteDistance = self.currentDistance
        self.count_distance.display(0.0)

    def startUsingConstantSpeed(self):
        desiredSpeed = self.constantSpeed.value()
        log.debug("Use constant speed {}".format(desiredSpeed))

    def startWeeding(self):
        # Disable the start button and enable the stop
        self.button_start.setEnabled(False)
        self.button_start_imaging.setEnabled(False)
        self.button_stop.setEnabled(True)

        self.reset_kph.setEnabled(True)
        self.reset_distance.setEnabled(True)
        self.reset_images_taken.setEnabled(True)

        self.status_current_operation.setText(constants.UI_OPERATION_WEEDING)
        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.START
        sessionName = shortuuid.ShortUUID().random(length=22)
        systemMessage.name = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + sessionName
        systemMessage.timestamp = time.time() * 1000
        self._systemRoom.sendMessage(systemMessage.formMessage())


    def stopOperation(self):
        # Enable the start button and disable the stop
        #self.button_start.setEnabled(True)
        self.button_start_imaging.setEnabled(True)
        self.button_stop.setEnabled(False)

        # self.reset_kph.setEnabled(False)
        # self.reset_distance.setEnabled(False)
        # self.reset_images_taken.setEnabled(False)

        self.status_current_operation.setText(constants.UI_OPERATION_NONE)

        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.STOP.name
        systemMessage.timestamp = time.time() * 1000
        self._systemRoom.sendMessage(systemMessage.formMessage())

        # Remove the icon in the tab
        self.tabWidget.setTabIcon(0, QtGui.QIcon())
        log.debug("Stop Weeding")

    def confirmOperation(self, text):
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
        qm.setWindowTitle("Weeding")
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

    def exitHandler(self):
        """
        Clean up on exit by waiting for threads to finish execution.
        """
        log.debug("Clean up items")
        # Indicate that the MUC processing should stop
        self.systemRoom.processing = False
        self.odometryRoom.processing = False
        self.treatmentRoom.processing = False

        pool = QThreadPool.globalInstance()

        # Wait for the threads to finish.
        terminated = pool.waitForDone(5000)
        log.debug("Termination of threads: {}".format(terminated))

    def initialize(self):

        log.debug("Initializing application")
        pool = QThreadPool.globalInstance()
        pool.start(self._taskHousekeeping)

    def runTasks(self, initializing: Semaphore):
        """
        Startup all threads needed for application
        """
        pool = QThreadPool.globalInstance()
        self._taskHousekeeping = Housekeeping(initializing, self._intializationSignals, self._systemRoom, self._odometryRoom, self._treatmentRoom)
        self._taskHousekeeping.setAutoDelete(True)
        pool.start(self._taskHousekeeping)

        self._taskSystem = WorkerSystem(self._systemRoom)
        self._taskSystem.setAutoDelete(True)
        self._systemSignals = self._taskSystem.signals

        self._taskOdometry = WorkerOdometry(self._odometryRoom)
        self._taskOdometry.setAutoDelete(True)
        self._odometrySignals = self._taskOdometry.signals

        self._taskTreatment = WorkerTreatment(self._treatmentRoom)
        self._taskTreatment.setAutoDelete(True)
        self._treatmentSignals = self._taskTreatment.signals

        self._treatmentSignals.plan.connect(self.setTreatments)
        self._treatmentSignals.image.connect(self.setImage)

        self._odometrySignals.progress.connect(self.updateProgress)
        self._odometrySignals.distance.connect(self.updateCurrentDistance)
        self._odometrySignals.speed.connect(self.updateCurrentSpeed)
        self._odometrySignals.latitude.connect(self.updateLatitude)
        self._odometrySignals.longitude.connect(self.updateLongitude)

        self._systemSignals.operation.connect(self.updateOperation)
        self._systemSignals.diagnostics.connect(self.updateDiagnostics)
        self._systemSignals.camera.connect(self.updateCamera)
        self._systemSignals.occupant.connect(self.setStatus)

        pool.start(self._taskSystem)
        pool.start(self._taskOdometry)
        pool.start(self._taskTreatment)





def process(conn, msg: xmpp.protocol.Message):
    global messageNumber
    global treatments

    if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY):
        log.debug("Processing Odometry message")
        odometryMessage = OdometryMessage(raw=msg.getBody())
        signals = window.taskOdometry.signals
        signals.distance.emit(odometryMessage.source, float(odometryMessage.totalDistance))
        signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
        signals.progress.emit(float(odometryMessage.distance))

        # See if we have lat/long.  Bad form here, as 0,0 is a legit value
        if odometryMessage.latitude != 0:
            signals.latitude.emit(float(odometryMessage.latitude))
            signals.longitude.emit(float(odometryMessage.longitude))
        else:
            signals.latitude.emit(0.0)
            signals.longitude.emit(0.0)

        log.debug("Speed: {:.02f}".format(odometryMessage.speed))
    elif msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT):
        treatmentMessage = TreatmentMessage(raw=msg.getBody())
        treatments += 1
        signals = window.taskTreatment.signals
        signals.plan.emit(treatments)
        position = treatmentMessage.position
        if len(treatmentMessage.url) > 0 and len(treatmentMessage.position) > 0:
            log.debug("Image is at: {}".format(treatmentMessage.url))
            signals.image.emit(position, treatmentMessage.url)
        #window.Right(treatments)
        log.debug("Treatments: {:.02f}".format(treatments))
    elif msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM):
        systemMessage = SystemMessage(raw=msg.getBody())
        if systemMessage.action == constants.Action.ACK.name:
            signals = window.taskSystem.signals
            signals.operation.emit(systemMessage.operation, systemMessage.name)
        if systemMessage.action == constants.Action.DIAG_REPORT.name:
            log.debug("Diagnostic report received for position {}".format(systemMessage.position))
            signals = window.taskSystem.signals
            signals.diagnostics.emit(systemMessage.position, systemMessage.diagnostics)
            signals.camera.emit(systemMessage.position, systemMessage.statusCamera)
    else:
        print("skipped message {}".format(messageNumber))

    messageNumber += 1

def presenceCB(conn, presence: xmpp.protocol.Message):
    log.debug("Presence changed for {}".format(presence.getFrom().getStripped()))
    if presence.getFrom().getStripped() == constants.JID_CONSOLE:
        log.debug("Presence for the console.  Ignored.")
    else:
        # Indicates leaving the room
        if presence.getID() == None:
            window.systemSignals.occupant.emit(presence.getFrom().getResource(), presence.getFrom().getStripped(), Presence.LEFT.name)
            #window.setStatus(presence.getFrom().getResource(), presence.getFrom().getStripped(), Presence.LEFT)
            if presence.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM):
                log.debug("{} left the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
                systemRoom.occupantExited(presence.getFrom().getResource())
            elif presence.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY):
                log.debug("{} left the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
                odometryRoom.occupantExited(presence.getFrom().getResource())
        # Otherwise the occupant entered the room
        else:
            log.debug("{} entered the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
            window.setStatus(presence.getFrom().getResource(), presence.getFrom().getStripped(), Presence.JOINED.name)
            if presence.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM):
                log.debug("{} entered the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
                systemRoom.occupantEntered(presence.getFrom().getResource(),presence.getFrom().getStripped())
            elif presence.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY):
                log.debug("{} entered the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
                odometryRoom.occupantEntered(presence.getFrom().getResource(), presence.getFrom().getStripped())
        #window.setStatus()

def startupCommunications(options: OptionsFile):
    # The room that will get the announcements about forward or backward progress
    odometryRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP,
                                                  constants.PROPERTY_SERVER),
                                   options.option(constants.PROPERTY_SECTION_XMPP,
                                                  constants.PROPERTY_JID_CONSOLE),
                                   options.option(constants.PROPERTY_SECTION_XMPP,
                                                  constants.PROPERTY_NICK_CONSOLE),
                                   options.option(constants.PROPERTY_SECTION_XMPP,
                                                  constants.PROPERTY_DEFAULT_PASSWORD),
                                   options.option(constants.PROPERTY_SECTION_XMPP,
                                                  constants.PROPERTY_ROOM_ODOMETRY),
                                   process,
                                   presenceCB,
                                   TIMEOUT=constants.PROCESS_TIMEOUT_SHORT)

    # The room that will get status reports about this process
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP,
                                                constants.PROPERTY_SERVER),
                                 options.option(constants.PROPERTY_SECTION_XMPP,
                                                constants.PROPERTY_JID_CONSOLE),
                                 options.option(constants.PROPERTY_SECTION_XMPP,
                                                constants.PROPERTY_NICK_CONSOLE),
                                 options.option(constants.PROPERTY_SECTION_XMPP,
                                                constants.PROPERTY_DEFAULT_PASSWORD),
                                 options.option(constants.PROPERTY_SECTION_XMPP,
                                                constants.PROPERTY_ROOM_SYSTEM),
                                 process,
                                 presenceCB,
                                 TIMEOUT = constants.PROCESS_TIMEOUT_SHORT)

    # The room that will receiver reports about images and treatment plans
    treatmentRoom = MUCCommunicator( options.option(constants.PROPERTY_SECTION_XMPP,
                                                    constants.PROPERTY_SERVER),
                                     options.option(constants.PROPERTY_SECTION_XMPP,
                                                    constants.PROPERTY_JID_CONSOLE),
                                     options.option(constants.PROPERTY_SECTION_XMPP,
                                                    constants.PROPERTY_NICK_CONSOLE),
                                     options.option(constants.PROPERTY_SECTION_XMPP,
                                                    constants.PROPERTY_DEFAULT_PASSWORD),
                                     options.option(constants.PROPERTY_SECTION_XMPP,
                                                    constants.PROPERTY_ROOM_TREATMENT),
                                     process,
                                     presenceCB,
                                     TIMEOUT=constants.PROCESS_TIMEOUT_SHORT)

    return (odometryRoom, systemRoom, treatmentRoom)

# def processMessages(room: MUCCommunicator):
#     # Connect to the XMPP server and just return
#     room.connect(False, True)

def processMessagesSync(room: MUCCommunicator):
    # Connect to the XMPP server and process incoming messages
    # Curious. Suddenly fetching the occupants list does not work on windows tablet. Perhaps this is a version problem?

    # Originally
    #room.connect(True, False)

    room.processing = True

    while room.processing:
        try:
            # This method should never return unless something went wrong
            room.connect(True)
            log.debug("Connected and processed messages, but encountered errors")
            time.sleep(5)
        except XMPPServerUnreachable:
            log.warning("Unable to connect and process messages.  Will retry and reinitialize.")
            time.sleep(5)
            room.processing = True
        except XMPPServerAuthFailure:
            log.fatal("Unable to authenticate using parameters")
            room.processing = False

    # This is the case where the server was not up to begin with
    # TODO: Sort through this sequence.  If the server is not up, we will get that in the init phase.
    # This is the case where the server can't be reached _after_ initialization
    if not room.connected:
        pass


# Delete this -- this is to keep track of the python threads
threads = list()

parser = argparse.ArgumentParser("Weeding Console")

parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument('-l', '--log', action="store", required=False, default="logging.ini", help="Logging INI")
parser.add_argument('-d', '--dns', action="store", required=False, help="DNS server address")

arguments = parser.parse_args()

# Use a character set that amazon aws will accept
shortuuid.set_alphabet('0123456789abcdefghijklmnopqrstuvwxyz')

# Force resolutions to come from a server that has the entries we want
print("DNS: {}".format(arguments.dns))
my_resolver = dns.resolver.Resolver(configure=False)
my_resolver.nameservers = [arguments.dns]

answer = my_resolver.resolve('jetson.weeds.com')

options = OptionsFile(arguments.ini)
if not options.load():
    print("Failed to load options from {}.".format(arguments.ini))
    sys.exit(1)


logging.config.fileConfig(arguments.log)
log = logging.getLogger("console")

(odometryRoom, systemRoom, treatmentRoom) = startupCommunications(options)

# Control the initialization with this semaphore
initializing = Semaphore()
initializing.acquire()

app = QtWidgets.QApplication(sys.argv)

window = MainWindow(initializing)
window.setupWindow()

dialogInit = DialogInit(3, window.initializationSignals)
dialogInit.show()

window.setWindowTitle("University of Arizona")
window.odometryRoom = odometryRoom
window.systemRoom = systemRoom
window.treatmentRoom = treatmentRoom
window.setupRooms(odometryRoom, systemRoom, treatmentRoom)

log.debug("Starting Qt Tasks")
window.runTasks(initializing)

app.aboutToQuit.connect(window.exitHandler)

#window.setStatus()
dialogInit.exec()
window.show()
window.setInitialState()
sys.exit(app.exec_())


