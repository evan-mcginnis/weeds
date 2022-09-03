#
# U I
#
import re
import sys
from enum import Enum
from time import sleep
import argparse
import threading
import time
import sys
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
import shortuuid

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from console import Ui_MainWindow
from dialog_init import Ui_initProgress

messageNumber = 0
treatments = 0

class Presence(Enum):
    JOINED = 0
    LEFT = 1

class Status(Enum):
    OK = 0
    ERROR = 1

class InitializationSignals(QObject):
    finished = pyqtSignal(name="finished")
    result = pyqtSignal(str, name="result")
    progress = pyqtSignal(int)

class OdometrySignals(QObject):
    distance = pyqtSignal(str, float, name="distance")
    speed = pyqtSignal(str, float, name="speed")
    latitude = pyqtSignal(float, name="latitude")
    longitude = pyqtSignal(float, name="longitude")
    virtual = pyqtSignal()

class SystemSignals(QObject):
    dianostics = pyqtSignal(bool, name="diagnostics")

class TreatmentSignals(QObject):
    plan = pyqtSignal(int, name="plan")


class Housekeeping(QRunnable):
    def __init__(self, signals, systemRoom, odometryRoom, treatmentRoom):
        super().__init__()
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
            # Slow things down a bit so we can read the messages.  Not really needed
            time.sleep(1)
            self._signals.progress.emit(100)

        # Signal that we are done
        self._signals.finished.emit()


class WorkerSystem(QRunnable):
    def __init__(self, room):
        super().__init__()
        self._signals = SystemSignals()

        self._room = room

    @property
    def signals(self) -> SystemSignals:
        return self._signals

    def run(self):
        processMessagesSync(self._room)


class WorkerOdometry(QRunnable):
    def __init__(self, room):
        super().__init__()
        self._signals = OdometrySignals()

        self._room = room

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


class WorkerTreatment(QRunnable):
    def __init__(self, room):
        super().__init__()
        self._signals = TreatmentSignals()

        self._room = room

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
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self._treatmentSignals = None
        self._taskOdometry = None
        self._taskSystem = None
        self._taskHousekeeping = None
        self._taskTreatment = None
        self.setupUi(self)

        # Wire up the buttons
        self.button_start.clicked.connect(self.startWeeding)
        self.button_start_imaging.clicked.connect(self.startImaging)
        self.button_stop.clicked.connect(self.stopOperation)

        self.reset_kph.clicked.connect(self.resetKPH)
        self.reset_distance.clicked.connect(self.resetDistance)
        self.reset_images_taken.clicked.connect(self.resetImageCount)

        self.applyConstantSpeed.clicked.connect(self.startUsingConstantSpeed)

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
    def initializationSignals(self):
        return self._intializationSignals

    @property
    def odometrySignals(self):
        return self._odometrySignals

    @property
    def systemSignals(self):
        return self._systemSignals

    @property
    def treatmentSignals(self):
        return self._treatmentSignals

    def updateLatitude(self, latitude: float):
        if latitude != 0.0:
            self.latitude.display(latitude)
        else:
            self.longitude.display("-----------")

    def updateLongitude(self, longitude: float):
        if longitude != 0.0:
            self.longitude.display(longitude)
        else:
            self.longitude.display("------------")

    def updateCurrentSpeed(self, source, speed: float):
        log.debug("Update current {} speed to {}".format(source, speed))
        if source == constants.SOURCE_VIRTUAL:
            self.average_kph.setStyleSheet("color: black; background-color: yellow")
        else:
            self.average_kph.setStyleSheet("color: black; background-color: white")
        self.setSpeed(speed)

    def updateCurrentDistance(self, source: str, distance: float):
        log.debug("Update current distance")
        self.count_distance.display(distance)
        if source == constants.SOURCE_VIRTUAL:
            self.count_distance.setStyleSheet("color: black; background-color: yellow")
        else:
            self.count_distance.setStyleSheet("color: black; background-color: white")

    def setupWindow(self):
        # Adjust the table headers.  Can't seem to set this in designer
        header = self.statusTable.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        # This seems like a bug in designer
        self.statusTable.horizontalHeader().setVisible(True)
        self.statusTable.verticalHeader().setVisible(True)

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
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD),
             "status": [3,2]},

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

        self.status_camera_left.setText(constants.UI_STATUS_OK)
        self.status_camera_left.setStyleSheet("color: white; background-color: green")
        self.status_camera_right.setText(constants.UI_STATUS_OK)
        self.status_camera_right.setStyleSheet("color: white; background-color: green")

        self.tractor_progress.setStyleSheet("color: white; background-color: green")
        self.tractor_progress.setValue(0)

        self.noteMissingEntities()

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

    def setStatus(self, occupant: str, roomName: str, presence: Presence):
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
                if presence == Presence.LEFT:
                    self.statusTable.setItem(x,y,QtWidgets.QTableWidgetItem(constants.UI_STATUS_NOT_OK))
                    item = self.statusTable.item(x,y)
                    item.setBackground(QtGui.QColor("red"))
                    item.setForeground(QtGui.QColor("white"))
                    #requiredOccupant.get("status").setText(constants.UI_STATUS_NOT_OK)
                    #requiredOccupant.get("status").setStyleSheet("color: white; background-color: red")
                else:
                    self.statusTable.setItem(x,y,QtWidgets.QTableWidgetItem(constants.UI_STATUS_OK))
                    item = self.statusTable.item(x,y)
                    item.setBackground(QtGui.QColor("green"))
                    item.setForeground(QtGui.QColor("black"))
                    #requiredOccupant.get("status").setText(constants.UI_STATUS_OK)
                    #requiredOccupant.get("status").setStyleSheet("color: white; background-color: green")

        self.statusTable.update()

        # Note in the tabs if someone is missing who should be there
        self.noteMissingEntities()


    def addImage(self):
        return

    def startOperation(self, operation: str):
        # Disable the start button and enable the stop
        self.button_start.setEnabled(False)
        self.button_start_imaging.setEnabled(False)
        self.button_stop.setEnabled(True)

        self.reset_kph.setEnabled(True)
        self.reset_distance.setEnabled(True)
        self.reset_images_taken.setEnabled(True)

        self.status_current_operation.setText(operation)
        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.START
        sessionName = shortuuid.uuid()
        systemMessage.name = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + sessionName

        # TODO: move this to time_ns
        systemMessage.timestamp = time.time() * 1000
        self._systemRoom.sendMessage(systemMessage.formMessage())

    def startImaging(self):
        if self.OKtoImage:
            text = constants.UI_CONFIRM_IMAGING_ALL_OK
        else:
            text = constants.UI_CONFIRM_IMAGING_WITH_ERRORS

        if self.confirmOperation(text):
            self.startOperation(constants.UI_OPERATION_IMAGING)
            self.tabWidget.setIconSize(QtCore.QSize(32, 32))
            self.tabWidget.setTabIcon(0, QtGui.QIcon('camera.png'))


    def resetKPH(self):
        self.setSpeed(0.0)

    def resetImageCount(self):
        global treatments
        treatments = 0
        self.setTreatments(0)

    def resetDistance(self):
        self.absoluteDistance = self.currentDistance

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

        systemMessage.action = constants.Action.STOP
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
        qm.exec_()
        if qm.standardButton(qm.clickedButton()) == QMessageBox.Yes:
            confirmed = True
        else:
            confirmed = False

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

    def runTasks(self):
        """
        Startup all threads needed for application
        """
        pool = QThreadPool.globalInstance()
        self._taskHousekeeping = Housekeeping(self._intializationSignals, self._systemRoom, self._odometryRoom, self._treatmentRoom)
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
        self._odometrySignals.distance.connect(self.updateCurrentDistance)
        self._odometrySignals.speed.connect(self.updateCurrentSpeed)
        self._odometrySignals.latitude.connect(self.updateLatitude)
        self._odometrySignals.longitude.connect(self.updateLongitude)

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
        #window.setTreatments(treatments)
        log.debug("Treatments: {:.02f}".format(treatments))
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
            window.setStatus(presence.getFrom().getResource(), presence.getFrom().getStripped(), Presence.LEFT)
            if presence.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM):
                log.debug("{} left the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
                systemRoom.occupantExited(presence.getFrom().getResource())
            elif presence.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY):
                log.debug("{} left the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
                odometryRoom.occupantExited(presence.getFrom().getResource())
        # Otherwise the occupant entered the room
        else:
            log.debug("{} entered the room {}".format(presence.getFrom().getResource(), presence.getFrom()))
            window.setStatus(presence.getFrom().getResource(), presence.getFrom().getStripped(), Presence.JOINED)
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

def processMessages(room: MUCCommunicator):
    # Connect to the XMPP server and just return
    room.connect(False, True)

def processMessagesSync(room: MUCCommunicator):
    # Connect to the XMPP server and just return
    room.connect(True, True)

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

app = QtWidgets.QApplication(sys.argv)

window = MainWindow()
window.setupWindow()

dialogInit = DialogInit(3, window.initializationSignals)
dialogInit.show()

window.setWindowTitle("University of Arizona")
window.odometryRoom = odometryRoom
window.systemRoom = systemRoom
window.treatmentRoom = treatmentRoom
window.setupRooms(odometryRoom, systemRoom, treatmentRoom)

log.debug("Starting Qt Tasks")
window.runTasks()

app.aboutToQuit.connect(window.exitHandler)

#window.setStatus()
dialogInit.exec()
window.show()
window.setInitialState()
sys.exit(app.exec_())


