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
from dateutil import tz

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5 import QtGui, QtCore

from OptionsFile import OptionsFile
import logging
import logging.config
import xmpp
import constants

from MUCCommunicator import MUCCommunicator
from MQCommunicator import ClientMQCommunicator
from Messages import MUCMessage, OdometryMessage, SystemMessage, TreatmentMessage
from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure

import shortuuid

from lorem_text import lorem

from PyQt5 import QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

#from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from console import Ui_MainWindow
from dialog_init import Ui_initProgress

shuttingDown = False

messageNumber = 0
treatments = 0
treatmentsBasler = 0
treatmentsIntel = 0

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

class MQSignals(WeedsSignals):
    distance = pyqtSignal(str, float, name="distance")
    pulses = pyqtSignal(str, float, name="pulses")
    speed = pyqtSignal(str, float, name="speed")
    latitude = pyqtSignal(float, name="latitude")
    longitude = pyqtSignal(float, name="longitude")
    progress = pyqtSignal(float, name="progress")
    agl = pyqtSignal(float, name="agl")
    virtual = pyqtSignal()

class OdometrySignals(WeedsSignals):
    distance = pyqtSignal(str, float, name="distance")
    pulses = pyqtSignal(str, float, name="pulses")
    speed = pyqtSignal(str, float, name="speed")
    latitude = pyqtSignal(float, name="latitude")
    longitude = pyqtSignal(float, name="longitude")
    progress = pyqtSignal(float, name="progress")
    agl = pyqtSignal(float, name="agl")
    virtual = pyqtSignal()

class SystemSignals(WeedsSignals):
    diagnostics = pyqtSignal(SystemMessage, name="diagnostics")
    camera = pyqtSignal(str, str, name="camera")
    operation = pyqtSignal(str, str, name="operation")
    occupant = pyqtSignal(str, str, str, name="occupant")

class TreatmentSignals(WeedsSignals):
    plan = pyqtSignal(int, str, name="plan")
    image = pyqtSignal(str, str, str, name="image")


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
                time.sleep(2)
            log.debug("Connected to {}".format(chatroom.muc))
            retries = 3
            # while retries and len(chatroom.occupants) == 0:
            #     retries -= 1
            #     log.debug("Fetching occupants")
            #
            #     # If we can't get the occupants, sleep for a bit to let the server settle
            #     if not chatroom.getOccupants():
            #         time.sleep(2)
            # log.debug("Occupant list for {} retrieved: {} occupants".format(chatroom.muc, len(chatroom.occupants)))

            self._signals.progress.emit(100)

        # Have everyone run diagnostics
        systemMessage = SystemMessage()
        systemMessage.action = constants.Action.START_DIAG.name
        systemMessage.timestamp = time.time() * 1000
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
        processMessagesSync(self._room, self._signals)


class WorkerOdometry(Worker):
    def __init__(self, room, communicator: ClientMQCommunicator):
        super().__init__(room)
        self._signals = OdometrySignals()
        self._communicator = communicator
        self._currentOdometrySequence = -1
    @property
    def signals(self) -> OdometrySignals:
        return self._signals

    def run(self):
        processMessagesSync(self._room, self._signals)
    def processOdometryMessage(self, message: str):

        log.debug("Process WorkerOdometry message: {}".format(message))
        odometryMessage = OdometryMessage(raw=message)
        if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
            # Ignore this message if already processed
            if odometryMessage.sequence == self._currentOdometrySequence:
                log.debug("Sequence {} has already been processed".format(odometryMessage.sequence))
                return
            else:
                self._currentOdometrySequence = odometryMessage.sequence

            self._signals.pulses.emit(odometryMessage.source, float(odometryMessage.pulses))
            self._signals.distance.emit(odometryMessage.source, float(odometryMessage.totalDistance))
            self._signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
            # See if we have lat/long.  Bad form here, as 0,0 is a legit value
            if odometryMessage.latitude != 0:
                self._signals.latitude.emit(float(odometryMessage.latitude))
                self._signals.longitude.emit(float(odometryMessage.longitude))
            else:
                self._signals.latitude.emit(0.0)
                self._signals.longitude.emit(0.0)
        elif odometryMessage.type == constants.OdometryMessageType.POSITION.name:
            self._signals.agl.emit(odometryMessage.depth)

    def process(self, conn, msg: xmpp.protocol.Message):
        log.debug("Process odometry XMPP message: {}".format(msg))
        if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP,
                                                         constants.PROPERTY_ROOM_ODOMETRY):
            odometryMessage = OdometryMessage(raw=msg.getBody())
            if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
                self._signals.pulses.emit(odometryMessage.source, float(odometryMessage.pulses))
                self._signals.distance.emit(odometryMessage.source, float(odometryMessage.speed))
                # window.setSpeed(odometryMessage.speed)
                self._signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
                # window.setDistance(odometryMessage.totalDistance)
            elif odometryMessage.type == constants.OdometryMessageType.POSITION.name:
                self._signals.agl.emit(odometryMessage.depth)
        else:
            log.error("Processed message that was not for odometry")
class WorkerMQ(Worker):
    def __init__(self, room, communicator: ClientMQCommunicator):
        super().__init__(room)
        self._signals = MQSignals()
        self._communicator = communicator
        self._currentOdometrySequence = -1
        self._processing = False
    @property
    def signals(self) -> MQSignals:
        return self._signals

    @property
    def processing(self) -> bool:
        return self._processing

    @processing.setter
    def processing(self, processingFlag):
        self._processing = processingFlag

    def connectMQ(self) -> bool:
        serverResponding = False
        self._communicator.connect()
        while not serverResponding:
            (serverResponding, response) = self._communicator.sendMessageAndWaitForResponse(constants.COMMAND_PING, 10000)
            if not serverResponding:
                log.error("Odometry server did not respond within 10 seconds. Will retry.")
                self._communicator.disconnect()
            else:
                log.debug("Odometry server responded successfully")
        return serverResponding

    def run(self):
        self._processing = True
        # Wait for the initial connection before proceeding
        serverResponding = self.connectMQ()
        while self._processing:
            (serverResponding, response) = self._communicator.sendMessageAndWaitForResponse(constants.COMMAND_ODOMETERY, 1000)
            # If the server responds, process the message, otherwise reconnect.
            if serverResponding:
                odometryStatus = constants.OperationalStatus.OK
                self.processOdometryMessage(response)
            else:
                odometryStatus = constants.OperationalStatus.FAIL
                log.debug("The odometry server failed to respond. Reconnecting")
                self._communicator.disconnect()
                self.connectMQ()

        # self._communicator.callback = self.processOdometryMessage
        # while self._processing:
        #     self._communicator.start(constants.COMMAND_ODOMETERY)
        #     self._communicator.messages = 100
    def processOdometryMessage(self, message: str):

        # log.debug("Process MQ message: {}".format(message))
        odometryMessage = OdometryMessage(raw=message)
        if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
            # if we have already processed the most current reading, just return
            if odometryMessage.sequence == self._currentOdometrySequence:
                return
            else:
                self._currentOdometrySequence = odometryMessage.sequence

            self._signals.pulses.emit(odometryMessage.source, float(odometryMessage.pulses))
            self._signals.distance.emit(odometryMessage.source, float(odometryMessage.totalDistance))
            self._signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
            # See if we have lat/long.  Bad form here, as 0,0 is a legit value
            if odometryMessage.latitude != 0:
                self._signals.latitude.emit(float(odometryMessage.latitude))
                self._signals.longitude.emit(float(odometryMessage.longitude))
            else:
                self._signals.latitude.emit(0.0)
                self._signals.longitude.emit(0.0)
        elif odometryMessage.type == constants.OdometryMessageType.POSITION.name:
            self._signals.agl.emit(odometryMessage.depth)
        else:
            log.error("Bad message type received")

    def process(self, conn, msg: xmpp.protocol.Message):
        if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP,
                                                         constants.PROPERTY_ROOM_ODOMETRY):
            odometryMessage = OdometryMessage(raw=msg.getBody())
            if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
                self._signals.pulses.emit(odometryMessage.source, float(odometryMessage.pulses))
                self._signals.distance.emit(odometryMessage.source, float(odometryMessage.speed))
                # window.setSpeed(odometryMessage.speed)
                self._signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
                # window.setDistance(odometryMessage.totalDistance)
            elif odometryMessage.type == constants.OdometryMessageType.POSITION.name:
                self._signals.agl.emit(odometryMessage.depth)
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
        processMessagesSync(self._room, self._signals)

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
        # log.debug("Update progress: {}".format(percentComplete))
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
        self._currentPulses = 0
        self.setupUi(self)

        # Wire up the buttons
        self.button_start.clicked.connect(self.startWeeding)
        self.button_start_imaging.clicked.connect(self.startImaging)
        self.button_stop.clicked.connect(self.stopOperation)

        self.reset_kph.clicked.connect(self.resetKPH)
        self.reset_distance.clicked.connect(self.resetDistance)
        self.reset_images_taken.clicked.connect(self.resetImageCount)

        self.purge_left.clicked.connect(self.purgeAllHandler)
        self.purge_right.clicked.connect(self.purgeAllHandler)

        # Thw sliders selecting the purge time for the emitters
        self.slider_left.valueChanged.connect(self.sliderValueChanged)
        self.slider_right.valueChanged.connect(self.sliderValueChanged)

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

        self._odometryMQCommunicator = None

        self._OKtoImage = False

        self.currentDistance = 0.0
        self.absoluteDistance = 0.0
        self.absolutePulses = 0

        self._requiredOccupants = list()

        self._intializationSignals = InitializationSignals()
        self._odometrySignals = OdometrySignals()
        self._systemSignals = SystemSignals()
        self._treatmentSignals = TreatmentSignals()
        self._mqSignals = MQSignals()

        # No longer used
        # self.statusTable.setUpdatesEnabled(True)

        # Wire up the combo boxes to display the images
        self.images_left.activated[str].connect(self.onImageSelectedLeft)
        self.images_right.activated[str].connect(self.onImageSelectedRight)

        self.images_left_intel.activated[str].connect(self.onImageSelectedLeft)
        self.images_right_intel.activated[str].connect(self.onImageSelectedRight)

        # Dialogs
        self._dialogDisconnected = QMessageBox()

        self._emitterButtons = [
            {'button': self.emitter_left_1_1, 'side': constants.Side.LEFT, 'tier': 1, 'position': 1},
            {'button': self.emitter_left_1_2, 'side': constants.Side.LEFT, 'tier': 1, 'position': 2},
            {'button': self.emitter_left_1_3, 'side': constants.Side.LEFT, 'tier': 1, 'position': 3},
            {'button': self.emitter_left_2_1, 'side': constants.Side.LEFT, 'tier': 2, 'position': 1},
            {'button': self.emitter_left_2_2, 'side': constants.Side.LEFT, 'tier': 2, 'position': 2},
            {'button': self.emitter_left_2_3, 'side': constants.Side.LEFT, 'tier': 2, 'position': 3},
            {'button': self.emitter_left_3_1, 'side': constants.Side.LEFT, 'tier': 3, 'position': 1},
            {'button': self.emitter_left_3_2, 'side': constants.Side.LEFT, 'tier': 3, 'position': 2},
            {'button': self.emitter_left_3_3, 'side': constants.Side.LEFT, 'tier': 3, 'position': 3},
            {'button': self.emitter_left_4_1, 'side': constants.Side.LEFT, 'tier': 4, 'position': 1},
            {'button': self.emitter_left_4_2, 'side': constants.Side.LEFT, 'tier': 4, 'position': 2},
            {'button': self.emitter_left_4_3, 'side': constants.Side.LEFT, 'tier': 4, 'position': 3},

            {'button': self.emitter_right_1_1, 'side': constants.Side.RIGHT, 'tier': 1, 'position': 1},
            {'button': self.emitter_right_1_2, 'side': constants.Side.RIGHT, 'tier': 1, 'position': 2},
            {'button': self.emitter_right_1_3, 'side': constants.Side.RIGHT, 'tier': 1, 'position': 3},
            {'button': self.emitter_right_2_1, 'side': constants.Side.RIGHT, 'tier': 2, 'position': 1},
            {'button': self.emitter_right_2_2, 'side': constants.Side.RIGHT, 'tier': 2, 'position': 2},
            {'button': self.emitter_right_2_3, 'side': constants.Side.RIGHT, 'tier': 2, 'position': 3},
            {'button': self.emitter_right_3_1, 'side': constants.Side.RIGHT, 'tier': 3, 'position': 1},
            {'button': self.emitter_right_3_2, 'side': constants.Side.RIGHT, 'tier': 3, 'position': 2},
            {'button': self.emitter_right_3_3, 'side': constants.Side.RIGHT, 'tier': 3, 'position': 3},
            {'button': self.emitter_right_4_1, 'side': constants.Side.RIGHT, 'tier': 4, 'position': 1},
            {'button': self.emitter_right_4_2, 'side': constants.Side.RIGHT, 'tier': 4, 'position': 2},
            {'button': self.emitter_right_4_3, 'side': constants.Side.RIGHT, 'tier': 4, 'position': 3}
        ]

        self.setEmitterButtonsState(True)
        self.setEmitterButtonHandler()

        # Run diagnostics every 5 seconds
        # Move this to the controller subsystem that is on the middle system.  The tablet may not always be in contact
        # with the weeding system
        # self.diagnosticTimer = QTimer()
        # self.diagnosticTimer.timeout.connect(self.startDiagnostics)
        # self.diagnosticTimer.start(5000)

        # Run another timer to mark things as failed it we haven't heard from them in a while
        self.unresponsiveTimer = QTimer()
        self.unresponsiveTimer.timeout.connect(self.markSystemsAsFailed)
        self.unresponsiveTimer.start(8000)

        self.diagnosticsReceivedFromLeft = 0
        self.diagnosticsReceivedFromRight = 0
        self.diagnosticsReceivedFromMiddle = 0

        self.KEY_DIAGNOSTIC_TIME = 'diagnostic'
        self.KEY_GROUP = 'group'
        self.KEY_GROUP_NAME = 'name'

        self._systems = [
            {self.KEY_DIAGNOSTIC_TIME: 0, self.KEY_GROUP: self.groupLeft, self.KEY_GROUP_NAME: constants.Position.LEFT.name},
            {self.KEY_DIAGNOSTIC_TIME: 0, self.KEY_GROUP: self.groupMiddle, self.KEY_GROUP_NAME: constants.Position.MIDDLE.name},
            {self.KEY_DIAGNOSTIC_TIME: 0, self.KEY_GROUP: self.groupRight, self.KEY_GROUP_NAME: constants.Position.RIGHT.name}
        ]

        # These are for the image selection combo boxes
        self.stylesheetCurrent = "background-color: green; font-size: 20pt"
        self.stylesheetNotSelected = "background-color: light grey; font-size: 20pt"

    def sliderValueChanged(self):
        sending = self.sender()
        # The name of the slider is slider_<side>
        sliderName = sending.objectName().split('_')
        side = sliderName[1]

        if side == constants.Side.LEFT.name.lower():
            self.purge_time_left.display(self.slider_left.value())
        elif side == constants.Side.RIGHT.name.lower():
            self.purge_time_right.display(self.slider_right.value())
        else:
            log.error("Unable to determine position of emitter for {}".format(sending.objectName()))

    def markSystemsAsFailed(self):
        """
        Mark system as failed if diagnostic report has not been received recently
        """
        # Go through all the systems, and mark them as failed if diagnostic reports have not been received in the last
        # 5 seconds
        stylesheet = "color: white; background-color: grey; font-size: 20pt"
        for group in self._systems:
            log.debug(f"Last diagnostic received for {group[self.KEY_GROUP_NAME]}: {time.time() - group[self.KEY_DIAGNOSTIC_TIME]} s ago")
            if (float(group[self.KEY_DIAGNOSTIC_TIME]) + 9.0) < time.time():
                log.debug("Diagnostics have not been received for position: {}".format(group[self.KEY_GROUP_NAME]))
                group[self.KEY_GROUP].setStyleSheet(stylesheet)

    def indicateSourceOfImage(self, object: QObject):
        name = object.objectName()

        pass

    def onImageSelectedLeft(self, imageName: str):
        sending = self.sender()

        url = sending.itemData(sending.currentIndex())

        # Get the original stylesheet so we can use this to indicate that something is not selected
        self.stylesheetOriginal = sending.styleSheet()
        self.images_left.setStyleSheet(self.stylesheetNotSelected)
        self.images_left_intel.setStyleSheet(self.stylesheetNotSelected)
        sending.setStyleSheet(self.stylesheetCurrent)
        self.showImage(constants.Position.LEFT, url)
        log.debug("Display: {}".format(url))

    def onImageSelectedRight(self, imageName: str):
        sending = self.sender()

        url = sending.itemData(sending.currentIndex())
        self.images_right.setStyleSheet(self.stylesheetNotSelected)
        self.images_right_intel.setStyleSheet(self.stylesheetNotSelected)
        sending.setStyleSheet(self.stylesheetCurrent)
        self.showImage(constants.Position.RIGHT, url)
        log.debug("Display: {}".format(url))

    def setEmitterButtonsState(self, enabled: bool):
        """
        Set the button state of the emitters
        :param enabled:
        """
        for buttonEntry in self._emitterButtons:
            buttonEntry['button'].setEnabled(enabled)

    def startDiagnostics(self):
        """
        Requests entities run diagnostics
        TODO: Move command to run diagnostics to the controller
        """
        # # avoid the user pressing the button twice
        # self.runDiagnostics.setEnabled(False)

        # Have everyone run diagnostics
        systemMessage = SystemMessage()
        systemMessage.action = constants.Action.START_DIAG.name
        self._systemRoom.sendMessage(systemMessage.formMessage())

        # self.runDiagnostics.setEnabled(True)

    def purgeAllHandler(self):
        sending = self.sender()

        # The name of the button is purge_<side>
        buttonName = sending.objectName().split('_')
        side = buttonName[1]

        treatmentMessage = TreatmentMessage()

        # Put a default here in case the reading fails
        treatmentMessage.duration = 5.0

        if side.upper() == constants.Side.RIGHT.name:
            treatmentMessage.duration = self.purge_time_right.value()
        elif side.upper() == constants.Side.LEFT.name:
            treatmentMessage.duration = self.purge_time_left.value()
        else:
            log.error("Unable to determine which side the emitter is on: {}".format(side))
            return

        log.debug("Purge all on side: {} for {} seconds".format(side.upper(), treatmentMessage.duration))

        treatmentMessage.side = side.upper()
        treatmentMessage.tier = constants.EMITTER_ALL
        treatmentMessage.number = constants.EMITTER_ALL
        treatmentMessage.timestamp = time.time() * 1000
        # These are direct emitter instructions, not a plan
        treatmentMessage.plan = constants.Treatment.EMITTER_INSTRUCTIONS

        self._treatmentRoom.sendMessage(treatmentMessage.formMessage())

    def emitterHandler(self):
        sending = self.sender()

        # Break apart the emitter button name of the form emitter_<side>_<tier>_<number>
        emitter = sending.objectName().split('_')
        side = emitter[1]
        tier = emitter[2]
        number = emitter[3]

        treatmentMessage = TreatmentMessage()
        treatmentMessage.side = side.upper()
        treatmentMessage.tier = tier
        treatmentMessage.number = number
        treatmentMessage.timestamp = time.time() * 1000
        # These are direct emitter instructions, not a plan
        treatmentMessage.plan = constants.Treatment.EMITTER_INSTRUCTIONS

        if treatmentMessage.side == constants.Side.RIGHT.value:
            treatmentMessage.duration = self.purge_time_right.value()
        elif treatmentMessage.side == constants.Side.LEFT.value:
            treatmentMessage.duration = self.purge_time_left.value()
        else:
            log.error("Unable to determine which side the emitter is on: [{}]".format(side.upper()))
            return

        log.debug("Enable emitter: ({},{},{}) duration {}".format(treatmentMessage.side,
                                                                  treatmentMessage.tier,
                                                                  treatmentMessage.number,
                                                                  treatmentMessage.duration))

        self._treatmentRoom.sendMessage(treatmentMessage.formMessage())

    def setEmitterButtonHandler(self):
        # for button in self._emitterButtons:
        #     button['button'].clicked.connect(lambda: self.emitterHandler(button['side'], button['tier'], button['position']))
        for button in self._emitterButtons:
            button['button'].clicked.connect(self.emitterHandler)

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
    def taskMQ(self) -> WorkerMQ:
        return self._taskMQ
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

    def updateAGL(self, agl: float):
        if agl != 0:
            self.agl.display(agl)
        else:
            self.agl.display("------------")

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

        # These may come back later, but this allows more room for the images
        # self.tractor_progress_left.setValue(percentage)
        # self.tractor_progress_right.setValue(percentage)

    def updateCurrentSpeed(self, source, speed: float):
        # log.debug("Update current {} speed to {}".format(source, speed))
        if source == constants.SOURCE_VIRTUAL:
            self.average_kph.setStyleSheet("color: black; background-color: yellow")
        else:
            self.average_kph.setStyleSheet("color: black; background-color: white")
        self.setSpeed(speed)

    def updateCurrentDistance(self, source: str, distance: float):
        log.debug("Update current distance: {}".format(distance))
        self.currentDistance = distance

        # The distance in the message is in mm -- display is in cm
        self.count_distance.display((distance - self.absoluteDistance) / 10)
        if source == constants.SOURCE_VIRTUAL:
            self.count_distance.setStyleSheet("color: black; background-color: yellow")
        else:
            self.count_distance.setStyleSheet("color: black; background-color: white")

    def updatePulses(self, source: str, pulses: float):
        log.debug("Update pulse count: {} Absolute Pulses Base: {}".format(pulses, self.absolutePulses))
        self._currentPulses = pulses

        # The distance in the message is in mm -- display is in cm
        self.count_pulses.display(pulses - self.absolutePulses)
        if source == constants.SOURCE_VIRTUAL:
            self.count_pulses.setStyleSheet("color: black; background-color: yellow")
        else:
            self.count_pulses.setStyleSheet("color: black; background-color: white")

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
        pass
        # TODO: Safe to remove
        # Adjust the table headers.  Can't seem to set this in designer
        # header = self.statusTable.horizontalHeader()
        # header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        # header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        # header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        # # This seems like a bug in designer
        # self.statusTable.horizontalHeader().setVisible(True)
        # self.statusTable.verticalHeader().setVisible(True)
        #
        # nNumRows = 7
        # nRowHeight = self.statusTable.rowHeight(0)
        # nTableHeight = (nNumRows * nRowHeight) + self.statusTable.horizontalHeader().height() + 2 * self.statusTable.frameWidth();
        # self.statusTable.setMinimumHeight(nTableHeight)
        # self.statusTable.setMaximumHeight(nTableHeight)


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
             "status": [0, 1]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": [1, 1]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": [2, 1]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": [0, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": [1, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": [2, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD_LEFT),
             "status": [3, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD_MIDDLE),
             "status": [4, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD_RIGHT),
             "status": [5, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONTROL),
             "status": [6, 2]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": [0, 0]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": [1, 0]},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": [2, 0]}
        ]
        self._regularExpression = re.compile(r'\.|@|/')

    def breakdownMUCJID(self, room: str) -> ():

        components = self._regularExpression.split(room)

        if len(components) == 4:
            log.error("Likely no nickname for occupant: {}".format(room))
            breakdown = (components[0], components[1], components[2], components[3], "UNKNOWN")
        elif len(components) == 5:
            breakdown = (components[0], components[1], components[2], components[3], components[4])
        else:
            raise ValueError(room)

        return breakdown

    @staticmethod
    def _statusToBool(status: str) -> bool:
        """
        Utility routine to convert status to boolean
        :param status: 'FAIL' or 'OK"
        :return: boolean
        """
        if status == constants.OperationalStatus.FAIL.name:
            return False
        else:
            return True

    def _diagnosticReceived(self, position: constants.Position):
        """
        Update the entry for the group to reflect diagnostics were received
        :param position: The position received
        """
        for group in self._systems:
            if group[self.KEY_GROUP_NAME] == position.name:
                group[self.KEY_DIAGNOSTIC_TIME] = time.time()

    def updateStatusOfSystem(self, systemMsg: SystemMessage):
        """
        Update the presentation of the status of a system
        :param position: left, middle, or right
        :param status: array of booleans (true == OK, false == NOT OK)
        """

        log.debug("Update status for position: {}".format(systemMsg.position))

        if systemMsg.diagnostics == constants.OperationalStatus.FAIL.name:
            stylesheet = "color: white; background-color: red; font-size: 20pt"
        elif systemMsg.diagnostics == constants.OperationalStatus.OK.name:
            stylesheet = "color: white; background-color: green; font-size: 20pt"
        else:
            stylesheet = "color: white; background-color: grey; font-size: 20pt"

        if systemMsg.position == constants.Position.LEFT.name.lower():
            self.left_checkbox_system.setChecked(self._statusToBool(systemMsg.statusSystem))
            self.left_checkbox_basler.setChecked(self._statusToBool(systemMsg.statusCamera))
            self.left_checkbox_intel.setChecked(self._statusToBool(systemMsg.statusIntel))
            self.left_checkbox_odometry.setChecked(self._statusToBool(systemMsg.statusOdometry))
            self._diagnosticReceived(constants.Position.LEFT)
            self.groupLeft.setStyleSheet(stylesheet)
            # Temporary -- this should retrieve the details from the URL provided in the report
            self.diagnostic_details_left.setText(lorem.paragraphs(2))
        if systemMsg.position == constants.Position.MIDDLE.name.lower():
            self.middle_checkbox_system.setChecked(self._statusToBool(systemMsg.statusSystem))
            self.middle_checkbox_daq.setChecked(self._statusToBool(systemMsg.statusDAQ))
            self.middle_checkbox_intel.setChecked(self._statusToBool(systemMsg.statusIntel))
            self.middle_checkbox_gps.setChecked(self._statusToBool(systemMsg.statusGPS))
            self._diagnosticReceived(constants.Position.MIDDLE)
            self.groupMiddle.setStyleSheet(stylesheet)
            # Temporary -- this should retrieve the details from the URL provided in the report
            self.diagnostic_details_middle.setText(lorem.paragraphs(2))
        if systemMsg.position == constants.Position.RIGHT.name.lower():
            self.right_checkbox_system.setChecked(self._statusToBool(systemMsg.statusSystem))
            self.right_checkbox_basler.setChecked(self._statusToBool(systemMsg.statusCamera))
            self.right_checkbox_intel.setChecked(self._statusToBool(systemMsg.statusIntel))
            self.right_checkbox_odometry.setChecked(self._statusToBool(systemMsg.statusOdometry))
            self._diagnosticReceived(constants.Position.RIGHT)
            self.groupRight.setStyleSheet(stylesheet)
            # Temporary -- this should retrieve the details from the URL provided in the report
            self.diagnostic_details_right.setText(lorem.paragraphs(2))

    def setInitialState(self):
        """
        Initialize values on the screen
        """

        self._initializing.acquire()

        window.setSpeed(0)

        # TODO: Remove references to status table
        # stylesheet = "QHeaderView::section{Background-color:rgb(211,211,211); border - radius: 14px;}"
        # self.statusTable.setStyleSheet(stylesheet)
        # # Mark every occupant as missing
        # requiredOccupantCount = len(self._requiredOccupants)
        # for occupant in self._requiredOccupants:
        #     x = occupant.get("status")[0]
        #     y = occupant.get("status")[1]
        #     self.statusTable.setItem(x, y, QtWidgets.QTableWidgetItem(constants.UI_STATUS_NOT_OK))
        #
        # # This will split up a JID of the form <room-name>@<conference-name>.<domain>.<domain>/<nickname>
        # log.debug("System room has {} occupants".format(len(self._systemRoom.occupants)))
        # log.debug("Odometry room has {} occupants".format(len(self._odometryRoom.occupants)))
        # log.debug("Treatment room has {} occupants".format(len(self._treatmentRoom.occupants)))
        # allOccupants = self._systemRoom.occupants + self._odometryRoom.occupants + self._treatmentRoom.occupants
        # currentOccupantCount = len(allOccupants)
        # for occupant in allOccupants:
        #     #for occupant in roomOccupants:
        #         # The room name will be in the form name@<roomname>.conference.<domain>/<nickname>
        #         room = occupant.get("jid")
        #         #components = regularExpression.split(room)
        #         (roomName, conferenceName, machineName, domainName, nickName) = self.breakdownMUCJID(room)
        #         #roomName = components[0] + "@" + components[1] + "." + components[2] + "." +components[3]
        #         fullRoomName = roomName + "@" + conferenceName + "." + machineName + "." + domainName
        #         log.debug("Initial state for occupant: {}".format(occupant.get("name")))
        #         self.setStatus(occupant.get("name"), fullRoomName, Presence.JOINED)
        #
        # #self.updateCamera(constants.Position.LEFT.name, constants.OperationalStatus.UNKNOWN.name)
        # #self.updateCamera(constants.Position.RIGHT.name, constants.OperationalStatus.UNKNOWN.name)
        #
        # # self.tractor_progress_left.setStyleSheet("color: white; background-color: green")
        # # self.tractor_progress_left.setValue(0)
        # # self.tractor_progress_right.setStyleSheet("color: white; background-color: green")
        # # self.tractor_progress_right.setValue(0)
        #
        # self.noteMissingEntities()
        #
        #
        # # Indicate that the status is not yet known
        # stylesheet = "color: white; background-color: grey; font-size: 20pt"
        # self.groupLeft.setStyleSheet(stylesheet)
        # self.groupMiddle.setStyleSheet(stylesheet)
        # self.groupRight.setStyleSheet(stylesheet)
        #
        # # Redo the status table sizing
        # self.statusTable.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        #
        # self.statusTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.statusTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #
        # self.statusTable.resizeColumnsToContents()
        # self.statusTable.setFixedSize(
        #     self.statusTable.horizontalHeader().length() + self.statusTable.verticalHeader().width(),
        #     self.statusTable.verticalHeader().length() + self.statusTable.horizontalHeader().height())
        #

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
        # TODO: Safe to remove
        # Iterate over the table and see if any entity is not there
        # missingEntities = 0
        # for row in range(self.statusTable.rowCount()):
        #     for column in range(self.statusTable.columnCount()):
        #         #self.statusTable.setItem(row, column, QtGui.)
        #         _item = self.statusTable.item(row, column)
        #         if _item:
        #             text = self.statusTable.item(row, column).text()
        #             if text == constants.UI_STATUS_NOT_OK:
        #                 # Highlight the missing entity
        #                 _item.setBackground(QtGui.QColor("red"))
        #                 _item.setForeground(QtGui.QColor("white"))
        #                 missingEntities += 1
        #
        # # Warn if all the occupants are not present
        # if missingEntities:
        #     log.error("All occupants are not in the rooms")
        #     self.tabWidget.setIconSize(QtCore.QSize(32, 32))
        #     # TODO: This causes an error on the console, as the color space has not been assigned
        #     self.tabWidget.setTabIcon(1, QtGui.QIcon('exclamation.png'))
        #     # Indicate if the user is to be warned about starting the imaging process
        #     self.OKtoImage = False
        # else:
        #     self.tabWidget.setIconSize(QtCore.QSize(32, 32))
        #     self.tabWidget.setTabIcon(1, QtGui.QIcon('checkbox.png'))
        #     self.OKtoImage = True


    @property
    def odometryMQ(self) -> ClientMQCommunicator:
        return self._odometryMQCommunicator

    @odometryMQ.setter
    def odometryMQ(self, communicator: ClientMQCommunicator):
        self._odometryMQCommunicator = communicator

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

    # def setPulses(self, pulses: float):
    #     self._currentPulses = pulses
    #     # If the user has reset the distance to 0, use the offset
    #     self.count_pulses.display(pulses - self.absolutePulses)

    def setDistance(self, distance: float):
        self.currentDistance = distance
        # If the user has reset the distance to 0, use the offset
        self.count_distance.display(distance - self.absoluteDistance)

    def setTreatments(self, treatments: int, source: str):
        if source == constants.Capture.RGB.name:
            self.count_images_basler.display(treatments)
        elif source == constants.Capture.DEPTH_RGB.name:
            self.count_images_intel.display(treatments)
        else:
            log.error("Unknown source for image: {}".format(source))

    def addImage(self, position: str, source: str, url: str):
        """
        Add the image to the list of images that can be selected
        :param position: RIGHT or LEFT
        :param source: RGB (basler) or DEPTH_RGB (Intel)
        :param url:  URL of the image
        """
        # Add the item to the list so it can be shown later
        log.debug("Add image to list for position {} source {}: {}".format(position, source, url))
        if position == constants.Position.LEFT.name.lower():
            if source == constants.Capture.RGB.name:
                self.images_left.addItem("Image {:02d}".format(treatments), url)
            elif source == constants.Capture.DEPTH_RGB.name:
                self.images_left_intel.addItem("Image {:02d}".format(treatments), url)
            else:
                log.error("Unknown source for image: {}".format(source))
        elif position == constants.Position.RIGHT.name.lower():
            if source == constants.Capture.RGB.name:
                self.images_right.addItem("Image {:02d}".format(treatments), url)
            elif source == constants.Capture.DEPTH_RGB.name:
                self.images_right_intel.addItem("Image {:02d}".format(treatments), url)
            else:
                log.error("Unknown source for image: {}".format(source))
        else:
            log.error("Unknown position: {}".format(position))
    def showImage(self, position: constants.Position, url: str):
        try:
            request = urllib.request.urlopen(url)
            data = request.read()
            #pixmap = QPixmap()
            width = self.image_camera_left.width()
            height = self.image_camera_left.height()
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            scaled = pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if position == constants.Position.LEFT:
                #self.image_camera_left.setMaximumSize(pixmap.size())
                self.image_camera_left.setPixmap(pixmap)
                #self.image_camera_left.setMaximumSize(QtCore.QSize(3990,3000))
            elif position == constants.Position.RIGHT:
                self.image_camera_right.setPixmap(pixmap)

        except urllib.error.HTTPError as httperror:
            log.error("Unable to fetch from URL: {}".format(url))
            log.error(httperror)
            self.displayError("Unable to fetch image.  The web service may need a restart")
        except urllib.error.URLError as urlerror:
            log.error("Unable to fetch from URL: {}".format(url))
            log.error(urlerror)
            self.displayError("Unable to fetch image.  The web service may need a restart")

    def setStatus(self, occupant: str, roomName: str, presence: Presence):
        """
        Sets the status for an occupant based on the list of required occupants.
        Only those occupants required to be in the room will have status updated.
        :param roomName: The name of the MUC
        :param occupant: The name of the occupant without the room name
        :param presence: Presence.JOINED or Presence.LEFT
        """
        # TODO: Safe to remove
        # log.debug("Set status for {} in room {}".format(occupant, roomName))
        # # Walk through the list of required occupants
        # for requiredOccupant in self._requiredOccupants:
        #     # If an occupant is supposed to be there, update the status
        #     log.debug("Checking {} in room {} against {} {}".format(occupant, roomName, requiredOccupant.get("name"), requiredOccupant.get("room")))
        #     if requiredOccupant.get("name") == occupant and requiredOccupant.get("room") == roomName:
        #         # The occupant left or joined
        #         x = requiredOccupant.get("status")[0]
        #         y = requiredOccupant.get("status")[1]
        #         if presence == Presence.LEFT.name:
        #             self.statusTable.setItem(x,y,QtWidgets.QTableWidgetItem(constants.UI_STATUS_NOT_OK))
        #             item = self.statusTable.item(x,y)
        #             item.setBackground(QtGui.QColor("red"))
        #             item.setForeground(QtGui.QColor("white"))
        #             #requiredOccupant.get("status").setText(constants.UI_STATUS_NOT_OK)
        #             #requiredOccupant.get("status").setStyleSheet("color: white; background-color: red")
        #         else:
        #             self.statusTable.setItem(x, y, QtWidgets.QTableWidgetItem(constants.UI_STATUS_OK))
        #             item = self.statusTable.item(x, y)
        #             if item is not None:
        #                 item.setBackground(QtGui.QColor("green"))
        #                 item.setForeground(QtGui.QColor("black"))
        #             #requiredOccupant.get("status").setText(constants.UI_STATUS_OK)
        #             #requiredOccupant.get("status").setStyleSheet("color: white; background-color: green")
        #
        # self.statusTable.update()
        #
        # # Curious -- this is the only way to get the table to update from NOT OK to OK.  The other way works just fine.
        # self.statusTable.viewport().update()
        # #self.statusTable.repaint()
        #
        # # Note in the tabs if someone is missing who should be there
        # self.noteMissingEntities()


    def startOperation(self, operation: str, operationDescription: str):
        # Disable the start button and enable the stop
        try:
            self.button_start.setEnabled(False)
            self.button_start_imaging.setEnabled(False)
            self.button_stop.setEnabled(True)

            self.reset_kph.setEnabled(True)
            self.reset_distance.setEnabled(True)
            self.reset_images_taken.setEnabled(True)

            # Clear the image lists for the cameras
            self.images_right_intel.clear()
            self.images_right.clear()
            self.images_left_intel.clear()
            self.images_left.clear()

            self.status_current_operation.setText(operationDescription)
        except Exception as e:
            log.fatal(e)

        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.START.name
        now = datetime.datetime.now()

        # Construct the name for this session that is legal for AWS
        now -= datetime.timedelta(hours=7, minutes=0)

        timeStamp = now.strftime('%Y-%m-%d-%H-%M-%S-')
        sessionName = timeStamp + shortuuid.uuid()
        if arguments.location is not None:
            systemMessage.name = arguments.location + "-" + sessionName
        else:
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

    def resetProgress(self):
        self._distanceOverCapturedLength = 0
        # self.tractor_progress_left.setValue(0)
        # self.tractor_progress_right.setValue(0)

    def resetKPH(self):
        self.setSpeed(0.0)

    def resetImageCount(self):
        global treatments
        global treatmentsIntel
        global treatmentsBasler
        treatments = 0
        treatmentsBasler = 0
        treatmentsIntel = 0
        self.setTreatments(0, constants.Capture.RGB.name)
        self.setTreatments(0, constants.Capture.DEPTH_RGB.name)

    def resetDistance(self):
        self.absoluteDistance = self.currentDistance
        self.absolutePulses = self._currentPulses
        self.count_pulses.display(0.0)
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
        self.resetImageCount()
        self.resetProgress()
        log.debug("Stop Operation")

    def xmppError(self, state: str):
        log.debug("Informing user of XMPP state")
        if state == constants.OperationalStatus.FAIL.name:
            font = QFont()
            font.setFamily("Arial")
            font.setPointSize(20)
            self._dialogDisconnected.setText("Autoreconnect to message server in progress")
            self._dialogDisconnected.setFont(font)
            self._dialogDisconnected.setWindowTitle("Communication Problem")
            self._dialogDisconnected.setStandardButtons(QMessageBox.Ok)
            self._dialogDisconnected.setWindowFlag(Qt.WindowStaysOnTopHint)
            try:
                self._dialogDisconnected.exec_()
            except Exception as e:
                log.fatal(e)
        elif state == constants.OperationalStatus.OK.name:
            self._dialogDisconnected.setText("Connection restored")
        else:
            log.debug("Unknown state: ({})".format(str))

    def displayError(self, text):
        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.showMessage(text)
        choice = QMessageBox.critical(None,
                                      "Error",
                                      text,
                                      QMessageBox.Ok)

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
        self.taskMQ.processing = False

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

        self._taskOdometry = WorkerOdometry(self._odometryRoom, self._odometryMQCommunicator)
        self._taskOdometry.setAutoDelete(True)
        self._odometrySignals = self._taskOdometry.signals

        self._taskMQ = WorkerMQ(self._odometryRoom, self._odometryMQCommunicator)
        self._taskMQ.setAutoDelete(True)
        self._mqSignals = self._taskMQ.signals

        self._taskTreatment = WorkerTreatment(self._treatmentRoom)
        self._taskTreatment.setAutoDelete(True)
        self._treatmentSignals = self._taskTreatment.signals

        self._treatmentSignals.plan.connect(self.setTreatments)
        self._treatmentSignals.image.connect(self.addImage)
        self._treatmentSignals.xmppStatus.connect(self.xmppError)

        self._odometrySignals.progress.connect(self.updateProgress)
        self._odometrySignals.distance.connect(self.updateCurrentDistance)
        self._odometrySignals.pulses.connect(self.updatePulses)
        self._odometrySignals.speed.connect(self.updateCurrentSpeed)
        self._odometrySignals.latitude.connect(self.updateLatitude)
        self._odometrySignals.longitude.connect(self.updateLongitude)
        self._odometrySignals.agl.connect(self.updateAGL)
        self._odometrySignals.xmppStatus.connect(self.xmppError)

        self._mqSignals.progress.connect(self.updateProgress)
        self._mqSignals.distance.connect(self.updateCurrentDistance)
        self._mqSignals.pulses.connect(self.updatePulses)
        self._mqSignals.speed.connect(self.updateCurrentSpeed)
        self._mqSignals.latitude.connect(self.updateLatitude)
        self._mqSignals.longitude.connect(self.updateLongitude)
        self._mqSignals.agl.connect(self.updateAGL)
        self._mqSignals.xmppStatus.connect(self.xmppError)

        self._systemSignals.operation.connect(self.updateOperation)
        self._systemSignals.diagnostics.connect(self.updateStatusOfSystem)
        self._systemSignals.camera.connect(self.updateCamera)
        self._systemSignals.occupant.connect(self.setStatus)
        self._systemSignals.xmppStatus.connect(self.xmppError)


        pool.start(self._taskSystem)
        pool.start(self._taskOdometry)
        pool.start(self._taskTreatment)
        pool.start(self._taskMQ)





def process(conn, msg: xmpp.protocol.Message):
    global messageNumber
    global treatments
    global treatmentsIntel
    global treatmentsBasler

    msgText = msg.getBody()
    if msgText is not None:
        message = MUCMessage(raw=msg.getBody())
        timeMessageSent = message.timestamp
        timeDelta = (time.time() * 1000) - timeMessageSent
        if timeDelta > 5000:
            log.debug("Old message seen -- ignored")
            return

    if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY):
        log.debug("Processing Odometry XMPP message")
        odometryMessage = OdometryMessage(raw=msg.getBody())
        signals = window.taskOdometry.signals

        # The distance message
        if odometryMessage.type == constants.OdometryMessageType.DISTANCE.name:
            signals.distance.emit(odometryMessage.source, float(odometryMessage.totalDistance))
            signals.speed.emit(odometryMessage.source, float(odometryMessage.speed))
            signals.pulses.emit(odometryMessage.source, float(odometryMessage.pulses))
            signals.progress.emit(float(odometryMessage.distance))
            # See if we have lat/long.  Bad form here, as 0,0 is a legit value
            if odometryMessage.latitude != 0:
                signals.latitude.emit(float(odometryMessage.latitude))
                signals.longitude.emit(float(odometryMessage.longitude))
            else:
                signals.latitude.emit(0.0)
                signals.longitude.emit(0.0)

            log.debug("Speed: {:.02f}".format(odometryMessage.speed))

        # The position message
        elif odometryMessage.type == constants.OdometryMessageType.POSITION.name:
            signals.agl.emit(odometryMessage.depth)

        else:
            log.warning("Ignoring odometry message {}".format(msg.getBody()))


    elif msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT):
        treatmentMessage = TreatmentMessage(raw=msg.getBody())
        # Just the raw image from the camera
        if treatmentMessage.plan == constants.Treatment.RAW_IMAGE:
            treatments += 1
            signals = window.taskTreatment.signals
            position = treatmentMessage.position
            source = treatmentMessage.source
            if source == constants.Capture.RGB.name:
                treatmentsBasler += 1
                signals.plan.emit(treatmentsBasler, treatmentMessage.source)
            elif source == constants.Capture.DEPTH_RGB.name:
                treatmentsIntel += 1
                signals.plan.emit(treatmentsIntel, treatmentMessage.source)
            else:
                log.error("Unknown source for capture: {}".format(source))

            if len(treatmentMessage.url) > 0 and len(treatmentMessage.position) > 0:
                log.debug("Image from {} is at: {}".format(treatmentMessage.source, treatmentMessage.url))
                signals.image.emit(position, source, treatmentMessage.url)
        #window.Right(treatments)
        log.debug("Treatments: {:.02f}".format(treatments))
    elif msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM):
        systemMessage = SystemMessage(raw=msg.getBody())
        # Start the operation
        if systemMessage.action == constants.Action.START.name:
            signals = window.taskSystem.signals
            signals.operation.emit(systemMessage.operation, systemMessage.name)
        # Stop the operation
        if systemMessage.action == constants.Action.STOP.name:
            signals = window.taskSystem.signals
            signals.operation.emit(constants.Operation.QUIESCENT.name, systemMessage.name)
        if systemMessage.action == constants.Action.ACK.name:
            signals = window.taskSystem.signals
            signals.operation.emit(systemMessage.operation, systemMessage.name)
        if systemMessage.action == constants.Action.DIAG_REPORT.name:
            log.debug("Diagnostic report received for position {}".format(systemMessage.position))
            log.debug(systemMessage)
            signals = window.taskSystem.signals
            signals.diagnostics.emit(systemMessage)
            # signals.diagnostics.emit(systemMessage.position, systemMessage.diagnostics)
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

def startupMQCommunications(options: OptionsFile) -> ClientMQCommunicator:
    """
    Startup communications to the MQ server, but do not exchange messages
    :param options:
    :return: The client communicator
    """
    try:
        communicator = ClientMQCommunicator(SERVER=options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_SERVER),
                                            PORT=constants.PORT_ODOMETRY)
        communicator.connect()
    except KeyError:
        log.error("Unable to find {}/{} in ini file".format(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_SERVER))
        communicator = None
    return communicator

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

def processingStarted(weedSignals: WeedsSignals):
    """
    Used as a callback to indicate that processing has resumed.
    :param weedSignals: A signals object
    """
    weedSignals.xmppStatus.emit(constants.OperationalStatus.OK.name)


def processMessagesSync(room: MUCCommunicator, signals: WeedsSignals):
    # Connect to the XMPP server and process incoming messages
    # Curious. Suddenly fetching the occupants list does not work on windows tablet. Perhaps this is a version problem?

    # Originally
    #room.connect(True, False)

    room.processing = True

    while room.processing:
        try:
            # This method should never return unless something went wrong
            room.connect(True, False, processingStarted, signals)
            log.debug("Connected and processed messages, but encountered errors")
            # Send a signal that the XMPP connection has failed
            signals.xmppStatus.emit(constants.OperationalStatus.FAIL.name)
            time.sleep(5)
        except XMPPServerUnreachable:
            log.warning("Unable to connect and process messages.  Will retry and reinitialize.")
            time.sleep(5)
            room.processing = True
        except XMPPServerAuthFailure:
            log.fatal("Unable to authenticate using parameters")
            room.processing = False

    # This is the case where the server was not up to 3egin with
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
parser.add_argument('-loc', '--location', action="store", required=False, help="The location")


arguments = parser.parse_args()
options = OptionsFile(arguments.ini)

if not options.load():
    print("Failed to load options from {}.".format(arguments.ini))
    sys.exit(1)

if arguments.dns is not None:
    dnsServer = arguments.dns
else:
    try:
        dnsServer = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_DNS_SERVER)
    except KeyError:
        print("DNS Server must be specified either in INI file or on command line")
        sys.exit(1)

# Use a character set that amazon aws will accept
shortuuid.set_alphabet('0123456789abcdefghijklmnopqrstuvwxyz')

# Force resolutions to come from a server that has the entries we want
print("DNS: {}".format(dnsServer))
my_resolver = dns.resolver.Resolver(configure=False)
my_resolver.nameservers = [dnsServer]

answer = my_resolver.resolve('jetson.weeds.com')

logging.config.fileConfig(arguments.log)
log = logging.getLogger("console")

(odometryRoom, systemRoom, treatmentRoom) = startupCommunications(options)
odometryMQ = startupMQCommunications(options)

# Control the initialization with this semaphore
initializing = Semaphore()
initializing.acquire()

app = QtWidgets.QApplication(sys.argv)

window = MainWindow(initializing)
window.setupWindow()

dialogInit = DialogInit(3, window.initializationSignals)
dialogInit.show()

window.setWindowTitle("University of Arizona")
window.odometryMQ = odometryMQ
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


