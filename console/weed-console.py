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
from PyQt5 import QtGui


from OptionsFile import OptionsFile
import logging
import logging.config
import xmpp
import constants

from MUCCommunicator import MUCCommunicator
from Messages import MUCMessage, OdometryMessage, SystemMessage, TreatmentMessage
import uuid

from PyQt5 import QtWidgets

from console import Ui_MainWindow

class Presence(Enum):
    JOINED = 0
    LEFT = 1

class Status(Enum):
    OK = 0
    ERROR = 1

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # Wire up the buttons
        self.button_start.clicked.connect(self.startWeeding)
        self.button_start_imaging.clicked.connect(self.startImaging)
        self.button_stop.clicked.connect(self.stopWeeding)

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

        self._requiredOccupants = list()




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
             "status": self.odometry_status_rio},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": self.odometry_status_left},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": self.odometry_status_right},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": self.system_status_rio},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": self.system_status_left},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": self.system_status_right},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CLOUD),
             "status": self.system_status_aws},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
             "status": self.treatment_status_rio},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_1),
             "status": self.treatment_status_left},

            {"room": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
             "name": options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_JETSON_2),
             "status": self.treatment_status_right}
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
        # This will split up a JID of the form <room-name>@<conference-name>.<domain>.<domain>/<nickname>
        log.debug("System room has {} occupants".format(len(self._systemRoom.occupants)))
        log.debug("Odometry room has {} occupants".format(len(self._odometryRoom.occupants)))
        log.debug("Treatment room has {} occupants".format(len(self._treatmentRoom.occupants)))
        allOccupants = self._systemRoom.occupants + self._odometryRoom.occupants + self._treatmentRoom.occupants
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

        #self.initializing.setStyleSheet("color: white; background-color: green")
        self.initializing.setValue(0)

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

    def setTabColor(self, tab:QtWidgets, status:Status):
        color = Qt.white

        if status == Status.OK:
            color = Qt.white
        elif status == Status.ERROR:
            color = Qt.red

        p = self.tabSystem.palette()
        p.setColor(tab.backgroundRole(), color)
        self.tabWidget.setPalette(p)

    def setSpeed(self, speed: float):
        self.average_kph.display(speed)

    def setDistance(self, distance: float):
        self.count_distance.display(distance)

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
                if presence == Presence.LEFT:
                    requiredOccupant.get("status").setText(constants.UI_STATUS_NOT_OK)
                    requiredOccupant.get("status").setStyleSheet("color: white; background-color: red")
                else:
                    requiredOccupant.get("status").setText(constants.UI_STATUS_OK)
                    requiredOccupant.get("status").setStyleSheet("color: white; background-color: green")



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
        systemMessage.name = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + str(uuid.uuid4())
        self._systemRoom.sendMessage(systemMessage.formMessage())

    def startImaging(self):

        self.startOperation(constants.UI_OPERATION_IMAGING)

    def resetKPH(self):
        self.setSpeed(0.0)

    def resetImageCount(self):
        pass

    def resetDistance(self):
        self.setDistance(0.0)

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
        systemMessage.name = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_PREFIX) + str(uuid.uuid4())
        self._systemRoom.sendMessage(systemMessage.formMessage())


    def stopWeeding(self):
        # Enable the start button and disable the stop
        self.button_start.setEnabled(True)
        self.button_start_imaging.setEnabled(True)
        self.button_stop.setEnabled(False)

        self.reset_kph.setEnabled(False)
        self.reset_distance.setEnabled(False)
        self.reset_images_taken.setEnabled(False)

        systemMessage = SystemMessage()

        systemMessage.action = constants.Action.STOP
        #systemMessage.name = str(uuid.uuid4())
        self._systemRoom.sendMessage(systemMessage.formMessage())
        log.debug("Stop Weeding")


messageNumber = 0
treatments = 0

def process(conn, msg: xmpp.protocol.Message):
    global messageNumber
    global treatments

    if msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY):
        odometryMessage = OdometryMessage(raw=msg.getBody())
        window.setSpeed(odometryMessage.speed)
        window.setDistance(odometryMessage.totalDistance)
        #log.debug("Speed: {:.02f}".format(odometryMessage.speed))
    elif msg.getFrom().getStripped() == options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT):
        treatmentMessage = TreatmentMessage(raw=msg.getBody())
        treatments += 1
        window.setTreatments(treatments)
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
                                   presenceCB)

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
                                 presenceCB)

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
                                     presenceCB)

    return (odometryRoom, systemRoom, treatmentRoom)

def processMessages(room: MUCCommunicator):
    # Connect to the XMPP server and just return
    room.connect(False, True)

def processMessagesSync(room: MUCCommunicator):
    # Connect to the XMPP server and just return
    room.connect(True, True)

def housekeeping(room: MUCCommunicator):
    while not systemRoom.connected:
        log.debug("Waiting for system room connection.")
        sleep(5)
    window.initializing.setValue(100)
    window.button_start.setEnabled(True)
    window.button_start_imaging.setEnabled(True)
    window.reset_kph.setEnabled(True)
    window.reset_distance.setEnabled(True)
    window.reset_images_taken.setEnabled(True)
    window.setTabColor(window.tabSystem, Status.OK)
    window.status_current_operation.setText(constants.UI_OPERATION_NONE)

threads = list()

parser = argparse.ArgumentParser("Weeding Console")

parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument('-l', '--log', action="store", required=False, default="logging.ini", help="Logging INI")
parser.add_argument('-d', '--dns', action="store", required=False, help="DNS server address")

arguments = parser.parse_args()

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
window.setWindowTitle("University of Arizona")
window.odometryRoom = odometryRoom
window.systemRoom = systemRoom
window.treatmentRoom = treatmentRoom
window.setupRooms(odometryRoom, systemRoom, treatmentRoom)

log.debug("Start housekeeping thread")
houseThread = threading.Thread(name=constants.THREAD_NAME_SYSTEM,target=housekeeping,args=(systemRoom,))
houseThread.daemon = True
threads.append(houseThread)
houseThread.start()

log.debug("Start system thread")
sysThread = threading.Thread(name=constants.THREAD_NAME_SYSTEM,target=processMessagesSync,args=(systemRoom,))
sysThread.daemon = True
threads.append(sysThread)
sysThread.start()

log.debug("Start odometry thread")
sysThread = threading.Thread(name=constants.THREAD_NAME_ODOMETRY,target=processMessagesSync,args=(odometryRoom,))
sysThread.daemon = True
threads.append(sysThread)
sysThread.start()

log.debug("Start treatment thread")
sysThread = threading.Thread(name=constants.THREAD_NAME_TREATMENT,target=processMessagesSync,args=(treatmentRoom,))
sysThread.daemon = True
threads.append(sysThread)
sysThread.start()





#window.setStatus()
window.show()
window.setInitialState()
app.exec()
