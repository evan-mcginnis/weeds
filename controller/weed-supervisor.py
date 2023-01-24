#
# W E E D S  C O N T R O L L E R
#
# This is a service that will start and stop the weeding operations
#

import argparse
import datetime
import glob
import platform
import sys
import threading
import time
from typing import Callable

import logging
import logging.config
import shortuuid
import shutil

import numpy as np
import xmpp
import os
# from xmpp import protocol

# This does not work
# from CameraFile import CameraFile, CameraBasler

from OptionsFile import OptionsFile
from MUCCommunicator import MUCCommunicator
from Messages import OdometryMessage, SystemMessage, TreatmentMessage
from WeedExceptions import XMPPServerUnreachable, XMPPServerAuthFailure
from CameraDepth import CameraDepth
from RealSense import RealSense

import constants

# Two consecutive readings should be within this percentage of each other
DEPTH_THRESHOLD = 0.75

parser = argparse.ArgumentParser("Weed system supervisor")

parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
arguments = parser.parse_args()

def startupDepthCamera(options: OptionsFile) -> CameraDepth:
    """
    Starts the attached depth camera
    :return: The depth camera instance or None if the camera cannot be found.
    """
    sensors = RealSense()
    sensors.query()
    markSensorAsFailed = False

    if sensors.count() < 1:
        log.error("Detected {} depth/IMU sensors. Expected at least 1.".format(sensors.count()))
        log.error("No sensor will be used.")
        markSensorAsFailed = True

    # Start the Depth Cameras
    try:
        cameraForDepth = CameraDepth(constants.Capture.DEPTH_RGB)
                                     #serial=options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_SERIAL_LEFT))
        if markSensorAsFailed:
            cameraForDepth.state.toMissing()
        else:
            cameraForDepth.state.toIdle()
    except KeyError:
        log.error("Unable to find serial number for depth camera: {}/{} & {}".format(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_SERIAL_LEFT, constants.PROPERTY_SERIAL_RIGHT))
        cameraForDepth = None

    return cameraForDepth


#
# X M P P   C O M M U N I C A T I O N S
#
# def process(conn,msg):# xmpp.protocol.Message):
#     log.debug("Callback for distance")
#     return

def startupCommunications(options: OptionsFile, callbackOdometer: Callable, callbackSystem: Callable, callbackTreatment: Callable) -> ():
    """

    :param options:
    :param callbackOdometer:
    :param callbackSystem:
    :return:
    """
    # The room that will get the announcements about forward or backward progress
    odometryRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_CONTROL),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONTROL),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
                                   callbackOdometer,
                                   None)

    # The room that will get status reports about this process
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_CONTROL),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONTROL),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
                                 callbackSystem,
                                 None)

    treatmentRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_CONTROL),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONTROL),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
                                    callbackTreatment,
                                    None)
    # print("XMPP communications started")

    return odometryRoom, systemRoom, treatmentRoom
#
# L O G G E R
#

def startupLogger(outputDirectory: str):
    """
    Initializes two logging systems: the image logger and python centric logging.
    :param outputDirectory: The output directory for the images
    :return: The image logger instance
    """

    # Initialize logging
    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("controller")

    return log

def readINI() -> OptionsFile:
    options = OptionsFile(arguments.ini)
    options.load()
    return options


def sendCurrentOperation(systemRoom: MUCCommunicator):
    systemMessage = SystemMessage()
    systemMessage.action = constants.Action.ACK.name
    systemMessage.operation = currentOperation
    try:
        position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
        systemMessage.position = position
    except KeyError:
        log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))

    systemRoom.sendMessage(systemMessage.formMessage())

def runDiagnostics(systemRoom: MUCCommunicator):
    """
    Run diagnostics for this subsystem, collecting information about the camera connectivity.
    :param systemRoom: The room to send the results
    """
    pass
    # systemMessage = SystemMessage()
    # systemMessage.action = constants.Action.DIAG_REPORT.name
    # systemMessage.diagnostics =  camera.status.name
    # systemMessage.gsdCamera = camera.gsd
    # try:
    #     position = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION)
    #     systemMessage.position = position
    # except KeyError:
    #     log.error("Can't find {}/{} in ini file".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_POSITION))
    # systemMessage.statusCamera = camera.status.name
    #
    # systemRoom.sendMessage(systemMessage.formMessage())

totalMovement = 0.0
keepAliveMessages = 0
movementSinceLastProcessing = 0.0
#
# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageSystemCB(conn,msg: xmpp.protocol.Message):
    global logger
    global processing
    global outputDirectory
    global currentSessionName
    global currentOperation
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
            body = msg.getBody()
            # Check if this is a real message and not just an empty keep-alive message
            if body is not None:
                log.debug("system message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
                # TODO: Check to see if this is my own message
                # systemMessage = SystemMessage(raw=msg.getBody())
                # if systemMessage.action == constants.Action.START.name:
                #     processing = True
                #     currentSessionName = systemMessage.name
                #     currentOperation = systemMessage.operation
                #     outputDirectory = arguments.output + "/" + currentSessionName
                #     log.debug("Begin processing to: {}".format(outputDirectory))
                #     logger = Logger()
                #     if not logger.connect(outputDirectory):
                #         log.error("Unable to connect to logging. {} does not exist.".format(outputDirectory))
                # if systemMessage.action == constants.Action.STOP.name:
                #     log.debug("----- Stop weeding ------")
                #     currentOperation = constants.Operation.QUIESCENT.name
                #     #postWeedingCleanup()
                # if systemMessage.action == constants.Action.CURRENT.name:
                #     sendCurrentOperation(roomSystem)
                # if systemMessage.action == constants.Action.START_DIAG.name:
                #     log.debug("Request for diagnostics")
                #     runDiagnostics(roomSystem, camera)

    elif msg.getType() == "chat":
        print("private: " + str(msg.getFrom()) + ":" + str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

#
# The callback for messages received in the odometry room.
# When the total distance is the width of the image, grab an image and process it.
#

def messageOdometryCB(conn, msg: xmpp.protocol.Message):
    global totalMovement
    global keepAliveMessages
    global movementSinceLastProcessing
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
        body = msg.getBody()
        # Check if this is a real message and not just an empty keep-alive message
        if body is not None:
            log.debug("Distance message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
            odometryMessage = OdometryMessage(raw=body)
            log.debug("Message: {}".format(odometryMessage.data))
            totalMovement += odometryMessage.distance
            movementSinceLastProcessing += odometryMessage.distance
            # The time of the observation
            timeRead = odometryMessage.timestamp
            # Determine how old the observation is
            # The version of python on the jetson does not support time_ns, so this a bit of a workaround until I
            # get that sorted out.  Just convert the reading to milliseconds for now
            #timeDelta = (time.time() * 1000) - (timeRead / 1000000)
            timeDelta = (time.time() * 1000) - timeRead
            log.debug("Total movement: {} at time: {}. Movement since last acquisition: {} Time now is {} delta from now {} ms".
                      format(totalMovement, timeRead, movementSinceLastProcessing, time.time() * 1000, timeDelta))


        else:
            # There's not much to do here for keepalive messages
            keepAliveMessages += 1
            # print("weeds: keepalive message from chatroom")
    elif msg.getType() == "chat":
        print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))

# The callback for messages received in the system room.
# When the total distance is the width of the image, grab an image and process it.
#
def messageTreatmentCB(conn,msg: xmpp.protocol.Message):
    # Make sure this is a message sent to the room, not directly to us
    if msg.getType() == "groupchat":
        body = msg.getBody()
        # Check if this is a real message and not just an empty keep-alive message
        if body is not None:
            log.debug("treatment message from {}: [{}]".format(msg.getFrom(), msg.getBody()))
    elif msg.getType() == "chat":
        print("private: " + str(msg.getFrom()) +  ":" +str(msg.getBody()))
    else:
        log.error("Unknown message type {}".format(msg.getType()))


#
#
# This method will never return.  Connect and start processing messages
#
def processMessages(communicator: MUCCommunicator):
    """
    Process messages for the chatroom -- note that this routine will never return.
    :param communicator: The chatroom communicator
    """
    log.info("Connecting to chatroom")
    processing = True

    while processing:
        try:
            communicator.connect(True)
            log.debug("Connected and processed messages")
        except XMPPServerUnreachable:
            log.warning("Unable to connect and process messages.  Will retry.")
            time.sleep(5)
            processing = True
        except XMPPServerAuthFailure:
            log.fatal("Unable to authenticate using parameters")
            processing = False


def takeDepthImages(camera: CameraDepth):

    cameraConnected = False

    if camera is None:
        log.error("Depth camera is not created")
        return cameraConnected

    # Connect to the camera and take an image
    log.debug("Connecting to depth camera")
    cameraConnected = camera.connect()

    if cameraConnected:
        if isinstance(camera, CameraDepth):
            camera.state.toClaim()
            camera.initialize()

            if camera.initializeCapture():
                try:
                    camera.startCapturing()
                except IOError as io:
                    camera.log.error(io)
                rc = 0
            else:
                rc = -1
        else:
            log.debug("Not a depth camera")
            camera.startCapturing()
    else:
        log.error("Unable to connect to depth camera")

def startOperation(operation: str, operationDescription: str):
    log.debug("Starting operation")
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
    roomSystem.sendMessage(systemMessage.formMessage())

def stopOperation():
    log.debug("Stopping operation")
    systemMessage = SystemMessage()

    systemMessage.action = constants.Action.STOP.name
    systemMessage.timestamp = time.time() * 1000
    roomSystem.sendMessage(systemMessage.formMessage())

def sendPosition(heightAGL: float):
    """
    Send position information to the odometry room.
    :param heightAGL: The average height AGL of the camera plate
    """
    log.debug("Send current status information")
    odometryMessage = OdometryMessage()
    odometryMessage.type = constants.OdometryMessageType.POSITION
    odometryMessage.depth = heightAGL
    odometryMessage.timestamp = time.time() * 1000
    roomOdometry.sendMessage(odometryMessage.formMessage())

def supervisor(camera: CameraDepth):
    log.debug("Beginning supervision")

    # Use a character set that amazon aws will accept
    shortuuid.set_alphabet('0123456789abcdefghijklmnopqrstuvwxyz')
    try:
        aglUp = int(options.option(constants.PROPERTY_SECTION_DEPTH, constants.PROPERTY_AGL_UP))
        aglDown = int(options.option(constants.PROPERTY_SECTION_DEPTH, constants.PROPERTY_AGL_DOWN))
        aglStop = int(options.option(constants.PROPERTY_SECTION_DEPTH, constants.PROPERTY_AGL_STOP))
    except KeyError as key:
        log.error("Unable to find depths in ini: {}/{} {} {}".format(constants.PROPERTY_SECTION_DEPTH, constants.PROPERTY_AGL_UP, constants.PROPERTY_AGL_DOWN, constants.PROPERTY_AGL_STOP))
        return

    weederPosition = constants.Orientation.UNKNOWN
    weederPreviousPosition = constants.Orientation.UNKNOWN
    weederPreviousOperation = constants.Operation.QUIESCENT

    armed = False

    while True:
        if camera.connected:
            time.sleep(5)
            # This is a hack until I can determine what is happening here. Occasionally there is an odd reading
            # where the depth is off by an order of magnitude.  The second reading is fine.
            depthImage = camera.capture()
            averageAGL = float(np.average(depthImage.depth))
            depthImage2 = camera.capture()
            averageAGL2 = float(np.average(depthImage2.depth))

            #log.debug("Average height: {}".format(averageAGL))
            with open('height.csv', 'a') as the_file:
                the_file.write("{}".format(averageAGL))

            # If the two numbers are more than X % apart, pass on this reading.
            # As soon as we get readings that are pretty close to each other, accept them as legitimate

            if (averageAGL != 0 and averageAGL2 != 0) and ((min(averageAGL, averageAGL2) / max(averageAGL, averageAGL2)) < DEPTH_THRESHOLD):
                log.debug("Depth readings not within threshold ({}): {} and {}".format(DEPTH_THRESHOLD, averageAGL, averageAGL2))
                continue

            sendPosition(averageAGL)

            if averageAGL > aglUp:
                # Down-UP
                if weederPreviousPosition == constants.Orientation.DOWN:
                    if weederPreviousOperation != constants.Operation.QUIESCENT:
                        stopOperation()
                    weederPreviousOperation = constants.Operation.QUIESCENT
                    # If we see a down, we are ready to begin an operation
                    armed = True
                weederPreviousPosition = constants.Orientation.UP
            elif averageAGL > aglDown:
                # Up->Down
                if weederPreviousPosition == constants.Orientation.UP:
                    if armed:
                        startOperation(constants.Operation.IMAGING.name, constants.UI_OPERATION_IMAGING)
                        weederPreviousOperation = constants.Operation.IMAGING
                        armed = False
                weederPreviousPosition = constants.Orientation.DOWN
            elif averageAGL > aglStop:
                if weederPreviousOperation == constants.Operation.IMAGING:
                    stopOperation()
                weederPreviousOperation = constants.Operation.QUIESCENT
        else:
            log.debug("Camera is not yet connected")

        time.sleep(0.5)

# This writes out the status into an HTML file
# TODO: Make this a bit more elegant and process the GET operations
def statusDaemon():

    processing = True

    while processing:
        with open("index.html", "w") as statusFile:
            statusFile.write("<html>\n<head>\n<title> \nOutput Data in an HTML file \
            </title>\n</head> <body><h1>Welcome to <u>GeeksforGeeks</u></h1>\
            \n<h2>A <u>CS</u> Portal for Everyone</h2> \n</body></html>")
        time.sleep(5)

#
# Start up various subsystems
#
options = readINI()

currentSessionName = ""
currentOperation = constants.Operation.QUIESCENT.name

# Confirm the INI exists
if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(1)

# Initialize logging
logging.config.fileConfig(arguments.logging)
log = logging.getLogger("supervisor")

depthCamera = startupDepthCamera(options)

if depthCamera is not None:
    log.info("Depth camera started")
else:
    log.error("Unable to start depth camera")

(roomOdometry, roomSystem, roomTreatment) = startupCommunications(options, messageOdometryCB, messageSystemCB, messageTreatmentCB)
log.debug("Communications started")

# Start the worker threads, putting them in a list
threads = list()

log.debug("Start supervisor")
supervisor = threading.Thread(name=constants.THREAD_NAME_SUPERVISOR, target=supervisor, args=(depthCamera,))
threads.append(supervisor)
supervisor.start()

log.debug("Start depth data acquisition")
acquire = threading.Thread(name=constants.THREAD_NAME_ACQUIRE, target=takeDepthImages, args=(depthCamera,))
threads.append(acquire)
acquire.start()

log.debug("Starting odometry receiver")
#generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=processMessages, args=(roomOdometry,))
generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=roomOdometry.processMessages, args=())
generator.daemon = True
threads.append(generator)
generator.start()

log.debug("Starting system receiver")
#sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=processMessages, args=(roomSystem,))
sys = threading.Thread(name=constants.THREAD_NAME_SYSTEM, target=roomSystem.processMessages, args=())
sys.daemon = True
threads.append(sys)
sys.start()

# log.debug("Starting treatment thread")
# #treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=processMessages, args=(roomTreatment,))
# treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=roomTreatment.processMessages, args=())
# treat.daemon = True
# threads.append(treat)
# treat.start()

# Wait for the workers to finish
while True:
    for index, thread in enumerate(threads):
        if not thread.is_alive():
            log.error("Thread {} exited. This is not normal".format(thread.name))
            os._exit(-1)
    time.sleep(60)





