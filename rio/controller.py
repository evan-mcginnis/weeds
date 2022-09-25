#
# R I O  C O N T R O L L E R
#
#
# This is the software resident on the National Instruments RIO controller
# It expects two things when operating in a physical setting.
# 1) A digital IO card for the odometer wheel
# 2) A digital IO card for the emitter
#
# These two bits can be simulated.
# If the odometer is simulated, you need to supply a rate of travel
#

import argparse
import os
import threading
import time
import sys
from typing import Callable
from OptionsFile import OptionsFile
import logging
import logging.config
import xmpp

try:
    import nidaqmx as ni
    virtualRequired = False
except ImportError:
    virtualRequired = True

from statemachine.exceptions import TransitionNotAllowed

import constants
from Odometer import Odometer, VirtualOdometer
from PhysicalOdometer import PhysicalOdometer
from Emitter import Emitter, PhysicalEmitter, VirtualEmitter
from MUCCommunicator import MUCCommunicator
from Messages import MUCMessage, OdometryMessage, SystemMessage
from GPSClient import GPS
from CameraDepth import CameraDepth



parser = argparse.ArgumentParser("RIO Controller")

parser.add_argument('-e', '--emitter', action="store_true", required=False, default=False,help="Virtual emitter mode")
parser.add_argument('-o', '--odometer', action="store_true", required=False, default=False, help="Virtual Odometry Mode")
parser.add_argument('-a', '--odometer_line_a', action="store", required=False, default="", help="Odometer line A")
parser.add_argument('-b', '--odometer_line_b', action="store", required=False, default="", help="Odometer line B")
parser.add_argument('-p', '--plan', action="store_true", required=False, default=False, help="Generate treatment plan")
parser.add_argument('-r', '--rio', action="store_true", required=False, default=False, help="Virtual RIO mode")
parser.add_argument('-s', '--speed', action="store", required=False, default=4, type=float, help="Virtual odometry speed (cm/s)")
parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME, help="Options INI")
parser.add_argument('-l', '--log', action="store", required=False, default="logging.ini", help="Logging INI")

arguments = parser.parse_args()

options = OptionsFile(arguments.ini)
if not options.load():
    print("Failed to load options from {}.".format(arguments.ini))
    sys.exit(1)
# Use -r -o -e when running somewhere other than a RIO

logging.config.fileConfig(arguments.log)
log = logging.getLogger("rio")

# Type of odometry used
if arguments.odometer:
    typeOfOdometry = constants.SubsystemType.VIRTUAL
else:
    typeOfOdometry = constants.SubsystemType.PHYSICAL

# Status of threads
statusOdometry = constants.Status.QUIESCENT
statusTreatment = constants.Status.QUIESCENT

# The current session & operation
currentSessionName = ""
currentOperation = constants.Operation.QUIESCENT.name

def sendSessionInformation(options: OptionsFile) -> bool:
    """
    Send the current operation information
    :param options:
    """
    systemMessage = SystemMessage()

    # The response to a query is just an ACK
    systemMessage.action = constants.Action.ACK.name
    # If there is no operation currently in progress, the name will be a static value
    systemMessage.name = currentSessionName
    systemMessage.operation = currentOperation
    messageText = systemMessage.formMessage()
    systemRoom.sendMessage(messageText)

    return True

def startSession(options: OptionsFile, sessionName: str) -> bool:
    """
    Prepare for a new session. The current working directory is set after the session name
    :param sessionName:
    """
    global log
    global currentSessionName
    global currentOperation

    started = False

    currentSessionName = sessionName
    currentOperation = constants.Operation.IMAGING.name
    path = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_OUTPUT) + "/" + sessionName

    try:
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        started = True
    except OSError as e:
        log.critical("Unable to prepare and set directory to {}".format(path))
        log.critical("Raw: {}".format(e))

    try:
        camera.state.toClaim()
    except TransitionNotAllowed as transition:
        log.critical("Unable to transition the camera to claim")
        log.critical(transition)

    log.debug("Session started")
    return started

def endSession(options: OptionsFile, name: str) -> bool:
    global log
    stopped = False

    path = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_ROOT) + "/output/" + name
    root = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_ROOT)

    # Move the log file over to the output directory
    try:
        source = root + "/rio/rio.log"
        destination = path + "/rio.log"
        os.rename(source, destination)
    except OSError as oserr:
        log.critical("Unable to move {} to {}".format(source, destination))
        log.critical(oserr)

    # Write out the stats for the session.
    log.debug("End session")
    try:
        finished = options.option(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_FILENAME_FINISHED)
        log.debug("Writing session statistics to: {}".format(finished))
    except KeyError as key:
        log.critical("Could not find {}/{} in {}".format(constants.PROPERTY_SECTION_GENERAL, constants.PROPERTY_FILENAME_FINISHED, options.filename))

    try:
        with open(finished, 'w') as fp:
            fp.write("Session complete")
    except IOError as e:
        log.error("Unable to write out end of run data to file: {}".format(finished))
        log.error("{}".format(e))

    camera.stop()
    # Change back to the general output directory not associated with any session
    try:
        os.chdir(path)
        stopped = True
    except OSError as e:
        log.critical("Unable to prepare and set directory to {}".format(path))
        log.critical("Raw: {}".format(e))

    # try:
    #     camera.state.toStop()
    # except TransitionNotAllowed as transition:
    #     log.critical("Unable to transition the camera to idle")
    #     log.critical(transition)



    log.debug("Session ended")
    return stopped

#
# N I  D I A G N O S T I C S
#
def diagnostics():
    # TODO: NI Diagnostics
    return True, "No diagnostics implemented"

#
# Start up system components and run diagnostics
#
def startupSystem():
    system = ni.system.System.local()
    #system.driver_version
    #devices = system.devices
    for device in system.devices:
        log.debug(" Found: {}".format(device))
        #print(device)
    # channels = ni.system.physical_channel.PhysicalChannel("Mod3")
    # print(channels)

    # Run diagnostics on the NI system
    diagnosticResult, diagnosticText = diagnostics()
    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)

    # Connect to the emitter and run diagnostics
    # Here, we assume that there is only one emitter, which is not a good assumption

    # V I R T U A L  E M I T T E R  O R  P H Y S I C A L  E M I T T E R

    # Here, we associated a module on the RIO with the emitter. I suppose this needs to be down
    # to a set of pins on that card as well, but this will do for now

    try:
        rightModuleName = options.option(constants.PROPERTY_SECTION_RIO, constants.PROPERTY_RIGHT)
        leftModuleName = options.option(constants.PROPERTY_SECTION_RIO, constants.PROPERTY_LEFT)
    except KeyError as key:
        log.fatal("Unable to find DAQ module location in options file.")
        log.fatal("Raw: {}".format(key))
        sys.exit(-1)

    if arguments.emitter:
        rightEmitter = VirtualEmitter(rightModuleName)
        leftEmitter = VirtualEmitter(leftModuleName)
    else:
        rightEmitter = PhysicalEmitter(rightModuleName)
        leftEmitter = PhysicalEmitter(leftModuleName)

    diagnosticResultRightEmitter, diagnosticTextRightEmitter = rightEmitter.diagnostics()
    diagnosticResultLeftEmitter, diagnosticTextLeftEmitter = leftEmitter.diagnostics()

    if not diagnosticResultRightEmitter:
        log.warning(diagnosticTextRightEmitter)
        rightEmitter = None

    if not diagnosticResultLeftEmitter:
        log.warning(diagnosticTextLeftEmitter)
        leftEmitter = None

    return system, rightEmitter, leftEmitter

#
# P R O C E S S
#
# For the odometer, there isn't much to do, as it just reports movement.
# For the treatement, there are a few things: listen for identifications, and for begin/end cycles
#
def process(conn, msg: xmpp.protocol.Message):
    global currentSessionName
    global currentOperation
    log.debug("Process message from {}".format(msg.getFrom()))

    # Messages from the system room will be in the form: system@conference.weeds.com/console

    console = options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM) \
              +  "/" \
              + options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONSOLE)

    if msg.getFrom() == console:
        if msg.getBody() is not None:
            systemMsg = SystemMessage(raw=msg.getBody())
            timeMessageSent = systemMsg.timestamp
            timeDelta = (time.time() * 1000) - timeMessageSent
            log.debug("Action: {} Time Delta: {}".format(systemMsg.action, timeDelta))

            # Determine if this is an old message or not
            if timeDelta < 5000:
                # S T A R T
                # For a start session, we need to at start logging with the name of the session
                if systemMsg.action == constants.Action.START.name:
                    currentSessionName = systemMsg.name
                    log.debug("Start session {}".format(systemMsg.name))
                    startSession(options, systemMsg.name)
                    currentOperation = constants.Operation.IMAGING.name
                # This is just an alive message, so respond
                if systemMsg.action == constants.Action.PING.name:
                    log.debug("PING")
                if systemMsg.action == constants.Action.STOP.name:
                    log.debug("Stopping odometry session")
                    endSession(options, currentSessionName)
                    currentOperation = constants.Operation.QUIESCENT.name
                    currentSessionName = ""
                if systemMsg.action == constants.Action.CURRENT.name:
                    log.debug("Query for current operation")
                    sendSessionInformation(options)
            else:
                log.debug("Old system message. Sent: {} Now: {} Delta: {}".format(timeMessageSent, time.time() * 1000, timeDelta))

def processOdometry(conn, msg: xmpp.protocol.Message):
    global typeOfOdometry

    log.debug("Process message from {}".format(msg.getFrom()))

    # Messages from the system room will be in the form: system@conference.weeds.com/console

    console = options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM) \
              +  "/" \
              + options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONSOLE)

    if msg.getFrom() == console:
        if msg.getBody() is not None:
            odometryMsg = OdometryMessage(raw=msg.getBody())
            log.debug("Action is {}".format(odometryMsg.action))

            # C O N F I G U R E
            if odometryMsg.action == constants.Action.CONFIGURE.name:
                source = odometryMsg.source
                oldTypeOfOdometry = typeOfOdometry
                log.debug("Configure source as {}".format(source))
                if source == constants.SubsystemType.PHYSICAL.name:
                    typeOfOdometry = constants.SubsystemType.PHYSICAL
                elif source == constants.SubsystemType.VIRTUAL.name:
                    typeOfOdometry = constants.SubsystemType.VIRTUAL
                else:
                    log.error("Unknown odometry type. Ignored")

                if oldTypeOfOdometry != typeOfOdometry:
                    log.debug("Restart of odometry required")
                    odometer.stop()

            # # This is just an alive message, so respond
            # if systemMsg.action == constants.Action.PING.name:
            #     log.debug("PING")
            # if systemMsg.action == constants.Action.STOP.name:
            #     log.debug("Stopping odometry")
            #     endSession(options)


def startupCommunications(options: OptionsFile) -> ():
    """
    Start communications with three MUCs: odometry, system, and treatment
    :param options: The options that contain an XMPP section with room names and nicknames
    :return: odometry, system rooms
    """
    # The room that will get the announcements about forward or backward progress
    odometryRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_RIO),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
                                   processOdometry,
                                   None) # Don't care about presence

    # The room that will get status reports about this process
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_RIO),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
                                 process,
                                 None) # Don't care about presence

    treatmentRoom = MUCCommunicator( options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_RIO),
                                     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
                                     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                     options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT),
                                     process,
                                     None) # Don't care about presence

    return (odometryRoom, systemRoom, treatmentRoom)

#
# G P S
#
def startupGPS() -> GPS:
    theGPS = GPS()
    theGPS.connect()
    if theGPS.isAvailable():
        packet = theGPS.getCurrentPosition()
        log.debug("GPS Position: {}".format(packet.position()))
        log.debug("GPS Error: {}".format(packet.position_precision()))
        log.debug("GPS Fix: {}".format(packet.mode))
    else:
        log.warning("GPS location is not available. Image exif will not include this information")
    return theGPS

#
# D E P T H  C A M E R A
#
def startupDepthCamera() -> CameraDepth:
    camera = CameraDepth(gyro=constants.PARAM_FILE_GYRO, acceleration=constants.PARAM_FILE_ACCELERATION)

    camera.state.toIdle()
    return camera

#
# O D O M E T R Y
#

# TODO: refactor the parameters to create the encoder so this is not needed
def callback():
    return

def startupOdometer(typeOfOdometer: constants.SubsystemType) -> Odometer:
    """
    Start up the odometry subsystem and run diagnostics
    :param imageProcessor: The image processor executed at each interval
    :return:
    """
    pulsesPerRotation = 0
    wheelSize = 0

    try:
        # V I R T U A L  O D O M E T E R
        #
        # This creates an odometer that will simulate moving forward with a given speed and will
        # call the treatment controller at set intervals
        if typeOfOdometer == constants.SubsystemType.VIRTUAL:
            log.debug("Using virtual odometry")
            pulsesPerRotation = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PPR))
            wheelSize = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_WHEEL_CIRCUMFERENCE))
            odometer = VirtualOdometer(WHEEL_SIZE=wheelSize, PULSES=pulsesPerRotation, SPEED=arguments.speed)
        else:
            # Get the A & B lines -- either from the INI file or on the command line
            if len(arguments.odometer_line_a) == 0:
                lineA = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PFI_A)
            else:
                lineA = arguments.odometer_line_a

            if len(arguments.odometer_line_b) == 0:
                lineB = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PFI_B)
            else:
                lineB = arguments.odometer_line_b

            if pulsesPerRotation == 0:
                try:
                    pulsesPerRotation = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PPR))
                except KeyError:
                    print("Pulses Per Rotation must be specified as command line option or in the INI file.")
            if wheelSize == 0:
                try:
                    wheelSize = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_WHEEL_CIRCUMFERENCE))
                except KeyError:
                    print("Wheel Size must be specified as command line option or in the INI file.")
            # The lines specified could not be found in either the INI or on the command line
            if len(lineA) == 0 or len(lineB) == 0:
                print(constants.MSG_LINES_NOT_SPECIFIED)
                log.error(constants.MSG_LINES_NOT_SPECIFIED)
                sys.exit(1)

            # Is forward clockwise or counter-clockwise?
            forward = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_FORWARD)

            # The counter used to read the rotations
            theCounter = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_COUNTER)
            #odometer = PhysicalOdometer(lineA, lineB, wheelSize, pulsesPerRotation, 0, callback)
            odometer = PhysicalOdometer(LINE_A = lineA,
                                        LINE_B = lineB,
                                        COUNTER = theCounter,
                                        WHEEL_SIZE = wheelSize,
                                        FORWARD = forward,
                                        PULSES = pulsesPerRotation,
                                        SPEED = 0)
    except KeyError as key:
        log.fatal("Unable to find a key in the INI file: {}".format(key))
        sys.exit(1)


    if not odometer.connect():
        print("Unable to connect to odometer.")
        log.error(constants.MSG_ODOMETER_CONNECTION)
        sys.exit(1)

    # Run diagnostics on the odometer before we begin.
    diagnosticResult, diagnosticText = odometer.diagnostics()

    if not diagnosticResult:
        print(diagnosticText)
        log.error(diagnosticText)
        sys.exit(1)

    return odometer




def reportProgress():
    # TODO: Report forward progress to message bus
    return

def nanoseconds() -> int:
    return time.time_ns()

def serviceQueue(odometer : PhysicalOdometer, odometryRoom: MUCCommunicator, announcements: int):
    """
    Service the queue of readings from line. This routine will not return.
    This will send a message to the odometry room indicating forward movement
    Backwards movement is handled by this logic, but no indication of that is sent to the room
    :param odometer: The odometer object with the queue
    :param odometryRoom: The MUC room
    :param announcements: Send announcements when this distance is covered (mm)
    """
    changeQueue = odometer.changeQueue

    # The previous angle -- the current reading will definitely be different
    previous = 0.0

    totalDistanceTraveled = 0.0
    distanceTraveledSinceLastMessage = 0.0
    servicing = True

    while not odometryRoom.connected:
        log.debug("Waiting for odometry room connection")
        time.sleep(5)

    # if odometryRoom.diagnostics():
    #     odometryRoom.sendMessage("Connection OK")


    log.debug("Waiting for angle changes to appear on queue")
    # Loop until graceful exit.
    i = 0
    starttime = nanoseconds()
    while servicing:
        angle = changeQueue.get(block=True)
        distanceTraveled = (angle - previous) * odometer.distancePerDegree
        totalDistanceTraveled += distanceTraveled
        previous = angle
        stoptime = nanoseconds()
        elapsedSeconds = (stoptime - starttime) / 1000000000
        starttime = nanoseconds()

        # Speed in kph
        speed = 0
        try:
            speed = (distanceTraveled / 100000) / (elapsedSeconds / 3600)
        except ZeroDivisionError as zero:
            log.warning("Elapsed time was 0.  Something is wrong")

        packet = gps.getCurrentPosition()
        if packet is not None:
            position = packet.position()
        else:
            position = (0.0,0.0)

        log.debug("{:.4f} mm Total: {:.4f} elapsed {:.4f} Speed {:.4f} kph location: {}".format(distanceTraveled, totalDistanceTraveled, elapsedSeconds, speed, position))

        # Send out a message every time the system traverses the distance specified

        distanceTraveledSinceLastMessage += distanceTraveled
        if distanceTraveledSinceLastMessage >= announcements:
            message = OdometryMessage()

            # Include GPS data if available
            if gps.connected:
                (message.latitude, message.longitude) = gps.getCurrentPosition().position()

            message.distance = announcements
            message.speed = speed
            message.totalDistance = totalDistanceTraveled
            # Timestamp is the nanoseconds in the epoch
            message.timestamp = time.time() * 1000
            #message.timestamp = time.time_ns()
            message.source = odometer.source
            messageText = message.formMessage()
            #log.debug("Sending: {}".format(message.formMessage()))

            odometryRoom.sendMessage(messageText)
            # try:
            #     odometryRoom.sendMessage(messageText)
            # except Exception as e:
            #     log.fatal("---- Error in sending message ----")
            #     log.fatal("Raw {}".format(e))
            distanceTraveledSinceLastMessage = 0.0
        if distanceTraveledSinceLastMessage <= -announcements:
            message = OdometryMessage()
            message.distance = -announcements
            message.timestamp = time.time() * 1000
            #message.timestamp = time.time_ns()
            message.source = odometer.source
            messageText = message.formMessage()
            #log.debug("Sending: {}".format(messageText))
            try:
                odometryRoom.sendMessage(messageText)
            except Exception as e:
                log.fatal("---- Error in sending message -----")
            distanceTraveledSinceLastMessage = 0.0

        i += 1


#log.setLevel(logging.INFO)
def processMessages(odometry: MUCCommunicator):
    # Connect to the XMPP server and just return
    odometry.connect(False)

def processMessagesSync(odometry: MUCCommunicator):
    # Connect to the XMPP server and just return
    odometry.connect(True)

def processOdometer(odometer: Odometer):
    log.debug("Connect and start odometer")
    # Connect the odometer and start.
    odometer.connect()
    # The start routines never return - this is executed in the main thread
    while True:
        if odometer.start():
            # This allows the type of odometry to be changed
            odometer = startupOdometer(typeOfOdometry)
        else:
            break

#
# Connect to the depth camera and grab readings
#

def takeImages(camera: CameraDepth):

    rc = 0

    # Loop until we get an error
    while rc == 0:
        while camera.state.is_idle:
            log.debug("Camera is idle")
            time.sleep(.5)

        # Connect to the camera and take an image
        log.debug("Connecting to camera")
        camera.connect()
        camera.initialize()
        camera.start()

        if camera.initializeCapture():
            try:
                camera.startCapturing()
            except IOError as io:
                log.critical("Failed to begin capturing")
                camera.log.error(io)
                rc = -1
            rc = 0
        else:
            rc = -1

# Start the NI system and run diagnostics
log.debug("Starting system")
systemNI, emitterRight, emitterLeft = startupSystem()

# If the emitters fail diagostics
if emitterLeft is None or emitterLeft is None:
    log.fatal("--- Failure in emitter diagnostics. This is a non-recoverable error. ---")
    sys.exit(-1)

# system is the object representing the RIO
# emitter is the associated emitter.

odometer = startupOdometer(typeOfOdometry)

camera = startupDepthCamera()

# Connect to the GPS
gps = startupGPS()

# Startup communication to the MUC
(odometryRoom, systemRoom, treatmentRoom) = startupCommunications(options)
threads = list()

# log.debug("Start generator thread")
# generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY,target=processMessagesSync, args=(odometryRoom,))
# threads.append(generator)
# generator.start()
odometryRoom.connect(False)

log.debug("Start system thread")
sysThread = threading.Thread(name=constants.THREAD_NAME_SYSTEM,target=processMessagesSync,args=(systemRoom,))
sysThread.daemon = True
threads.append(sysThread)
sysThread.start()

# Start a thread to service the readings queue
try:
    announcementInterval = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_ANNOUNCEMENTS))
except KeyError as key:
    log.error("INI file must contain [{}] {}".format(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_ANNOUNCEMENTS))
    sys.exit(1)

log.debug("Start service thread")
service = threading.Thread(name=constants.THREAD_NAME_SERVICE,target=serviceQueue, args=(odometer,odometryRoom,announcementInterval))
service.daemon = True
threads.append(service)
service.start()

log.debug("Starting treatment thread")
treat = threading.Thread(name=constants.THREAD_NAME_TREATMENT, target=processMessagesSync, args=(treatmentRoom,))
treat.daemon = True
threads.append(treat)
treat.start()

log.debug("Start odometry thread")
odometry = threading.Thread(name=constants.THREAD_NAME_ODOMETRY, target=processOdometer, args=(odometer,))
odometry.daemon = True
threads.append(odometry)
odometry.start()

log.debug("Start IMU thread")
imu = threading.Thread(name=constants.THREAD_NAME_IMU, target=takeImages, args=(camera,))
imu.daemon = True
threads.append(imu)
imu.start()


while True:
    time.sleep(5)
    for thread in threads:
        if not thread.is_alive():
            log.error("Thread {} exited. This is not normal.".format(thread.name))
            sys.exit()


sys.exit(0)