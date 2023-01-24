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

import gpsd

from OptionsFile import OptionsFile
# import logging
import logging.config
import xmpp
import numpy as np

try:
    import nidaqmx as ni
    virtualRequired = False
except ImportError:
    virtualRequired = True

from statemachine.exceptions import TransitionNotAllowed

import constants
from Odometer import Odometer, VirtualOdometer
from PhysicalOdometer import PhysicalOdometer
from Emitter import PhysicalEmitter, VirtualEmitter
from MUCCommunicator import MUCCommunicator
from Messages import OdometryMessage, SystemMessage, TreatmentMessage
from GPSClient import GPS
from CameraDepth import CameraDepth
from NationalInstruments import VirtualNationalInstruments, PhysicalNationalInstruments
from WeedExceptions import XMPPServerAuthFailure, XMPPServerUnreachable, DAQError
from RealSense import RealSense
from Diagnostics import DiagnosticsDAQ


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
        if cameraForIMU is not None:
            cameraForIMU.state.toClaim()

    except TransitionNotAllowed as transition:
        log.critical("Unable to transition the camera to claim")
        log.critical(transition)

    log.debug("Session started")
    return started

def endSession(options: OptionsFile, name: str) -> bool:
    global log
    global imageNumber
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

    if cameraForIMU is not None:
        cameraForIMU.stop()
    imageNumber = 0

    # Change back to the general output directory not associated with any session
    try:
        os.chdir(path)
        stopped = True
    except OSError as e:
        log.critical("Unable to prepare and set directory to {}".format(path))
        log.critical("Raw: {}".format(e))

    log.debug("Session ended")
    return stopped

#
# N I  D I A G N O S T I C S
#
def diagnostics() -> DiagnosticsDAQ:
    # TODO: NI Diagnostics
    diagnosticResults = DiagnosticsDAQ()

    return diagnosticResults

#
# Start up system components and run diagnostics
#
def startupSystem(options: OptionsFile):

    # The system is running in a mode where a physical RIO is expected

    if not arguments.rio:
        attachedDAQ = PhysicalNationalInstruments()
        system = ni.system.System.local()
        attachedDAQ.system = system

        #system.driver_version

        # This delays system startup until the DAQ is connected to USB and power

        while len(system.devices) == 0:
            log.error("Unable to locate DAQ devices. Is the DAQ plugged in and powered?")
            # This delay is completely arbitrary. Could be eliminated with no ill effects.
            time.sleep(5)
            system = ni.system.System.local()

        devices = system.devices.device_names
        for device in devices:
            log.debug("Found: {}".format(device))

        # Make certain the cards are in the position described
        emitterCardMissing = True
        odometerCardMissing = True

        # The default names for the cards
        emitterCardName = "Mod4"
        odometerCardName = "Mod3"

        try:
            emitterCardName = options.option(constants.PROPERTY_SECTION_RIO, constants.PROPERTY_RIGHT)
            emitterCardMissing = emitterCardName not in system.devices.device_names
        except KeyError as key:
            log.error("Unable to file {}/{} in options file.".format(constants.PROPERTY_SECTION_RIO, constants.PROPERTY_RIGHT))

        try:
            odometerCardName = options.option(constants.PROPERTY_SECTION_RIO, constants.PROPERTY_CARD_ODOMETER)
            odometerCardMissing = odometerCardName not in system.devices.device_names
        except KeyError as key:
            log.error("Unable to file {}/{} in options file.".format(constants.PROPERTY_SECTION_RIO, constants.PROPERTY_CARD_ODOMETER))

        if odometerCardMissing:
            errText = "Unable to find card for the odometer ({}) in {}".format(odometerCardName, system.devices.device_names)
            raise DAQError(errText, True)

        if emitterCardMissing:
            errText = "Unable to find card for the emitter ({}) in {}".format(emitterCardName, system.devices.device_names)
            raise DAQError(errText, True)
    else:
        attachedDAQ = VirtualNationalInstruments()

    # channels = ni.system.physical_channel.PhysicalChannel("Mod3")
    # print(channels)

    # Run diagnostics on the NI system
    diagnosticResult = diagnostics()

    # Connect to the emitter and run diagnostics

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

    try:
        emitterDiagnosticType = options.option(constants.PROPERTY_SECTION_EMITTER, constants.PROPERTY_EMITTER_DIAG)
        if emitterDiagnosticType == constants.EMITTER_DIAGNOSTIC_WET:
            enableEmitterInDiagnostics = True
        else:
            enableEmitterInDiagnostics = False

    except KeyError as key:
        log.error("Unable to find emitter diagnostic type. Assuming default of dry")
        enableEmitterInDiagnostics = False


    diagnosticResultRightEmitter, diagnosticTextRightEmitter = rightEmitter.diagnostics(enableEmitterInDiagnostics)
    diagnosticResultLeftEmitter, diagnosticTextLeftEmitter = leftEmitter.diagnostics(enableEmitterInDiagnostics)

    if not diagnosticResultRightEmitter:
        log.warning(diagnosticTextRightEmitter)
        rightEmitter = None

    if not diagnosticResultLeftEmitter:
        log.warning(diagnosticTextLeftEmitter)
        leftEmitter = None

    return attachedDAQ, rightEmitter, leftEmitter

#
# P R O C E S S
#
# For the odometer, there isn't much to do, as it just reports movement.
# For the treatment, there are a few things: listen for identifications, and for begin/end cycles
#
def process(conn, msg: xmpp.protocol.Message):
    global currentSessionName
    global currentOperation
    log.debug("Process message from {}: {}".format(msg.getFrom(), msg.getBody()))

    # Not sure what is happening here, but sometimes we get messages with an empty body.
    if msg.getBody() is None:
        log.debug("Empty message received")
        return

    # Messages from the system room will be in the form: system@conference.weeds.com/console

    console_in_system = options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM) \
              +  "/" \
              + options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONSOLE)

    if msg.getFrom() == console_in_system:
        if msg.getBody() is not None:
            systemMsg = SystemMessage(raw=msg.getBody())
            timeMessageSent = systemMsg.timestamp
            timeDelta = (time.time() * 1000) - timeMessageSent
            log.debug("Action: {} Time Delta: {}".format(systemMsg.action, timeDelta))

            # Determine if this is an old message or not
            if timeDelta < 5000:
                # S T A R T  O R  S T O P  S E S S I O N
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

def processTreatment(conn, msg: xmpp.protocol.Message):
    """
    Process messages from the treatment room
    :param conn:
    :param msg:
    """
    log.debug("Process treatment message from [{}]: {}".format(msg.getFrom(), msg.getBody()))

    # Not sure what is happening here, but sometimes we get messages with an empty body.
    if msg.getBody() is None:
        log.debug("Empty message received")
        return

    # Messages from the system room will be in the form: system@conference.weeds.com/console

    console_in_treatment = options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_TREATMENT) \
              +  "/" \
              + options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_CONSOLE)

    # This is the case where a user requests the emitter change from the application
    if msg.getFrom() == console_in_treatment:
        treatmentMsg = TreatmentMessage(raw=msg.getBody())
        timeMessageSent = treatmentMsg.timestamp
        timeDelta = (time.time() * 1000) - timeMessageSent
        if timeDelta < 5000:
            log.debug("Treatment for emitter: (side={},tier={},number={}) for {} seconds Time Delta: {}".format(
                treatmentMsg.side,
                treatmentMsg.tier,
                treatmentMsg.number,
                treatmentMsg.duration,
                timeDelta))

            # A value in the tier of EMITTER_ALL means all the emitters in all the tiers.
            if treatmentMsg.tier == int(constants.EMITTER_ALL):
                log.debug("Purging emitters on side: {}".format(constants.Side(treatmentMsg.side)))
                emitterRight.beginPreparations()
                emitterRight.addAllEmitters(constants.Side(treatmentMsg.side))
                emitterRight.turnOnEmitters(treatmentMsg.duration)
                # emitterRight.cleanup()

            else:
                # Turn on the emitter and leave it on for the duration specified
                emitterRight.on(constants.Side(treatmentMsg.side), treatmentMsg.tier, treatmentMsg.number, treatmentMsg.duration)
        else:
            log.info("Old message seen: time delta {}".format(timeDelta))
#
# Processing the odometry messages is required here only because the depth cameras are on the same system that interprets the odometry
# signals -- this should be moved to the nvidia systems later

imageNumber = 0
distanceTravelledSinceLastCapture = 0

def nullProcessor(conn, msg: xmpp.protocol.Message):
    log.debug("Null message processor")
    pass

def processOdometry(conn, msg: xmpp.protocol.Message):
    global typeOfOdometry
    global distanceTravelledSinceLastCapture
    global imageNumber

    depthArray = np.ndarray
    log.debug("Process odometry message from {}".format(msg.getFrom()))
    odometryMsg = OdometryMessage(raw=msg.getBody())
    distanceTravelledSinceLastCapture += odometryMsg.distance
    log.debug("Distance travelled: {}".format(odometryMsg.distance))

    try:
        imageWidth = int(options.option(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_IMAGE_WIDTH))
    except KeyError:
        # This is normal when things are on the odometer system, as we don't have the attribute we need there
        log.error("Unable to find {}/{} in ini file".format(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_IMAGE_WIDTH))
        imageWidth = 310

    # If the system has travelled the width of the RGB image, take a depth reading.
    # if distanceTravelledSinceLastCapture > imageWidth:
    #     imageNumber = cameraForDepthLeft.imageNumber + 1
    #     cameraForDepthLeft.imageNumber = imageNumber
    #     cameraForDepthRight.imageNumber = imageNumber
    #
    #     # Left and right depth images
    #     if cameraForDepthLeft.state.is_capturing:
    #         depthArray = cameraForDepthLeft.capture()
    #         imageName = "depth-left-{:05d}".format(imageNumber)
    #         log.debug("Saving depth image {}".format(imageName))
    #         np.save(imageName,depthArray)
    #
    #     if cameraForDepthRight.state.is_capturing:
    #         depthArray = cameraForDepthRight.capture()
    #         imageName = "depth-right-{:05d}".format(imageNumber)
    #         log.debug("Saving depth image {}".format(imageName))
    #         np.save(imageName,depthArray)

        distanceTravelledSinceLastCapture = 0


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

    odometryRoom2 = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_SERVER),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_RIO),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_IMU),
                                    options.option(constants.PROPERTY_SECTION_XMPP,
                                                   constants.PROPERTY_DEFAULT_PASSWORD),
                                    options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
                                    nullProcessor,
                                    None)  # Don't care about presence

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
                                     processTreatment,
                                     None) # Don't care about presence

    return (odometryRoom, odometryRoom2, systemRoom, treatmentRoom)

#
# G P S
#
def startupGPS() -> GPS:
    theGPS = GPS()
    if not theGPS.connect():
        log.error("Unable to connect to the GPS")

    if theGPS.isAvailable():
        packet = theGPS.getCurrentPosition()

        # The GPS in my office is a bit intermittent -- sometimes it reports that it does not
        # have at least a 2D fix, so we see errors.  But, a call moments later will succeed.

        if packet is not None:
            try:
                log.debug("GPS Position: {}".format(packet.position()))
                log.debug("GPS Error: {}".format(packet.position_precision()))
                log.debug("GPS Fix: {}".format(packet.mode))
            except gpsd.NoFixError as fix:
                log.error("Unable to start GPS client: {}".format(fix))
        else:
            log.error("The GPS does not yet have a 2D fix")

    else:
        log.warning("GPS location is not available. Image exif will not include this information")
    return theGPS

#
# D E P T H  C A M E R A
#
def startupDepthCamera() -> CameraDepth:

    sensors = RealSense()
    markSensorAsFailed = False
    intelIMU = None

    sensors.query()
    if sensors.count() < 1:
        log.error("Detected {} depth/IMU sensors. Expected 1 or more.".format(sensors.count()))
        log.error("No sensor will be used.")
        intelIMU = CameraDepth(constants.Capture.IMU,
                               gyro=constants.PARAM_FILE_GYRO,
                               acceleration=constants.PARAM_FILE_ACCELERATION)
        intelIMU.state.toMissing()
        return intelIMU

    # Start the IMU camera
    try:
        intelIMU = CameraDepth(constants.Capture.IMU,
                               gyro=constants.PARAM_FILE_GYRO,
                               acceleration=constants.PARAM_FILE_ACCELERATION)
        if markSensorAsFailed:
            intelIMU.state.toMissing()
        else:
            intelIMU.state.toIdle()
    except KeyError:
        log.error("Unable to find serial number for depth camera: {}/{} & {}".format(constants.PROPERTY_SECTION_CAMERA, constants.PROPERTY_SERIAL_LEFT, constants.PROPERTY_SERIAL_RIGHT))
        intelIMU = None
    except ValueError as val:
        log.error("Camera construction failed: {}".format(val))
        intelIMU = None

    return intelIMU

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
    wheelSize = 0.0

    try:
        # V I R T U A L  O D O M E T E R
        #
        # This creates an odometer that will simulate moving forward with a given speed and will
        # call the treatment controller at set intervals
        if typeOfOdometer == constants.SubsystemType.VIRTUAL:
            log.debug("Using virtual odometry")
            pulsesPerRotation = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PPR))
            wheelSize = float(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_WHEEL_CIRCUMFERENCE))
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
                    log.debug("Using PPR: {}".format(pulsesPerRotation))
                except KeyError:
                    log.error("Pulses Per Rotation must be specified as command line option or in the INI file.")
            if wheelSize == 0:
                try:
                    wheelSize = float(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_WHEEL_CIRCUMFERENCE))
                    log.debug("Using wheel size: {}".format(wheelSize))
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

def nanoseconds() -> float:
    ns = time.time() * 1e9
    # ns = time.time_ns()
    return ns

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

    # The pulses of the encoder
    pulses = 0
    mmTotalTravel = 0.0
    distanceTraveledSinceLastMessage = 0.0
    servicing = True

    while not odometryRoom.connected:
        log.debug("Waiting for odometry room connection")
        time.sleep(5)
    log.debug("Connected to odometry room")

    # if odometryRoom.diagnostics():
    #     odometryRoom.sendMessage("Connection OK")


    log.debug("Waiting for angle changes to appear on queue")
    # Loop until graceful exit.
    i = 0
    #starttime = time.time_ns()
    starttime = nanoseconds()
    while servicing:
        angle = changeQueue.get(block=True)
        # mm of travel
        mmTraveled = (angle - previous) * odometer.distancePerDegree
        mmTotalTravel += mmTraveled
        previous = angle
        pulses += 1
        # Record the time of this reading -- not quite correct, as this should be in the observation
        #stoptime = time.time_ns()
        stoptime = nanoseconds()
        #elapsedSeconds = (stoptime - starttime) / 1000000000
        elapsedSeconds = (stoptime - starttime) / 1e9
        starttime = nanoseconds()

        kph = 0
        try:
            log.debug("Travelled: {} mm  in {} seconds".format(mmTraveled, elapsedSeconds))
            kph = (mmTraveled / 1e6) / (elapsedSeconds / 3600)
        except ZeroDivisionError as zero:
            log.warning("Elapsed time was 0.  Something is wrong")

        packet = gps.getCurrentPosition()
        if packet is not None:
            try:
                position = packet.position()
            except gpsd.NoFixError as fix:
                log.error("Unable to obtain a 2D fix to determine location")
                position = (0.0, 0.0)
        else:
            position = (0.0, 0.0)

        if cameraForIMU.connected:
            gyro = np.array2string(cameraForIMU.gyro, formatter={'float_kind': lambda x: "%.2f" % x})
            acceleration = np.array2string(cameraForIMU.acceleration, formatter={'float_kind': lambda x: "%.2f" % x})
            log.debug("{:.4f} mm Total: {:.4f} mm elapsed {:.4f} Speed {:.4f} kph location: {} gyro: {} acceleration {}".
                      format(mmTraveled, mmTotalTravel, elapsedSeconds, kph, position, gyro, acceleration))
        else:
            log.debug("{:.4f} mm Total: {:.4f} mm elapsed {:.4f} Speed {:.4f} kph location: {} ".
                      format(mmTraveled, mmTotalTravel, elapsedSeconds, kph, position))

        # Send out a message every time the system traverses the distance specified -- forward or backward

        distanceTraveledSinceLastMessage += mmTraveled
        if distanceTraveledSinceLastMessage >= announcements or distanceTraveledSinceLastMessage <= -announcements:

            # Create a blank odometry message
            message = OdometryMessage()

            # Include GPS data if available
            if gps.connected:
                 (message.latitude, message.longitude) = position

            # Include gyro information if available

            if cameraForIMU.connected:
                message.gyro = cameraForIMU.gyro
                message.acceleration = cameraForIMU.acceleration

            message.distance = distanceTraveledSinceLastMessage
            message.speed = kph
            message.totalDistance = mmTotalTravel
            message.pulses = pulses
            # Timestamp is the nanoseconds in the epoch
            #message.timestamp = time.time() * 1000
            message.timestamp = nanoseconds()
            message.source = odometer.source
            message.type = constants.OdometryMessageType.DISTANCE
            messageText = message.formMessage()
            #log.debug("Sending: {}".format(message.formMessage()))

            odometryRoom.sendMessage(messageText)
            # try:
            #     odometryRoom.sendMessage(messageText)
            # except Exception as e:
            #     log.fatal("---- Error in sending message ----")
            #     log.fatal("Raw {}".format(e))
            distanceTraveledSinceLastMessage = 0.0

        i += 1


#log.setLevel(logging.INFO)
def processMessages(odometry: MUCCommunicator):
    # Connect to the XMPP server and just return
    odometry.connect(False)

def processMessagesSync(communicator: MUCCommunicator):
    # Connect to the XMPP server and just return
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
        log.debug("Disconnected from chatroom")
        communicator.state.toDisconnected()

    #odometry.connect(True)

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
# Connect to the depth camera and grab readings for IMU
#

def takeIMUReadings(camera: CameraDepth):

    rc = 0

    # Loop until we get an error
    while rc == 0:

        # This is the case where we could not detect the camera at startup

        if camera.state.is_missing:
            log.warning("Unable to detect IMU. Sleeping for now")
            time.sleep(5)
        else:
            while camera.state.is_idle:
                #log.debug("Camera is idle (This is normal when an operation is not in progress")
                time.sleep(.5)

            # Connect to the camera and take an image
            log.debug("Connecting to IMU for capture type: {}".format(camera.captureType))
            camera.connect()
            camera.initialize()
            camera.start()

            if camera.initializeCapture():
                log.debug("Starting capture type: {}".format(camera.captureType))
                try:
                    camera.startCapturing()
                except IOError as io:
                    log.critical("Failed to begin capturing")
                    camera.log.error(io)
                    rc = -1
                rc = 0
            else:
                log.critical("Unable to initialize camera")
                rc = -1

# Start the NI system and run diagnostics
log.debug("Starting system")
try:
    systemNI, emitterRight, emitterLeft = startupSystem(options)
except DAQError as daq:
    log.fatal(daq._message)
    sys.exit(-1)

# If the emitters fail diagostics
if emitterLeft is None or emitterLeft is None:
    log.fatal("--- Failure in emitter diagnostics. This is a non-recoverable error. ---")
    sys.exit(-1)

# system is the object representing the RIO
# emitter is the associated emitter.

odometer = startupOdometer(typeOfOdometry)

cameraForIMU = startupDepthCamera()

# Connect to the GPS
gps = startupGPS()

# Startup communication to the MUC
(odometryRoom, odometryRoom2, systemRoom, treatmentRoom) = startupCommunications(options)
threads = list()

# log.debug("Start generator thread")
# generator = threading.Thread(name=constants.THREAD_NAME_ODOMETRY,target=processMessagesSync:, args=(odometryRoom,))
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
imu = threading.Thread(name=constants.THREAD_NAME_IMU, target=takeIMUReadings, args=(cameraForIMU,))
imu.daemon = True
threads.append(imu)
imu.start()



# # The position thread will not be required when the depth cameras are moved to the nvidia systems
# log.debug("Start position thread")
# position = threading.Thread(name=constants.THREAD_NAME_POSITION, target=processMessagesSync, args=(odometryRoom2,))
# position.daemon = True
# threads.append(position)
# position.start()

while True:
    time.sleep(5)
    for thread in threads:
        if not thread.is_alive():
            log.error("Thread {} exited. This is not normal.".format(thread.name))
            sys.exit()


sys.exit(0)