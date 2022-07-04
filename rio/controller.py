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
import threading
import time
import sys
from typing import Callable
from OptionsFile import OptionsFile
import logging
import logging.config
import nidaqmx as ni

import constants
from Odometer import Odometer, VirtualOdometer
from PhysicalOdometer import PhysicalOdometer
from Emitter import Emitter, PhysicalEmitter, VirtualEmitter
from MUCCommunicator import MUCCommunicator
from Messages import MUCMessage, OdometryMessage



# TODO: Add this as a parameter and sort out multiple emitters

RIO_MODULE = "Mod4"

parser = argparse.ArgumentParser("RIO Controller")

parser.add_argument('-e', '--emitter', action="store_true", required=False, default=False,help="Virtual emitter mode")
parser.add_argument('-o', '--odometer', action="store_true", required=False, default=False, help="Virtual Odometry Mode")
parser.add_argument('-a', '--odometer_line_a', action="store", required=False, default="", help="Odometer line A")
parser.add_argument('-b', '--odometer_line_b', action="store", required=False, default="", help="Odometer line B")
parser.add_argument('-p', '--plan', action="store_true", required=False, default=False, help="Generate treatment plan")
parser.add_argument('-r', '--rio', action="store_true", required=False, default=False, help="Virtual RIO mode")
parser.add_argument('-s', '--speed', action="store", required=False, default=44.704, type=float, help="Virtual odometry speed (cm/s)")
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
    # TODO: Allow multiple emitters

    # V I R T U A L  E M I T T E R  O R  P H Y S I C A L  E M I T T E R

    # Here, we associated an module on the RIO with the emitter. I suppose this needs to be down
    # to a set of pins on that card as well, but this will do for now

    if arguments.emitter:
        emitter = VirtualEmitter(RIO_MODULE)
    else:
        emitter = PhysicalEmitter(RIO_MODULE)


    diagnosticResultEmitter, diagnosticTextEmitter = emitter.diagnostics()
    if not diagnosticResultEmitter:
        print(diagnosticTextEmitter)

    return system, emitter

#
# M U C
#
def process(msg):
    global messageNumber
    print("Process {}".format(messageNumber))
    messageNumber += 1

def startupCommunications(options: OptionsFile):

    # The room that will get the announcements about forward or backward progress
    odometryRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_RIO),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                   options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_ODOMETRY),
                                   process)

    # The room that will get status reports about this process
    systemRoom = MUCCommunicator(options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_JID_RIO),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_NICK_ODOMETRY),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_DEFAULT_PASSWORD),
                                 options.option(constants.PROPERTY_SECTION_XMPP, constants.PROPERTY_ROOM_SYSTEM),
                                 process)

    return (odometryRoom, systemRoom)

#
# O D O M E T R Y
#

# TODO: refactor the parameters to create the encoder so this is not needed
def callback():
    return

def startupOdometer(treatmentController: Callable) -> Odometer:
    """
    Start up the odometry subsystem and run diagnostics
    :param imageProcessor: The image processor executed at each interval
    :return:
    """
    pulsesPerRotation = 0
    wheelSize = 0

    # V I R T U A L  O D O M E T E R
    #
    # This creates an odometer that will simulate moving forward with a given speed and will
    # call the treatment controller at set intervals
    if arguments.odometer:
        log.debug("Using virtual odometry")
        odometer = VirtualOdometer(arguments.speed, treatmentController)
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

        odometer = PhysicalOdometer(lineA, lineB, wheelSize, pulsesPerRotation, 0, callback)

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

    if odometryRoom.diagnostics():
        odometryRoom.sendMessage("Connection OK")


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
        elapsed = stoptime - starttime
        starttime = nanoseconds()

        log.debug("{:.4f} mm Total: {:.4f} Elapsed time {} ns".format(distanceTraveled, totalDistanceTraveled, elapsed))

        # Send out a message every time the system traverses the distance specified

        distanceTraveledSinceLastMessage += distanceTraveled
        if distanceTraveledSinceLastMessage >= announcements:
            message = OdometryMessage()
            message.distance = announcements
            # Timestamp is the nanoseconds in the epoch
            message.timestamp = time.time_ns()
            messageText = message.formMessage()
            log.debug("Sending: {}".format(message.formMessage()))
            odometryRoom.sendMessage(messageText)
            distanceTraveledSinceLastMessage = 0.0
        if distanceTraveledSinceLastMessage <= -announcements:
            message = OdometryMessage()
            message.distance = -announcements
            message.timestamp = time.time_ns()
            messageText = message.formMessage()
            log.debug("Sending: {}".format(messageText))
            odometryRoom.sendMessage(messageText)
            distanceTraveledSinceLastMessage = 0.0

        i += 1


#log.setLevel(logging.INFO)
def processMessages(odometry: MUCCommunicator):
    # Connect to the XMPP server and just return
    odometry.connect(False)

# Start the system and run diagnostics
log.debug("Starting system")
system, emitter = startupSystem()

# system is the object representing the RIO
# emitter is the associated emitter.

# Start the odometer, and have it call the routine to apply the treatment
# Here, we assume that the method to apply the treatment will be called at the precision of the system
# Let's say it is every 1 cm

odometer = startupOdometer(emitter.applyTreatment)

# Startup communication to the MUC
(odometryRoom, systemRoom) = startupCommunications(options)

generator = threading.Thread(target=processMessages(odometryRoom))
generator.start()

# Start a thread to service the readings queue
try:
    announcementInterval = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_ANNOUNCEMENTS))
except KeyError as key:
    log.error("INI file must contain [{}] {}".format(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_ANNOUNCEMENTS))
    sys.exit(1)

service = threading.Thread(target=serviceQueue, args=(odometer,odometryRoom,announcementInterval))
service.start()



log.debug("Connect and start odometer")
# Connect the odometer and start.
odometer.connect()
# The start routines never return - this is executed in the main thread
odometer.start()

service.join()
generator.join()

sys.exit(0)