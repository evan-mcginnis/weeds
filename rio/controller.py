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
import time
import sys
from typing import Callable

from Odometer import Odometer, VirtualOdometer
from Emitter import Emitter, PhysicalEmitter, VirtualEmitter
import nidaqmx as ni

# TODO: Add this as a parameter and sort out multiple emitters

RIO_MODULE = "Mod4"

parser = argparse.ArgumentParser("RIO Controller")

parser.add_argument('-r', '--rio', action="store_true", required=False, default=False, help="Virtual RIO mode")
parser.add_argument('-o', '--odometer', action="store_true", required=False, default=False, help="Virtual odometry mode")
parser.add_argument('-s', '--speed', action="store", required=False, default=44.704, type=float, help="Virtual odometry speed (cm/s)")
parser.add_argument('-e', '--emitter', action="store_true", required=False, default=False,help="Virtual emitter mode")
parser.add_argument('-p', '--plan', action="store_true", required=False, default=False, help="Generate treatment plan")
arguments = parser.parse_args()

# Use -r -o -e when running somewhere other than a RIO

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
    system.driver_version
    devices = system.devices
    for device in system.devices:
        print(device)
    channels = ni.system.physical_channel.PhysicalChannel("Mod5")
    print(channels)

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
# O D O M E T R Y
#

def startupOdometer(treatmentController: Callable) -> Odometer:
    """
    Start up the odometry subsystem and run diagnostics
    :param imageProcessor: The image processor executed at each interval
    :return:
    """
    # V I R T U A L  O D O M E T E R
    #
    # This creates an odometer that will simulate moving forward with a given speed and will
    # call the treatment controller at set intervals
    odometer = VirtualOdometer(arguments.speed, treatmentController)
    if not odometer.connect():
        print("Unable to connect to odometer.")
        sys.exit(1)

    # TODO: Connect to physical odometry

    # Run diagnostics on the odometer before we begin.
    diagnosticResult, diagnosticText = odometer.diagnostics()

    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)

    return odometer




def reportProgress():
    # TODO: Report forward progress to message bus
    return

# Start the system and run diagnostics
system, emitter = startupSystem()

# system is the object representing the RIO
# emitter is the associated emitter.

# Start the odometer, and have it call the routine to apply the treatment
# Here, we assume that the method to apply the treatment will be called at the precision of the system
# Let's say it is every 1 cm

odometer = startupOdometer(emitter.applyTreatment)

# This is an endless loop
odometer.start()

sys.exit(0)