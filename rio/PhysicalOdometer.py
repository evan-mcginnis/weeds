#
# P H Y S I C A L  O D O M E T E R
#
# A physical odometer assumes a quadature encoder wheel and reads values from that wheel using
# a National Instruments RIO and a 9411 card
#
# The characteristics of the encoder are in the INI file, along with the wheel dimensions
#

import threading

import nidaqmx.constants

# Needs YAML on RIO platform
#import yaml

import constants
from Odometer import Odometer
from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime
import time
#from collections import deque
import queue

import NIUtilities
import nidaqmx as ni

from nidaqmx.constants import AcquisitionType, TaskMode, CountDirection, Edge
from nidaqmx._task_modules.channels.ci_channel import CIChannel
from nidaqmx._task_modules.channel_collection import ChannelCollection
from nidaqmx.stream_readers import CounterReader
from nidaqmx.constants import AngleUnits
from nidaqmx.constants import EncoderType
from nidaqmx.constants import LineGrouping
from nidaqmx._task_modules.timing import Timing
import nidaqmx.errors


# Debounce time in seconds. Lower values don't seem to compensate for the noisy line
# This seems like a balancing act, as the encoder will generate 1000 clicks per rotation and if we assume
# 4kph top speed, that 1111.11 mm per second, or 1.2 rotations of the wheel per second.
ODOMETER_DEBOUNCE = 0.0003

# The number of samples to collect
MAX_ODOMETER_SAMPLES = 1

# Size of the wheel in mm
WHEEL_SIZE = 923


# Clicks of the encoder per revolution
# See https://cdn.automationdirect.com/static/specs/encoderhd.pdf
ENCODER_CLICKS_PER_REVOLUTION = 1000


class PhysicalOdometer(Odometer):
    #
    # There is a bit of deviation from the virtual odometer here. The speed indicated in the constructor
    # is the MAXIMUM speed we can tolerate, where the virtual odometer accepts the speed at which the
    # odometer moves.  OK, that could be considered "always moves at the maximum", I suppose.
    #
    def __init__(self, **kwargs):
        """
        A physical odometer. This is for an Automation Direct TRD-NXXXX-RZWD
        :param kwargs: LINE_A, LINE_B, PULSES, SPEED, WHEEL_SIZE
        """
        # The card on the RIO where the encoder connects
        super().__init__(**kwargs)

        wheel_size = kwargs[constants.KEYWORD_WHEEL_CIRCUMFERENCE]
        encoder_clicks = kwargs[constants.KEYWORD_PPR]
        lineA = kwargs[constants.KEYWORD_LINE_A]
        lineB = kwargs[constants.KEYWORD_LINE_B]
        speed = kwargs[constants.KEYWORD_SPEED]

        self._counter = kwargs[constants.KEYWORD_COUNTER]

        # Indicate of forward means clockwise or counter-clockwise
        self._forward = kwargs[constants.KEYWORD_FORWARD]

        self._maxSpeed = speed

        # LineA and lineB should be in the format ModX/portY/lineZ
        # These are ot needed for the angular encoder technique, as the terminals are specified in other options (PFI0)
        breakdownA = NIUtilities.breakdown(lineA)
        breakdownB = NIUtilities.breakdown(lineB)
        self._module = breakdownA[0]

        self._start = 0
        self._elapsed_milliseconds = 0
        self._elapsed = 0
        # The circumference of the wheel in mm
        self._wheel_size = wheel_size
        # The number of clicks per 1 rotation
        self._encode_clicks = encoder_clicks
        # The distance travelled per click
        self._distance_per_click = wheel_size / encoder_clicks
        # The distance travelled per degree
        self._distancePerDegree = self._wheel_size / 360
        self._totalClicks = 0



        self.log = logging.getLogger(__name__)

        # The reading task
        self._task = None

        # Rotational items for the a,b,z lines
        self._aCurrent = False
        self._aPrevious = False
        self._bCurrent = False
        self._bPrevious = False
        self._zCurrent = False
        self._zPrevious = False

        self._source = constants.SOURCE_PHYSICAL

        # Not used
        self._aArmed = False



    @property
    def encoderClicksPerRevolution(self) -> int:
        """
        The number of clicks per revolution
        :return:
        """
        return self._encode_clicks

    @property
    def distancePerDegree(self) -> float:
        return self._distancePerDegree

    @property
    def distancePerClick(self) -> float:
        """
        The distance travelled with a single click
        :return:
        """
        return self._distance_per_click

    def connect(self) -> bool:
        """
        Connect to the wheel odometer
        :return: True on success, False otherwise
        """
        self._connected = True
        return True

    def disconnect(self):
        """
        Disconnect from wheel odometer
        :return: True on success, False otherwise
        """
        return True

    def diagnostics(self):
        """
        Run diagnostics on the odometer.
        :return: (bool, str) with the pass/fail of the diagnostics + short string with the results.
        """
        diagnosticResult = False
        diagnosticText = "Diagnostics not implemented"
        self.log.debug("Odometer diaganostics")
        # TODO: Complete diagnostics
        # Check to see if the 9411 has incoming power

        if self._connected:
            diagnosticResult = True
        return (diagnosticResult, diagnosticText)

    def registerCallback(self,callback):
        self._callback = callback


    def start(self) -> bool:
        """
        Treat the encoder as an angular encoder, putting new readings on the change queue
        If an error is encountered in starting the readings, return False.  Transient errors will not stop processing.
        """
        global running
        total = 0.0

        daqErrorEncountered = False
        self.log.debug("Begin rotation detection using Angular Encoding")

        try:
            task = ni.Task(new_task_name=constants.RIO_TASK_NAME)
        except nidaqmx.errors.DaqError:
            self.log.fatal("Unable to create NI task. Usually this is the result of an ungraceful shutdown")
            daqErrorEncountered = True

        #channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = 'Mod3/ctr0', decoding_type = EncoderType.X_1, zidx_enable=True, units=AngleUnits.DEGREES, pulses_per_rev=1000, initial_angle=0.0)

        if not daqErrorEncountered:
            # Add the channel as an angular encoder without Z index support, as we really don't care about the number of rotations
            try:
                # Original
                channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = self._counter, decoding_type = EncoderType.X_1, zidx_enable=False, zidx_val=0, units=AngleUnits.DEGREES, pulses_per_rev=self.encoderClicksPerRevolution, initial_angle=0.0)
                #channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = self._counter, decoding_type = EncoderType.X_4, zidx_enable=False, zidx_val=0, units=AngleUnits.DEGREES, pulses_per_rev=self.encoderClicksPerRevolution, initial_angle=0.0)
            except nidaqmx.errors.DaqError as daq:
                self.log.fatal("DAQ setup error encountered")
                self.log.fatal("Raw: {}".format(daq))
                daqErrorEncountered = True

            if self._forward == constants.Forward.CLOCKWISE.name:
                channelA.ci_encoder_a_input_term = 'PFI0'
                channelA.ci_encoder_b_input_term = 'PFI1'
            elif self._forward == constants.Forward.COUNTER_CLOCKWISE.name:
                channelA.ci_encoder_a_input_term = 'PFI1'
                channelA.ci_encoder_b_input_term = 'PFI0'
            else:
                self.log.fatal("[FORWARD = CLOCKWISE | COUNTER_CLOCKWISE] must be specified.  Got {}".format(self._forward))
                daqErrorEncountered = True

        if not daqErrorEncountered:
            # The pulse width specified works at low speed, but not at higher speeds
            # The minimum pulse width is .000006, but also depends on timing.
            # TODO: Sort through timings and the filter enable
            # channelA.ci_encoder_a_input_dig_fltr_min_pulse_width = 0.0001
            # channelA.ci_encoder_a_input_dig_fltr_enable = True
            # # Test of counter-clockwise/clockwise.  I believe we can just swap the definitions of A & B
            # #channelA.ci_encoder_a_input_term = 'PFI0'
            # channelA.ci_encoder_b_input_dig_fltr_min_pulse_width = 0.0001
            # channelA.ci_encoder_b_input_dig_fltr_enable = True
            # # Clockwise/counterclockwise test
            # #channelA.ci_encoder_b_input_term = 'PFI1'
            # channelA.ci_encoder_z_input_dig_fltr_min_pulse_width = 0.0001
            # channelA.ci_encoder_z_input_dig_fltr_enable = True
            # channelA.ci_encoder_z_input_term = 'PFI2'



            #task.timing.samp_clk_overrun_behavior = nidaqmx.constants.OverflowBehavior.TOP_TASK_AND_ERROR

            try:
                task.start()
            except nidaqmx.errors.DaqError as daqerr:
                self.log.fatal("Unable to start reading odometer.  Usually this means the resource is still reserved")
                self.log.fatal(daqerr)
                return daqErrorEncountered

            previous = 0.0
            self._processing = True
            running = True
            errorsEncountered = 0
            angleChangeSeen = False

            # This loop will run until things are gracefully shut down by another thread
            # setting running to False.
            while self._processing:
                try:
                    # This works at low speeds
                    ang = task.read(number_of_samples_per_channel=1)
                    # ang = task.read(nidaqmx.constants.READ_ALL_AVAILABLE)
                    # self.log.debug("Angle reading length: {} Angle: {}".format(len(ang), ang))
                except nidaqmx.errors.DaqError:
                    self.log.error("Read error encountered")
                    daqErrorEncountered = True
                    errorsEncountered += 1
                    # Tolerate at most 100 read errors
                    if errorsEncountered > 100:
                        self.log.fatal("Encountered more than 100 read errors from DAQ")
                        break
                    else:
                        continue

                # If the current reading has changed from the previous reading, the wheel has moved
                if ang[0] != 0 and ang[0] != previous:
                    try:
                        # Put the current angular reading in the queue
                        self._changeQueue.put(float(ang[0]), block=False)
                    except queue.Full:
                        self.log.error("Distance queue is full. Reading is lost.")
                    previous = ang[0]
                    angleChangeSeen = True
                if ang[0] == 0 and angleChangeSeen:
                    self.log.error("Read the angle as 0 after wheel has moved")

            self.log.info("Reading of odometer stopped")
            # Shutdown the task so the resources can be cleaned up
            task.stop()
            task.close()

        return not daqErrorEncountered



# #
# The Odometer class as a utility.
#
if __name__ == "__main__":
    import argparse
    import sys
    import os

    from logging.config import fileConfig
    # There is a bit of a snag here, as the YAML parsers are not installed, and I don't have wifi working yet
    # to get them.  I'll treat this as a future item.

    import constants
    from OptionsFile import OptionsFile


    running = False
    def userIO():
        global running
        input("Return to stop")
        running = False

    def nanoseconds() -> float:
        #return time.time_ns()
        return time.time() * 10000

    def serviceQueue(odometer : PhysicalOdometer):
        """
        Service the queue of readings from line. This routine will not return.
        :param odometer: The odometer object with the queue
        """
        changeQueue = odometer.changeQueue

        # The previous angle -- the current reading will definitely be different
        previous = 0.0

        totalDistanceTraveled = 0.0
        servicing = True

        starttime = nanoseconds()
        # Loop until graceful exit.
        i = 0
        while servicing:
            angle = changeQueue.get(block=True)
            distanceTraveled = (angle - previous) * odometer.distancePerDegree
            totalDistanceTraveled += distanceTraveled
            previous = angle

            # This is not really correct, as it computes elapsed time as it is fetched from the queue, not when
            # the observation was made.  Good enough for now.
            stoptime = nanoseconds()
            elapsed = stoptime - starttime
            starttime = nanoseconds()

            print("Travel: {:.4f} mm Total: {:.4f} Speed {}".format(distanceTraveled, totalDistanceTraveled, elapsed))
            i += 1
            # Determine if the wheel has undergone one rotation
            if i % odometer.encoderClicksPerRevolution == 0:
                log.debug("--- One revolution complete ---")




    parser = argparse.ArgumentParser("RIO Odometer Utility")

    parser.add_argument('-c', '--card', action="store", required=False, help="Card on the RIO")
    parser.add_argument('-a', '--odometer_line_a', action="store", default="", required=False, help="Line A")
    parser.add_argument('-b', '--odometer_line_b', action="store", default="", required=False, help="Line B")
    parser.add_argument('-w', '--wheel', action="store", default=0, type=int, required=False, help="Wheel circumference in mm")
    parser.add_argument('-e', '--encoder', action="store", default=0, type=int, required=False, help="Number of clicks per revolution")
    parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
    parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME,
                        help="Options INI")

    arguments = parser.parse_args()

    # Confirm the YAML file exists
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    lineA = arguments.odometer_line_a
    lineB = arguments.odometer_line_b
    pulsesPerRotation = arguments.wheel
    wheelSize = arguments.encoder

    # Load up the options file.
    options = OptionsFile(arguments.ini)
    if not options.load():
        print("Failed to load options from {}.".format(arguments.ini))
        sys.exit(1)
    else:
        # print("Line A: {}".format(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_LINE_A)))
        # print("Line B: {}".format(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_LINE_B)))
        # print("PULSES: {}".format(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PPR)))
        # print("SIZE: {}".format(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_WHEEL_CIRCUMFERENCE)))

        if len(lineA) == 0:
            try:
                lineA = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PFI_A)
            except KeyError:
                print("Line A must be specified on command line option or in the INI file.")
        if len(lineB) == 0:
            try:
                lineB = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PFI_B)
            except KeyError:
                print("Line B must be specified on command line option or in the INI file.")

        if pulsesPerRotation == 0:
            try:
                pulsesPerRotation = int(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_PPR))
            except KeyError:
                print("Pulses Per Rotation must be specified as command line option or in the INI file.")
        if wheelSize == 0:
            try:
                wheelSize = float(options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_WHEEL_CIRCUMFERENCE))
            except KeyError:
                print("Wheel Size must be specified as command line option or in the INI file.")

    if lineA is None or len(lineA) == 0:
        sys.exit(1)
    if lineB is None or len(lineB) == 0:
        sys.exit(1)
    if wheelSize == 0 or pulsesPerRotation == 0:
        sys.exit(1)

    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("rio")

    log.debug("Using lineA: {} lineB: {} Wheel Size: {} Pulses Per Rotation: {}".format(lineA, lineB, wheelSize, pulsesPerRotation))

    def callback():
        return


    # Needs YAML on rio platform
    # with open(arguments.logging, "rt") as f:
    #     config = yaml.safe_load(f.read())
    #     logging.config.dictConfig(config)

    # Check that the format of the lines is what we expect
    #evalutionText, lines = checkLineNames(arguments.emitter)
    # Is forward clockwise or counter-clockwise?
    forward = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_FORWARD)

    #odometer = PhysicalOdometer(lineA, lineB, wheelSize, pulsesPerRotation, 0, callback)
    theCounter = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_COUNTER)
    odometer = PhysicalOdometer(LINE_A=lineA,
                                LINE_B=lineB,
                                COUNTER=theCounter,
                                WHEEL_SIZE=wheelSize,
                                FORWARD=forward,
                                PULSES=pulsesPerRotation,
                                SPEED=0)

    # Start a thread to service the readings queue
    service = threading.Thread(target = serviceQueue, args=(odometer,))
    service.start()

    # Start a thread to handle user IO. This is not required in normal operation
    io = threading.Thread(target=userIO)
    io.start()

    # Connect the odometer and start.
    odometer.connect()
    # The start routines never return - this is executed in the main thread
    #odometer.startUsingChangeDetection()
    odometer.start()
    #odometer.startEdgeCount()

    sys.exit(0)

