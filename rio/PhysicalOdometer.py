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

from Odometer import Odometer
from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime, time
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
    # TODO: Simplfy this by accepting a dictionary of values for the parameters.  These should all be in the INI file
    def __init__(self, lineA: str, lineB: str, wheel_size: int, encoder_clicks: int, speed: int, processor: Callable):
        """
        A physical odometer
        :param speed: Maximum speed of movement in meters per second
        :param processor: The image processing routine to callback at each processing step
        """
        # The card on the RIO where the encoder connects
        super().__init__("")

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

        # This is where the readings go -- this size is way too big
        self._changeQueue = queue.Queue(maxsize=5000)

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

        # Not used
        self._aArmed = False

    @property
    def changeQueue(self):
        """
        The queue of readings for the line transitions of the input pins
        :return:
        """
        return self._changeQueue

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
        # TODO: Complete diagnostics
        # Check to see if the 9411 has incoming power

        if self._connected:
            diagnosticResult = True
        return (diagnosticResult, diagnosticText)

    def registerCallback(self,callback):
        self._callback = callback


    def startUsingAngularEncoder(self):
        """
        Treat the encoder as an angular encoder, putting new readings on the change queue
        """
        global running
        total = 0.0
        self.log.debug("Begin rotation detection using Angular Encoding")
        task = ni.Task(new_task_name="readCtr0")
        #channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = 'Mod3/ctr0', decoding_type = EncoderType.X_1, zidx_enable=True, units=AngleUnits.DEGREES, pulses_per_rev=1000, initial_angle=0.0)

        # Add the channel as an angular encoder without Z index support, as we really don't care about the number of rotations
        channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = 'Mod3/ctr0', decoding_type = EncoderType.X_1, zidx_enable=False, zidx_val=0, units=AngleUnits.DEGREES, pulses_per_rev=self.encoderClicksPerRevolution, initial_angle=0.0)


        channelA.ci_encoder_a_input_dig_fltr_min_pulse_width = 0.001
        channelA.ci_encoder_a_input_dig_fltr_enable = True
        channelA.ci_encoder_a_input_term = 'PFI0'
        channelA.ci_encoder_b_input_dig_fltr_min_pulse_width = 0.001
        channelA.ci_encoder_b_input_dig_fltr_enable = True
        channelA.ci_encoder_b_input_term = 'PFI1'
        channelA.ci_encoder_z_input_dig_fltr_min_pulse_width = 0.001
        channelA.ci_encoder_z_input_dig_fltr_enable = True
        channelA.ci_encoder_z_input_term = 'PFI2'

        #task.timing.samp_clk_overrun_behavior = nidaqmx.constants.OverflowBehavior.TOP_TASK_AND_ERROR

        task.start()
        previous = 0.0
        running = True

        # This loop will run until things are gracefully shut down by another thread
        # setting running to False.
        while running:
            try:
                ang =task.read(number_of_samples_per_channel = 1) #nidaqmx.constants.READ_ALL_AVAILABLE)
                #print("Current register is {}".format(channelA.ci_count))
            except nidaqmx.errors.DaqError:
                self.log.error("Read error encountered")
                continue

            # If the current reading has changed from the previous reading, the wheel has moved
            if ang[0] != 0 and ang[0] != previous:
                try:
                    # Put the current angular reading in the queue
                    self._changeQueue.put(float(ang[0]), block=False)
                except queue.Full:
                    self.log.error("Distance queue is full. Reading is lost.")
                previous = ang[0]

        task.stop()


    # Keep this around for now, as that is the only way to read the currently installed encoder
    # that is useful for edge counting and not much else

    def startEdgeCount(self):
        """
        Not used: Use an edge count strategy to determine wheel movement

        """
        taskA = ni.Task(new_task_name="readA")
        taskB = ni.Task(new_task_name="readB")

        channelA = taskA.di_channels.add_di_chan("Mod3/port0/line0",line_grouping=LineGrouping.CHAN_PER_LINE)
        channelB = taskB.di_channels.add_di_chan("Mod3/port0/line2",line_grouping=LineGrouping.CHAN_PER_LINE)
        taskA.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line0",
                                                falling_edge_chan="Mod3/port0/line0",
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=MAX_ODOMETER_SAMPLES)
        taskA.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION

        taskB.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line2",
                                                falling_edge_chan="Mod3/port0/line2",
                                                sample_mode=AcquisitionType.CONTINUOUS,
                                                samps_per_chan=MAX_ODOMETER_SAMPLES)
        taskB.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION

        # Debounce the signal
        channelA.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
        channelB.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
        #channelZ.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE

        channelA.di_dig_fltr_enable = True
        channelB.di_dig_fltr_enable = True
        #channelZ.di_dig_fltr_enable = True

        taskA.timing.change_detect_di_rising_edge_physical_chans = channelA
        taskB.timing.change_detect_di_rising_edge_physical_chans = channelB
        taskA.timing.change_detect_di_falling_edge_physical_chans = channelA
        taskB.timing.change_detect_di_falling_edge_physical_chans = channelB

        taskA.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION
        taskB.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION

        taskA.register_signal_event(nidaqmx.constants.Signal.CHANGE_DETECTION_EVENT, self._changeDetected)
        taskB.register_signal_event(nidaqmx.constants.Signal.CHANGE_DETECTION_EVENT, self._changeDetected)


        aOK = False
        bOK = False

        try:
            print("Task A uses:{}".format(taskA.channel_names))
            taskA.start()
            aOK = True
        except nidaqmx.errors.DaqError as daq:
            print("Failure in starting task A. Cleanup")
            taskA.close()
        try:
            print("Task B uses:{}".format(taskB.channel_names))
            taskB.start()
            bOK = True
        except nidaqmx.errors.DaqError as daq:
            print("Failure in starting task B. {}".format(daq))
            taskA.close()


        if aOK and bOK:
            # This is only needed for debugging on the bench
            print("Begin wheel rotation.")

            # The running flag will be set to false by the user input thread, but in production, that will never happen
            running = True
            while running:
                sleep(5)

            print("Cleaning up")
            taskA.close()
            taskB.close()
        else:
            print("Tasks not started.")


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

        # Loop until graceful exit.
        i = 0
        while servicing:
            angle = changeQueue.get(block=True)
            distanceTraveled = (angle - previous) * odometer.distancePerDegree
            totalDistanceTraveled += distanceTraveled
            previous = angle
            print("{:.4f} mm Total: {:.4f}".format(distanceTraveled, totalDistanceTraveled))
            i += 1
            # Determine if the wheel has undergone one rotation
            if i % odometer.encoderClicksPerRevolution == 0:
                print("--- One revolution complete ---")




    parser = argparse.ArgumentParser("RIO Odometer Utility")

    parser.add_argument('-c', '--card', action="store", required=False, help="Card on the RIO")
    parser.add_argument('-a', '--odometer_line_a', action="store", default="", required=False, help="Line A")
    parser.add_argument('-b', '--odometer_line_b', action="store", default="", required=False, help="Line B")
    parser.add_argument('-w', '--wheel', action="store", default=0, type=int, required=False, help="Wheel circumference in mm")
    parser.add_argument('-e', '--encoder', action="store", default=0, type=int, required=False, help="Number of clicks per revolution")
    parser.add_argument("-lg", "--logging", action="store", default="info-logging.ini", help="Logging configuration file")
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
                lineA = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_LINE_A)
            except KeyError:
                print("Line A must be specified on command line option or in the INI file.")
        if len(lineB) == 0:
            try:
                lineB = options.option(constants.PROPERTY_SECTION_ODOMETER, constants.PROPERTY_LINE_B)
            except KeyError:
                print("Line B must be specified on command line option or in the INI file.")

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

    if lineA is None or len(lineA) == 0:
        sys.exit(1)
    if lineB is None or len(lineB) == 0:
        sys.exit(1)
    if wheelSize == 0 or pulsesPerRotation == 0:
        sys.exit(1)

    print("Using lineA: {} lineB: {} Wheel Size: {} Pulses Per Rotation: {}".format(lineA, lineB, wheelSize, pulsesPerRotation))

    def callback():
        return

    #fileConfig(arguments.logging)

    # Needs YAML on rio platform
    # with open(arguments.logging, "rt") as f:
    #     config = yaml.safe_load(f.read())
    #     logging.config.dictConfig(config)

    # Check that the format of the lines is what we expect
    #evalutionText, lines = checkLineNames(arguments.emitter)

    odometer = PhysicalOdometer(lineA, lineB, wheelSize, pulsesPerRotation, 0, callback)


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
    odometer.startUsingAngularEncoder()
    #odometer.startEdgeCount()

    sys.exit(0)

