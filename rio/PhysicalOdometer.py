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

# Direction of travel
# A_LEADS_B = 0
# B_LEADS_A = 1
#
# DIRECTION_FORWARD = 0
# DIRECTION_BACKWARD = 1
# DIRECTION_UNKNOWN = 3

# Not used begin
#
# The transitions we expect to see as the encoder travels back and forward.
#
# KEY_FROM = "from"
# KEY_TO = "to"
# KEY_DIRECTION = "direction"
#
# # Transitions indicating forward or backward travel
# transitions = [
#     {"from": [False, False], "to": [True, False], "direction": DIRECTION_BACKWARD},
#     #{"from": [True, True], "to": [True, False], "direction": DIRECTION_BACKWARD},
#     #{"from": [True, False], "to": [True, True], "direction": DIRECTION_BACKWARD},
#     #{"from": [False, False], "to": [True, False], "direction": DIRECTION_BACKWARD},
#     #{"from": [False, True], "to": [False, False], "direction": DIRECTION_BACKWARD},
# ]

# Clicks of the encoder per revolution
# See https://cdn.automationdirect.com/static/specs/encoderhd.pdf
ENCODER_CLICKS_PER_REVOLUTION = 1000

# Not used end


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

    # def direction(self, previous: [], current: []) -> int:
    #     """
    #     Not used: The direction of travel given the current and previous readings
    #     :param previous: The previous observation
    #     :param current: The current observation
    #     :return: 0 or 1 (FORWARD/BACKWARD)
    #     """
    #
    #     raise NotImplementedError
    #
    #     direction = DIRECTION_UNKNOWN
    #     found = False
    #     i = 0
    #     while not found and i < len(transitions):
    #         if transitions[i].get(KEY_FROM) == previous and transitions[i].get(KEY_TO) == current:
    #             direction = transitions[i].get(KEY_DIRECTION)
    #             found = True
    #         else:
    #             i = i + 1
    #
    #     return direction

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

    # def startUsingCounters(self):
    #     """
    #     Not used. Count edges
    #     """
    #     global running
    #     total = 0.0
    #     self.log.debug("Begin rotation detection using counters")
    #     task = ni.Task(new_task_name="readCtr0")
    #     # Can't seem to make this work, and perhaps this is not what we need, as direction matters
    #     #channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = 'Mod3/ctr0', decoding_type = EncoderType.X_1, zidx_enable=True, units=AngleUnits.DEGREES, pulses_per_rev=4000, initial_angle=0.0)
    #     #channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = 'Mod3/ctr0', decoding_type = EncoderType.X_1, zidx_phase=nidaqmx.constants.EncoderZIndexPhase.AHIGH_BHIGH, zidx_val=0, zidx_enable=True, units=AngleUnits.DEGREES, pulses_per_rev=1000, initial_angle=0.0)
    #     #task.timing.cfg_samp_clk_timing(rate=10000, sample_mode=AcquisitionType.CONTINUOUS,samps_per_chan=100)
    #     #task.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.ON_DEMAND
    #
    #     # This segment works as inted
    #     channelA = task.ci_channels.add_ci_count_edges_chan(counter='Mod3/ctr0')
    #     channelA.ci_count_edges_dig_fltr_min_pulse_width = 0.0003
    #     channelA.ci_count_edges_dig_fltr_enable = True
    #
    #     #task.timing.samp_clk_dig_sync_enable = True
    #     task.timing.samp_clk_overrun_behavior = nidaqmx.constants.OverflowBehavior.TOP_TASK_AND_ERROR
    #     #channelA.ci_encoder_decoding_type = nidaqmx.constants.EncoderType.X_1
    #     #channelA.ci_count_edges_active_edge = nidaqmx.constants.Edge.RISING
    #     task.start()
    #     previous = [0.0]
    #     running = True
    #     while running:
    #         try:
    #             count=task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
    #             print("Current register is {}".format(channelA.ci_count))
    #         except nidaqmx.errors.DaqError:
    #             self.log.error("Read error encountered")
    #             continue
    #
    #         if count[0] != previous[0]:
    #             if count[0] != previous[0] + 1:
    #                 print("Increased count by more than one")
    #             #total += (count[0] - previous[0])
    #             total = count[0]
    #             print("Total movement {:.3f} Total clicks {}".format(total, count[0]))
    #             if total % self._encode_clicks == 0:
    #                 self.log.debug("Wheel revolution complete")
    #         previous=count
    #     print("Cleanup")
    #     task.stop()

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

    # def _isPositive(self, samples: list) -> bool:
    #     """
    #     Not used: Determines if the samples are predominantly False or True.
    #     :param samples: List of booleans
    #     :return: bool
    #     """
    #     trueCount = 0
    #     falseCount = 0
    #
    #     #print(samples)
    #
    #     if(len(samples) == 1):
    #         return samples[0]
    #
    #     for i in range(len(samples)):
    #         if samples[i]:
    #             trueCount += 1
    #         else:
    #             falseCount += 1
    #
    #     return trueCount >= falseCount


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

    # def _start_poll(self):
    #     """
    #     Not used: Start the odometer. This method will not return.
    #     """
    #     self._start = datetime.now()
    #
    #     task = ni.Task()
    #
    #     # These are the three pins on the encoder for a,b, and z
    #     channelA = task.di_channels.add_di_chan("Mod3/port0/line0",line_grouping=LineGrouping.CHAN_PER_LINE)
    #     channelB = task.di_channels.add_di_chan("Mod3/port0/line1",line_grouping=LineGrouping.CHAN_PER_LINE)
    #     #channelZ = task.di_channels.add_di_chan("Mod3/port0/line2")
    #
    #
    #     # Detect the rising edge
    #     #task.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line0,Mod3/Port0/line1", sample_mode=AcquisitionType.CONTINUOUS,samps_per_chan=10)
    #     task.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line0,Mod3/Port0/line1",
    #                                             falling_edge_chan="Mod3/port0/line0,Mod3/Port0/line1",
    #                                             sample_mode=AcquisitionType.CONTINUOUS,
    #                                             samps_per_chan=MAX_ODOMETER_SAMPLES)
    #     task.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION
    #     #task.timing.delay_from_samp_clk_delay = 0.001
    #     #task.timing.delay_from_samp_clk_delay_units = nidaqmx.constants.DigitalWidthUnits.SECONDS
    #     #task.timing.cfg_dig_edge_start_trig(trigger_source="Mod3/port0/line0", trigger_edge=Edge.RISING)
    #
    #     # Debounce the signal
    #     channelA.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
    #     channelB.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
    #     #channelZ.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
    #
    #     channelA.di_dig_fltr_enable = True
    #     channelB.di_dig_fltr_enable = True
    #     #channelZ.di_dig_fltr_enable = True
    #
    #     channels = task.di_channels
    #     print(channels.channel_names)
    #
    #
    #     task.start()
    #     value = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
    #     #value = task.read()
    #     aPrevious = self._isPositive(value[0])
    #     bPrevious = self._isPositive(value[1])
    #     zPrevious = False
    #     i = 0
    #     # The distance traveled in mm
    #     totalClicks = 0
    #     # We want to detect rising to rising as a single click, so the first indicates the start, and the second, the
    #     # end
    #     aArmed = False
    #
    #     while True:
    #         i = i + 1
    #         # This tends to read only one
    #         value = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
    #         # This will wait until all samples are available
    #         # try:
    #         #     value = task.read(number_of_samples_per_channel=MAX_ODOMETER_SAMPLES)
    #         # except nidaqmx.errors.DaqError as daq:
    #         #     print("Wheel is stationary")
    #         #     continue
    #
    #         #value = task.read()
    #         # Determine if the samples are mostly true or false
    #         if(len(value[0]) > 0):
    #             aCurrent = self._isPositive(value[0])
    #             bCurrent = self._isPositive(value[1])
    #             #zCurrent = self._isPositive(value[2])
    #
    #             current = [aCurrent, bCurrent]
    #             previous = [aPrevious, bPrevious]
    #             #direction = self._direction(previous, current)
    #             #print("Direction {}".format(direction))
    #
    #             # Temporary
    #             totalClicks += 1
    #             print("Total distance: {:.2f} mm Total Clicks {:d} aCurrent {} bCurrent {}   "
    #                   .format(totalClicks * self._distance_per_click, totalClicks, aCurrent, bCurrent), end="\n")
    #
    #             # This section will determine if the transition counts as a click
    #             # if aArmed:
    #             #     # This transition was down to up
    #             #     if aCurrent and not aPrevious:
    #             #         totalClicks += 1
    #             #         # print("aCurrent {} bCurrent {} (previous {},{})".format(aCurrent, bCurrent, aPrevious, bPrevious))
    #             #         print("Total distance: {:.2f} mm Total Clicks {:d} aCurrent {} bCurrent {}   ".format(
    #             #             totalClicks * self._distance_per_click, totalClicks, aCurrent, bCurrent), end="\n")
    #             # else:
    #             #     aArmed = True
    #
    #             # z isn't behaving as i expected.  I thought this would generate only a single pulse on every rotation
    #             # if zCurrent != zPrevious:
    #             #     print("Complete rotation: ")
    #             #     print(*value[2])
    #
    #             aPrevious = aCurrent
    #             bPrevious = bCurrent
    #             #zPrevious = zCurrent
    #
    #         #else:
    #         #    print("No samples read")
    #         #print("{} -> {}".format(value[0],aCurrent))

    # def _changeDetected(self, taskHandle, signalType, callbackData):
    #     """
    #     Not used: Called by the NI system when a change is detected on one or more lines
    #     :param taskHandle: Integer task handle.  Ignored.
    #     :param signalType: Signal Type. Ignored.
    #     :param callbackData: Opaque Data. Ignored.
    #     :return: 0
    #     """
    #
    #     self._totalClicks += 1
    #     # For debugging, just print out the number of clicks
    #     #print(self._totalClicks, end='\r')
    #     #return 0
    #     # Get the current readings of the pins
    #     value = self._task.read(number_of_samples_per_channel=1)
    #     # Put the readings into the queue
    #     try:
    #         self._changeQueue.put(value, block=False)
    #     except queue.Full:
    #         print("Queue is full. Reading is lost.")
    #
    #     # This is just for debug.  Comment this out for production
    #     # if self._totalClicks % (self._encode_clicks*2) == 0:
    #     #     print("Rotation complete {}".format(self._totalClicks))
    #     return 0



#     def startUsingChangeDetection(self):
#         """
#         Not used: Use a change detection strategy to determine wheel movement
#         """
#         self._start = datetime.now()
#
#         global running
#         task = ni.Task(new_task_name="ReadA")
#         self._task = task
#
#         # These are the three pins on the encoder for a,b, and z
#         #channelA = task.di_channels.add_di_chan("Mod3/port0/line0")
#         #channelB = task.di_channels.add_di_chan("Mod3/port0/line1")
#         # Orginal
#         channelA = task.di_channels.add_di_chan("Mod3/port0/line0",line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
#         channelB = task.di_channels.add_di_chan("Mod3/port0/line2",line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
#         #channelZ = task.di_channels.add_di_chan("Mod3/port0/line2")
#
#
#         # Works: Rising/Falling on A only, specify only A in detection_timing statement
#         # Works: Rising/Falling on A & B, specify only A in detection_timing statement
#         # Doesn't work: Rising/Falling on A& B, specify both A & B for rising_edge_chan
#         # Doesn't work: Rising/Falling on A& B, specify both A & B in detection_timing statement
#         task.timing.change_detect_di_rising_edge_physical_chans = channelA
#         task.timing.change_detect_di_rising_edge_physical_chans = channelB
#         task.timing.change_detect_di_falling_edge_physical_chans = channelA
#         task.timing.change_detect_di_falling_edge_physical_chans = channelB
#         task.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line0",
#                                                 falling_edge_chan="Mod3/port0/line0",
#                                                 sample_mode=AcquisitionType.CONTINUOUS)
#         # task.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line1",
#         #                                         falling_edge_chan="Mod3/port0/line1",
#         #                                         sample_mode=AcquisitionType.CONTINUOUS)
#         task.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION
#
#         # These are not applicable
#         #channelA.di_dig_fltr_timebase_rate = 100
#         #channelA.di_dig_sync_enable = True
#
#         #channelA.di_dig_sync_enable = True
#         #channelA.di_dig_fltr_timebase_src = "Mod3/port0/line0"
#         #channelA.di_dig_fltr_timebase_rate = 100
#
#         # Debounce the signal
#         channelA.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
#         channelB.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
#         #channelZ.di_dig_fltr_min_pulse_width = ODOMETER_DEBOUNCE
#
#         channelA.di_dig_fltr_enable = True
#         channelB.di_dig_fltr_enable = True
#         #channelZ.di_dig_fltr_enable = True
#
#         #channels = task.di_channels
#         #print(channels.channel_names)
#         # A note of service time requirements:
#         # Register the signal service routine.  The processing callback should be very fast
#         # As the wheel will continue to rotate, another pulse could come in, so the time must be
#         # the width between the falling edge of A/B and the rising edge of A/B
#         # This is 1/4 of the time per pulse. At 1000 pulses per rotation, this will be 0.0002 seconds service,
#         # to support a speed pf 4 kph, or 1.2 rotations per second of a 923 mm circumference wheel
#         task.register_signal_event(nidaqmx.constants.Signal.CHANGE_DETECTION_EVENT, self._changeDetected)
#
#
#         # If a task is not properly stopped, it still holds on to the resources
#         # This is a crude attempt at detecting this and cleaning up.
#         aOK = False
#
#         try:
#             task.start()
#             aOK = True
#         except nidaqmx.errors.DaqError as daq:
#             print("Failure in starting task A. Cleanup")
#             task.close()
#
#         if aOK:
#             # This is only needed for debugging on the bench
#             print("Begin wheel rotation.")
#
#             # The running flag will be set to false by the user input thread, but in production, that will never happen
#             running = True
#             while running:
#                 sleep(5)
#
#             print("Cleaning up")
#             task.close()
#         else:
#             print("Tasks not started.")
#

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

