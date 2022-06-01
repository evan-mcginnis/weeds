#
# P H Y S I C A L  O D O M E T E R
#
# A physical odometer assumes a quadature encoder wheel and reads values from that wheel using
# a National Instruments RIO and a 9411 card
#
import nidaqmx.constants

from Odometer import Odometer
from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime, time

import NIUtilities
import nidaqmx as ni

from nidaqmx.constants import AcquisitionType, TaskMode, CountDirection, Edge
from nidaqmx._task_modules.channels.ci_channel import CIChannel
from nidaqmx._task_modules.channel_collection import ChannelCollection
from nidaqmx.stream_readers import CounterReader
from nidaqmx.constants import AngleUnits
from nidaqmx.constants import EncoderType
from nidaqmx._task_modules.timing import Timing

# The number of pins to read
MAX_ODOMETER_LINES = 4

# Size of the wheel in mm
WHEEL_SIZE = 923

# Direction of travel
A_LEADS_B = 0
B_LEADS_A = 1

DIRECTION_FORWARD = 0
DIRECTION_BACKWARD = 1
DIRECTION_UNKNOWN = 3

#
# The transitions we expect to see as the encoder travels back and forward.
#
KEY_FROM = "from"
KEY_TO = "to"
KEY_DIRECTION = "direction"

transitions = [
    {"from": [True, True],  "to": [True, False],  "direction": DIRECTION_BACKWARD}, #
    {"from": [True, True],  "to": [False, True],  "direction": DIRECTION_BACKWARD}, #
    {"from": [True, True],  "to": [True,False],   "direction": DIRECTION_BACKWARD},
    {"from": [True, False], "to": [True, True],   "direction": DIRECTION_BACKWARD},
    {"from": [False, False],"to": [False,True],   "direction": DIRECTION_BACKWARD},
    {"from": [False, True], "to": [True, True],   "direction": DIRECTION_BACKWARD},
]



# Clicks of the encoder per revolution
# See https://cdn.automationdirect.com/static/specs/encoderhd.pdf
ENCODER_CLICKS_PER_REVOLUTION = 1000


class PhysicalOdometer(Odometer):
    #
    # There is a bit of deviation from the virtual odometer here. The speed indicated in the constructor
    # is the MAXIMUM speed we can tolerate, where the virtual odometer accepts the speed at which the
    # odometer moves.  OK, that could be considered "always moves at the maximum", I suppose.
    #
    def __init__(self, module: str, wheel_size: int, encoder_clicks: int, speed: int, processor: Callable):
        """
        A physical odometer
        :param speed: Maximum speed of movement in meters per second
        :param processor: The image processing routine to callback at each processing step
        """
        # The card on the RIO where the encoder connects
        self._module = module
        self._start = 0
        self._elapsed_milliseconds = 0
        self._elapsed = 0
        # The circumference of the wheel in mm
        self._wheel_size = wheel_size
        # The number of clicks per 1 rotation
        self._encode_clicks = encoder_clicks
        # The distance travelled per click
        self._distance_per_click = wheel_size / encoder_clicks


    def _direction(self, previous: [], current: []) -> int:
        """
        The direction of travel given the current and previous readings
        :param previous: The previous observation
        :param current: The current observation
        :return: 0 or 1 (FORWARD/BACKWARD)
        """
        direction = DIRECTION_UNKNOWN
        found = False
        i = 0
        while not found and i < len(transitions):
            if transitions[i].get(KEY_FROM) == previous and transitions[i].get(KEY_TO) == current:
                direction = transitions[i].get(KEY_DIRECTION)
                found = True
            else:
                i = i + 1

        print("{} -> {} = {}".format(previous, current, direction))

        return direction

    # The connect method is where (presumably) we will connect to the message bus,
    # but that will come later

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
        if self._connected:
            diagnosticResult = True
        return (diagnosticResult, diagnosticText)

    def registerCallback(self,callback):
        self._callback = callback

    # This doesn't do quite what I want, as it returns movement even when the wheel is motionless.
    # And the only thing reported is 0 and -0.360 == sort of correct in that 360 / 1000 is 0.360, but to alternate
    # between 0 and -0.360 is something I can't explain
    def _start(self):
        total = 0.0
        with ni.Task() as task:
            channelA = task.ci_channels.add_ci_ang_encoder_chan(counter = 'Mod3/ctr0', decoding_type = EncoderType.X_1, zidx_val=0, zidx_enable=True, units=AngleUnits.DEGREES, pulses_per_rev=1000, initial_angle=0.0)
            #task.timing.cfg_samp_clk_timing(rate=10.0, sample_mode=AcquisitionType.FINITE,samps_per_chan=100)
            task.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.ON_DEMAND
            channelA.ci_count_edges_count_dir_dig_fltr_min_pulse_width = 0.001
            channelA.ci_count_edges_count_reset_dig_fltr_enable = True
            task.start()
            previous = 0.0
            while True:
                ang=task.read()
                if ang != 0:
                    total += ang
                    print("Total movement{:.3f}".format(total))
                previous=ang

    def _isPositive(self, samples: list) -> bool:
        """
        Determines if the samples are predominantly False or True.
        :param samples: List of booleans
        :return: bool
        """
        trueCount = 0
        falseCount = 0

        #print(samples)

        if(len(samples) == 1):
            return samples[0]

        for i in range(len(samples)):
            if samples[i]:
                trueCount += 1
            else:
                falseCount += 1

        return trueCount >= falseCount

    def start(self):
        """
        Start the odometer. This method will not return.
        """
        self._start = datetime.now()

        task = ni.Task()

        # These are the three pins on the encoder for a,b, and z
        channelA = task.di_channels.add_di_chan("Mod3/port0/line0")
        channelB = task.di_channels.add_di_chan("Mod3/port0/line1")
        channelZ = task.di_channels.add_di_chan("Mod3/port0/line2")


        # Detect the rising edge
        #task.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line0,Mod3/Port0/line1", sample_mode=AcquisitionType.CONTINUOUS,samps_per_chan=10)
        task.timing.cfg_change_detection_timing(rising_edge_chan="Mod3/port0/line0,Mod3/Port0/line1", falling_edge_chan="Mod3/port0/line0,Mod3/Port0/line1", sample_mode=AcquisitionType.CONTINUOUS)
        task.timing.samp_timing_type = nidaqmx.constants.SampleTimingType.CHANGE_DETECTION
        #task.timing.delay_from_samp_clk_delay = 0.001
        #task.timing.delay_from_samp_clk_delay_units = nidaqmx.constants.DigitalWidthUnits(SAMPLE_CLOCK_PERIODS)
        #task.timing.cfg_dig_edge_start_trig(trigger_source="Mod3/port0/line0", trigger_edge=Edge.RISING)

        # Debounce the signal for 0.01 second
        channelA.di_dig_fltr_min_pulse_width = 0.0005
        channelB.di_dig_fltr_min_pulse_width = 0.0005
        channelZ.di_dig_fltr_min_pulse_width = 0.0005

        channelA.di_dig_fltr_enable = True
        channelB.di_dig_fltr_enable = True
        channelZ.di_dig_fltr_enable = True

        channels = task.di_channels
        print(channels.channel_names)


        task.start()
        value = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
        #value = task.read()
        aPrevious = self._isPositive(value[0])
        bPrevious = self._isPositive(value[1])
        zPrevious = False
        i = 0
        # The distance traveled in mm
        totalClicks = 0
        # We want to detect rising to rising as a single click, so the first indicates the start, and the second, the
        # end
        aArmed = False

        while True:
            i = i + 1
            value = task.read(number_of_samples_per_channel=nidaqmx.constants.READ_ALL_AVAILABLE)
            #value = task.read(number_of_samples_per_channel=1)
            #value = task.read()
            # Determine if the samples are mostly true or false
            if(len(value[0]) > 0):
                aCurrent = self._isPositive(value[0])
                bCurrent = self._isPositive(value[1])
                zCurrent = self._isPositive(value[2])

                current = [aCurrent, bCurrent]
                previous = [aPrevious, bPrevious]
                direction = self._direction(previous, current)
                #print("Direction {}".format(direction))

                if aArmed:
                    # This transition was down to up
                    if aCurrent and not aPrevious:
                        totalClicks += 1
                        # print("aCurrent {} bCurrent {} (previous {},{})".format(aCurrent, bCurrent, aPrevious, bPrevious))
                        #print("Total distance: {:.2f} mm Total Clicks {:d} aCurrent {} bCurrent {}   ".format(
                        #    totalClicks * self._distance_per_click, totalClicks, aCurrent, bCurrent), end="\n")
                else:
                    aArmed = True

                # z isn't behaving as i expected.  I thought this would generate only a single pulse on every rotation
                # if zCurrent != zPrevious:
                #     print("Complete rotation: ")
                #     print(*value[2])

                aPrevious = aCurrent
                bPrevious = bCurrent
                zPrevious = zCurrent

            #else:
            #    print("No samples read")
            #print("{} -> {}".format(value[0],aCurrent))


#
# The Odometer class as a utility
#
if __name__ == "__main__":
    import argparse
    import sys

    def callback():
        return

    parser = argparse.ArgumentParser("RIO Odometer Utility")

    parser.add_argument('-c', '--card', action="store", required=True, help="Card on the RIO")
    parser.add_argument('-w', '--wheel', action="store", default=WHEEL_SIZE, required=False, help="Wheel circumference in mm")
    parser.add_argument('-e', '--encoder', action="store", default=ENCODER_CLICKS_PER_REVOLUTION, required=False, help="Number of clicks per revolution")
    arguments = parser.parse_args()

    # Check that the format of the lines is what we expect
    #evalutionText, lines = checkLineNames(arguments.emitter)

    odometer = PhysicalOdometer(arguments.card, arguments.wheel, arguments.encoder, 0, callback)

    odometer.connect()
    odometer.start()

    sys.exit(0)

