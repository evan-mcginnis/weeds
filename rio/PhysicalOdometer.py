#
# P H Y S I C A L  O D O M E T E R
#
# A physical odometer assumes a quadatrue encoder wheel and reads values from that wheel using
# a National Instruments RIO and a 9411 card
#

from Odometer import Odometer
from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime

import NIUtilities
import nidaqmx as ni

# The number of pins to read
MAX_ODOMETER_LINES = 4

# Size of the wheel in mm
WHEEL_SIZE = 921

# Direction of travel
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
    {"from": [False, True],  "to": [True, True],   "direction": DIRECTION_FORWARD},
    {"from": [True, True],   "to": [True, False],  "direction": DIRECTION_FORWARD},
    {"from": [True, False],  "to": [False, False], "direction": DIRECTION_FORWARD},
    {"from": [False, False], "to": [False, True],  "direction": DIRECTION_FORWARD},
    {"from": [True, True],   "to": [False, True],  "direction": DIRECTION_BACKWARD},
    {"from": [True, False],  "to": [True, True],   "direction": DIRECTION_BACKWARD},
    {"from": [False, False], "to": [True, False],  "direction": DIRECTION_BACKWARD},
    {"from": [False, True],  "to": [False, False], "direction": DIRECTION_BACKWARD},
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
        while not found and i <= len(transitions):
            if transitions[i].get(KEY_FROM) == previous and transitions[i].get(KEY_TO):
                direction = transitions[i].get(KEY_DIRECTION)
                found = True
            else:
                i = i + 1

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

    def start(self):
        """
        Start the odometer. This method will not return.
        """
        self._start = datetime.now()

        # with ni.Task() as task:
        #     for line in range(1, MAX_ODOMETER_LINES + 1):
        #         # Form a channel descriptor line "Mod4/port0/line3"
        #         channel = self.channelName(self.module, 0, line)
        #         task.di_channels.add_di_chan(channel)
        task = ni.Task()

        # # Use the counting channel methods on RIO
        # channelA = task.ci_channels.add_ci_chan("Mod3/port0/line1")
        # channelB = task.ci_channels.add_ci_chan("Mod3/port0/line1")
        # channelZ = task.ci_channels.add_ci_chan("Mod3/port0/line2")
        # task.ci_channels.ci_ang_encoder_pulses_per_rev = self._encode_clicks

        # These are the three pins on the encoder for a,b, and z
        channelA = task.di_channels.add_di_chan("Mod3/port0/line0")
        channelB = task.di_channels.add_di_chan("Mod3/port0/line1")
        channelZ = task.di_channels.add_di_chan("Mod3/port0/line2")
        #
        # channelA.di_dig_fltr_min_pulse_width = 1
        # channelB.di_dig_fltr_min_pulse_width = 1
        # channelZ.di_dig_fltr_min_pulse_width = 1
        # channelA.di_dig_fltr_enable = True
        # channelB.di_dig_fltr_enable = True
        # channelZ.di_dig_fltr_enable = True
        #
        # channels = task.di_channels
        #print(channels.channel_names)

        task.start()

        value = task.read()
        aPrevious = value[0]
        bPrevious = value[1]
        i = 0
        # The distance traveled in mm
        totalClicks = 0
        while True:
            i = i + 1
            value = task.read()
            aCurrent = value[0]
            bCurrent = value[1]
            if (aCurrent != aPrevious) or (bCurrent != bPrevious):
                # We have moved 1 click
                totalClicks = totalClicks + 1
                print("Total distance: {:.2f} mm".format(totalClicks * self._distance_per_click),end="\r")
            #else:
                # print("{}".format(i),end="\r")
            aPrevious = aCurrent
            bPrevious = bCurrent

        # Call the processor based on distance
        # while self._processor(1):
        #     self._elapsed = datetime.now() - self._start
        #     self._elapsed_milliseconds = self._elapsed.total_seconds() * 1000
        #
        #
        #     # Call the processing routine every 1cm of travel
        #     self._log.debug("Sleep for {:.2f} seconds".format(self._timeToTravel1CM))
        #     sleep(self._timeToTravel1CM)
        #     self._start = datetime.now()

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

