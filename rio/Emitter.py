#
# E M I T T E R  A N D  T R E A T M E N T
#
import logging
import time

import numpy as np
from abc import ABC, abstractmethod
# from Treatment import Treatment
# from typing import Callable
# import logging
# from time import sleep
# from datetime import datetime
import threading
import nidaqmx as ni
from nidaqmx import DaqError, DaqWarning
import threading
import queue

# Pieces for how NI names their ports
import constants

from collections import deque

from OptionsFile import OptionsFile

class Treatments:
    def __init__(self):
        self._elements = deque()

    def enqueue(self, element):
        self._elements.append(element)

    def dequeue(self):
        return self._elements.popleft()

NI_PORT = "port"
NI_PORT0 = "port0"
NI_LINE = "line"
NI_SEPARATOR = "/"

MAX_EMITTERS = 24


#
# W A R N I N G
#
# This is temporary until I can get things integrated. This is just for debugging.
#

#
# T R E A T M E N T
#
class Treatment:
    def __init__(self, numberOfEmitters: int):

        # The number of emitters.   Typically this is just one tier on a side, but for the purge operation, it may be
        # all of the emitters
        self._numberOfEmitters = numberOfEmitters
        self._plan = [False] * numberOfEmitters

    #
    # G E N E R A T E  S A M P L E  P L A N
    #
    # This is something that will not be used in production. Only for testing
    #
    @classmethod
    def generateDummyPlan(self):
        self._plan = [
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True],
                [False, False, False, False, False, False],
                [True, True, True, True, True, True]
        ]

    def create(self, value: bool) -> []:
        """
        Create a treatment plan with the value specified.
        :param value: True = turn on emitters, False = turn off emitters
        """
        for i in range(self._numberOfEmitters):
            self._plan[i] = value
        return self._plan

    # Access the plan
    @property
    def plan(self) -> []:
        """
        The current plan
        :return: A list of booleans
        """
        return self._plan


# End warning/test code

class Emitter(ABC):
    def __init__(self, module: str):
        """
        An emitter that can be controlled from a specific module.
        This will currently not correspond to a specific set of pins within the module
        :param module: A string name of the module -- usually something like Mod4
        """
        # a list of all the treatments.  I suspect this will never be greater than two
        self._treatments = Treatments()
        self._module = module
        self._log = logging.getLogger(__name__)



        return

    @property
    def module(self):
        return self._module

    @abstractmethod
    def applyTreatment(self, distanceTraveled: int) -> bool:
        """
        Apply the treatment according to the distance traveled.
        :param distanceTraveled: The distance traveled in cm
        """
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self, turnOnEmitter: bool) -> (bool, str):
        """
        Run diagnostics on the emitter array.
        """
        raise NotImplementedError

    #@abstractmethod
    def add(self, plan: Treatment) -> bool:
        """
        Add the plan to the list of plans to process
        :param plan:
        :return:
        """
        self._treatments.enqueue(plan)
        return True

    @staticmethod
    def channelNameUsingLineNumber(module: str, port: int, line: int) -> str:
        """
        Channel name in NI format, i.e., Mod4/0/line3
        :param module:
        :param port:
        :param line:
        :return: Channel name in NI format
        """
        return module + NI_SEPARATOR + NI_PORT + str(port) + NI_SEPARATOR + NI_LINE + str(line)

    @staticmethod
    def channelNameUsingLineName(module: str, port: int, line: str) -> str:
        return module + NI_SEPARATOR + NI_PORT + str(port) + NI_SEPARATOR + line
#
# A Virtual emitter is one that does not correspond to any specific hardware
#

class VirtualEmitter(Emitter):
    def __init__(self, module: str):
        super().__init__(module)
    def applyTreatment(self, distanceTraveled: int) -> bool:
        print("Apply treatment")
        return True

    def diagnostics(self, turnOnEmitter: bool) -> (bool, str):

        return True, "Emitter passed diagnostics"


#
# A physical emitter expects a NI 9403
#
class PhysicalEmitter(Emitter):

    def __init__(self, module: str):
        super().__init__(module)

    #
    # A P P L Y  T R E A T M E N T
    #
    def applyTreatment(self, distanceTravelled: int) -> bool:
        """
        Apply the treatment for the specified distance.
        :param distanceTravelled: The distance covered
        :param treatment: The treatment plan
        """
        #
        # For testing, if we have completed one plan, just generate a new one
        #
        sample = [True, False]
        with ni.Task() as task:
            task.do_channels.add_do_chan('Mod4/port0/line1')
            task.do_channels.add_do_chan('Mod4/port0/line2')
            task.do_channels.add_do_chan('Mod4/port0/line3')
            task.do_channels.add_do_chan('Mod4/port0/line4')
            task.do_channels.add_do_chan('Mod4/port0/line5')
            task.do_channels.add_do_chan('Mod4/port0/line6')
            #task.do_channels.add_do_chan('Mod4/port0')

            print('1 Channel 6 Sample Write: ')
            emitterValues = np.random.choice(sample, size=6)
            print(task.write(emitterValues))
            #print(task.write([True,True,True]))
            time.sleep(10)
            emitterValues = np.random.choice(sample, size=6)
            print(task.write(emitterValues))
            #print(task.write([False,False,False]))
            time.sleep(2)

        return True


    # These are the line assignments for the emitters. Note that these are NOT the pin assignments

    linesLeft = {
        "LEFT11": "line0", "LEFT21": "line3", "LEFT31": "line6", "LEFT41": "line9",
        "LEFT12": "line1", "LEFT22": "line4", "LEFT32": "line7", "LEFT42": "line10",
        "LEFT13": "line2", "LEFT23": "line5", "LEFT33": "line8", "LEFT43": "line11",
    }

    linesRight = {
        "RIGHT11": "line12", "RIGHT21": "line15", "RIGHT31": "line18", "RIGHT41": "line21",
        "RIGHT12": "line13", "RIGHT22": "line16", "RIGHT32": "line19", "RIGHT42": "line22",
        "RIGHT13": "line14", "RIGHT23": "line17", "RIGHT33": "line20", "RIGHT43": "line23"
    }

    # All the lines in the system
    lines = {constants.Side.LEFT.name: linesLeft, constants.Side.RIGHT.name: linesRight}

    def lineFor(self, side: constants.Side, tier: int, position: int) -> str:
        """
        Gets the line for the associatated emitter given the side, tier, and position
        :param side: Side.LEFT or Side.RIGHT
        :param tier: 1-4, 4 is closest to tractor, 1 is furthest
        :param position: 1-3, 1 is closest to left when facing rear of tractor
        """
        line = ""
        key = side.name + str(tier) + str(position)
        try:
            line = self.lines[side.name].get(key)
        except KeyError as key:
            self._log.error("Unable to find line for Side: {} tier: {} position: {}".format(side.name, tier, position))
        return line

    def beginPreparations(self):
        """
        Start preparing for an operation on the emitters.
        """
        self._task = ni.Task()

    def cleanup(self):
        """
        Clean up the NI write tasks.  Call this before ending the program
        """
        self._task.close()

    def addAllEmitters(self, side: constants.Side):
        """
        Add all the emitters for a specified side
        :param side:
        """
        for emitterName, line in self.lines[side.name].items():
            channel = "{}/port{}/{}".format(self.module, 0, line)
            self._task.do_channels.add_do_chan(channel)

    def addEmitter(self, side: constants.Side, tier: int, position: int):
        """
        Add the emitter to the operation
        :param side: The side of the emitter, left or right
        :param tier: The tier of the emitter (row)
        :param position:  The position of the emitter (column)
        """
        line = self.lineFor(side, tier, position)
        channel = "{}/port{}/{}".format(self.module, 0, line)
        self._task.do_channels.add_do_chan(channel)

    def turnOffEmitters(self, cleanup: bool = False):
        """
        Turn off the emitters that have been previously added with addEmitter()
        """
        treatment = Treatment(len(self._task.channel_names))
        treatment.create(False)

        try:
            self._task.write(treatment.plan)

            # For a simple purge, this will release the resources held by the task
            if cleanup:
                self.cleanup()

        except DaqError as daq:
            self._log.error("Unable to turn off the emitters")


    def turnOnEmitters(self, _duration: float = 0):
        """
        Turn on the emitters that have been previously added with addEmitter()
        :param _duration: (optional) The duration in seconds.  Use 0 to leave the emitters on until explicitly turned off
        """

        treatment = Treatment(len(self._task.channel_names))
        treatment.create(True)

        try:
            self._task.write(treatment.plan)
            self._task.start()
        except DaqError as daq:
            self._log.error("Unable to turn on the emitter: {}".format(daq))

        # If we are turning on the emitters for a specific interval, set up a timer to turn them off
        if _duration > 0:
            off = lambda: self.turnOffEmitters(True)
            delayed = threading.Timer(_duration, off)
            delayed.start()
        else:
            self._task.stop()


    def on(self, side: constants.Side, tier: int, position: int, time: float):
        with ni.Task() as task:
            line = self.lineFor(side, tier, position)
            channel = "{}/port{}/{}".format(self.module, 0, line)
            self._log.debug("Add channel: {}".format(channel))

            try:
                task.do_channels.add_do_chan(channel)
                task.write(True)
            except DaqError as daq:
                self._log.error("Unable to turn on emitter {} {} {}".format(side, tier, position))

            # If the emitter will be on only for a specific interval, set up a callback to turn it off
            if time > 0:
                off = lambda: self.off(side, tier, position)
                delayed = threading.Timer(time, off)
                delayed.start()
    def off(self, side: constants.Side, tier: int, position: int):
        with ni.Task() as task:
            line = self.lineFor(side, tier, position)
            channel = "{}/port{}/{}".format(self.module, 0, line)
            self._log.debug("Add channel: {}".format(channel))

            try:
                task.do_channels.add_do_chan(channel)
                task.write(False)
            except DaqError as daq:
                self._log.error("Unable to turn off emitter {} {} {}".format(side, tier, position))

    def diagnostics(self, turnOnSprayer: bool) -> (bool, str):
        """
        Perform diagnostics on the emitters
        :param turnOnSprayer: TRUE if this is to be a wet test, FALSE otherwise
        :return: (bool to indicate diagnostic pass, str to provide details)
        """
        diagnosticResult = True
        diagnosticText = "Diagnostic test passed"
        try:
            for side in constants.Side:
                for tier in range(4):
                    for position in range(3):
                        with ni.Task() as task:
                            line = self.lineFor(side, tier + 1, position + 1)
                            #channel = self.channelNameUsingLineNumber(self.module, 0, line)
                            channel = "{}/port{}/{}".format(self.module, 0, line)
                            self._log.debug("Add channel: {}".format(channel))
                            try:
                                task.do_channels.add_do_chan(channel)
                            except DaqError as daq:
                                diagnosticText = "Unable to add channel: {}".format(daq)
                                self._log.fatal(diagnosticText)
                                diagnosticResult = False
                                diagnosticText = diagnosticText
                            # Don't turn on the emitters in the diagnostics in the field
                            if turnOnSprayer:
                                task.write(True)
                                time.sleep(0.5)
                                task.write(False)
        except DaqError as daq:
            self._log.fatal("Unable to write to emitters for diagnostics")
            diagnosticText = "Unable to write to the emitters for diagnostics"
            diagnosticResult = False

        return diagnosticResult, diagnosticText






#
# The emitter class as a utility for turning on and off various emitters
#
if __name__ == "__main__":
    import argparse
    import sys
    import re
    import nidaqmx.utils
    parser = argparse.ArgumentParser("RIO Emitter Utility")

    group = parser.add_mutually_exclusive_group()

    emitterGroup = parser.add_argument_group("Specify a specific emitter")
    #emitterSpecified = parser.add_mutually_exclusive_group()

    group.add_argument('-on', '--on', action="store_true", required=False, default=False, help="Turn the emitters on")
    group.add_argument('-off', '--off', action="store_true", required=False, default=False, help="Turn the emitters off")
    parser.add_argument('-s', '--side', action="store", required=True, choices=["left", "right"], help="Left or right side")
    emitterGroup.add_argument('-t', '--tier', action="store", required=False, type=int, help="Tier")
    emitterGroup.add_argument('-p', '--position', action="store", required=False, type=int, help="Position within tier")
    parser.add_argument('-a', '--all', action="store_true", required=False, default=False, help="Turn on all emitters for side specified")
    parser.add_argument('-d', '--duration', action="store", required=False, default=5, type=int, help="Duration for emitter to be turned on.  Use 0 to leave emitter on.")
    parser.add_argument('-e', '--emitter', action="store", required=False, help="Emitter in NI syntax, i.e., Mod4/port0/line3 or Mod4/port0/line0:5")
    arguments = parser.parse_args()


    if arguments.side.upper() == constants.Side.RIGHT.name:
        side = constants.Side.RIGHT
    elif arguments.side.upper() == constants.Side.LEFT.name:
        side = constants.Side.LEFT
    else:
        print("Side must be LEFT or RIGHT")
        sys.exit(-1)

    emitter = PhysicalEmitter('Mod4')
    emitter.beginPreparations()

    if arguments.all:
        emitter.addAllEmitters(side)
    else:
        emitter.addEmitter(side, arguments.tier, arguments.position)

    emitter.turnOnEmitters(arguments.duration)
    time.sleep(arguments.duration)
    emitter.cleanup()

    # emitter.on(side, arguments.tier, arguments.position, arguments.duration)

    # def checkLineNames(line: str) -> (str, int):
    #     """
    #     Check that the line designation is valid
    #     :param line: The line in NI syntax (Mod3/port0/line3)
    #     """
    #
    #     numLines = 1
    #     evaluationText = "Expected the line in the form ModN/portN/lineN or ModN/portN/lineN:M"
    #
    #     elements = line.split("/")
    #
    #     if len(elements) == 3:
    #         print("Module: {}".format(elements[0]))
    #         print("Port: {}".format(elements[1]))
    #         print("Line: {}".format(elements[2]))
    #
    #
    #         rangeOfLines = re.match(r'line[0-9]+\:[0-9]+', elements[2])
    #         #channels = nidaqmx.utils.flatten_channel_string(elements[2])
    #
    #
    #         # Not that flexible, but if the line designation is more than just "line", it must be a range
    #         if rangeOfLines:
    #             # What we expect here is that only the line can be extended with the lineN:M syntax
    #             # Remove the line and match the range as low:high.
    #             # I realize that NI accepts reverse order high:low, but I'm too lazy to support that
    #             range = elements[1].replace('line', '')
    #             lineDescriptors = range.split(":")
    #             # print("Range: {}".format(lineDescriptors))
    #             numLines = int(lineDescriptors[0]) - int(lineDescriptors[0]) + 1
    #             # print("Total lines: {}".format(numLines))
    #             evaluationText = "Line designation OK"
    #
    #     return evaluationText, numLines
    #
    # # Check that the format of the lines is what we expect
    # evalutionText, lines = checkLineNames(arguments.emitter)

    # if lines == 0:
    #     print(evalutionText)
    #     sys.exit(-1)

    sys.exit(0)


