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
import nidaqmx as ni
from nidaqmx import DaqError, DaqWarning
import threading, queue

# Pieces for how NI names their ports
import constants

from collections import deque

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
    def __init__(self):
        plan = []
        return

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

        # Access the plan
        @property
        def plan(self):
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
    def diagnostics(self) -> (bool, str):
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

    def applyTreatment(self, distanceTraveled: int) -> bool:
        print("Apply treatment")
        return True

    def diagnostics(self) -> (bool, str):

        return True, "Emitter passed diagnostics"


#
# A physical emitter expects a NI 9403
#
class PhysicalEmitter(Emitter):

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

    lines = {
        # The left side
        "LEFT11" : "line0", "LEFT21" : "line3", "LEFT31": "line6", "LEFT41": "line9",
        "LEFT12" : "line1", "LEFT22" : "line4", "LEFT32": "line7", "LEFT42": "line10",
        "LEFT13" : "line2", "LEFT23" : "line5", "LEFT33": "line8", "LEFT43": "line11",
        # The right side
        "RIGHT11" : "line12", "RIGHT21" : "line15", "RIGHT31": "line18", "RIGHT41": "line21",
        "RIGHT12" : "line13", "RIGHT22" : "line16", "RIGHT32": "line19", "RIGHT42": "line22",
        "RIGHT13" : "line14", "RIGHT23" : "line17", "RIGHT33": "line20", "RIGHT43": "line23"
    }

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
            line = self.lines.get(key)
        except KeyError as key:
            self._log.error("Unable to find line for Side: {} tier: {} position: {}".format(side.name, tier, position))
        return line

    def on(self, side: constants.Side, tier: int, position: int):
        with ni.Task() as task:
            channel = self.channelNameUsingLineNumber(self.module, self.lineFor(side, tier, position))

            try:
                task.do_channels.add_do_chan(channel)
                task.write(True)
            except DaqError as daq:
                self.log.error("Unable to turn on emitter {} {} {}".format(side, tier, position))

    def diagnostics(self) -> (bool, str):
        diagnosticResult = True
        diagnosticText = "Diagnostic test passed"
        try:
            for side in constants.Side:
                for tier in range(4):
                    for position in range(3):
                        with ni.Task() as task:
                            line = self.lineFor(side, tier + 1, position + 1)
                            #channel = self.channelNameUsingLineNumber(self.module, 0, line)
                            channel = "{}/port{}/{}".format(self.module,0,line)
                            self._log.debug("Add channel: {}".format(channel))
                            task.do_channels.add_do_chan(channel)
                            task.write(True)
                            time.sleep(1)
                            task.write(False)
        except DaqError as daq:
            self._log.fatal("Unable to write to emitters for diagnostics")
            diagnosticText = "Unable to write to the emitters for diagnostics"
            diagnosticResult = False

        return diagnosticResult, diagnosticText



def checkLineNames(line: str) -> (str, int):
    """
    Check that the line designation is valid
    :param line: The line in NI syntax (Mod4/port0/line3)
    """

    lines = []
    elements = line.split("/")
    if len(elements)== 3:
        #print("Module: {}".format(elements[0]))
        #print("Port: {}".format(elements[1]))
        #print("Line: {}".format(elements[2]))

        # Not that flexible, but it the line designation is more than just "line", it must be a range
        if len(elements[2]) > len(NI_LINE):
            # What we expect here is that only the line can be extended with the lineN:M syntax
            # Remove the line and match the range as low:high.
            # I realize that NI accepts reverse order high:low, but I'm too lazy to support that
            range = elements[2].replace('line','')
            lineDescriptors = range.split(":")
            #print("Range: {}".format(lineDescriptors))
            numLines = int(lineDescriptors[1]) - int(lineDescriptors[0]) + 1
            #print("Total lines: {}".format(numLines))

        evaluationText = "Line designation OK"
    else:
        evaluationText = "Expected the line in the form ModN/portN/lineN or ModN/portN/lineN:M"

    return (evaluationText, numLines)


#
# The emitter class as a utility for turning on and off various emitters
#
if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser("RIO Emitter Utility")

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-on', '--on', action="store_true", required=False, default=False, help="Turn the emitters on")
    group.add_argument('-off', '--off', action="store_true", required=False, default=False, help="Turn the emitters off")
    parser.add_argument('-e', '--emitter', action="store", required=True, help="Emitter in NI syntax, i.e., Mod4/port0/line3 or Mod4/port0/line0:5")
    arguments = parser.parse_args()

    # Check that the format of the lines is what we expect
    evalutionText, lines = checkLineNames(arguments.emitter)

    if lines == 0:
        print(evalutionText)
        sys.exit(-1)

    sys.exit(0)


