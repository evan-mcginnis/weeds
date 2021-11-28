#
# E M I T T E R  A N D  T R E A T M E N T
#

import time

import numpy as np
from abc import ABC, abstractmethod
# from Treatment import Treatment
# from typing import Callable
# import logging
# from time import sleep
# from datetime import datetime
import nidaqmx as ni
import threading, queue

# Pieces for how NI names their ports
NI_PORT = "port"
NI_PORT0 = "port0"
NI_LINE = "line"
NI_SEPARATOR = "/"

MAX_EMITTERS = 6

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
        self._treatments = queue.SimpleQueue()
        self._module = module
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
        self._treatments.put(plan)
        return True

    @staticmethod
    def channelName(module: str, port: int, line: int) -> str:
        return module + NI_SEPARATOR + NI_PORT + str(port) + NI_SEPARATOR + NI_LINE + str(line)
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

    def diagnostics(self) -> (bool, str):
        """
        Execute emitter diagnostics.
        :return: False on failure
        """

        # Arrays for turning all the lines on and off
        diagnosticsOn = [True] * MAX_EMITTERS
        diagnosticsOff = [False] * MAX_EMITTERS

        with ni.Task() as task:
            for line in range(1, MAX_EMITTERS + 1):
                # Form a channel descriptor line "Mod4/port0/line3"
                channel = self.channelName(self.module, 0, line)
                task.do_channels.add_do_chan(channel)

            # Turn off all the emitters as cleanup
            task.write(diagnosticsOff)

            # Not much of a diagnostic here -- just turn the emitters on and off
            diagnosticResult = True
            diagnosticText = "Emitter diagnostics passed"
            try:
                for i in range(5):
                    task.write(diagnosticsOn)
                    time.sleep(1)
                    task.write(diagnosticsOff)
                    time.sleep(1)
            except ni.errors.DaqError:
                diagnosticResult = False
                diagnosticText = "Error encountered in NI: "


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


