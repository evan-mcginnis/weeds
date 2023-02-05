#
# O D O M E T E R
#

from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime
import queue
import constants

class Odometer(ABC):
    def __init__(self, options: str):
        self.options = options
        # This is where the readings go -- this size is way too big
        self._changeQueue = queue.Queue(maxsize=50000)
        self._source = None
        self._processing = False
        self._reporting = False
        self._connected = False

    @property
    def reporting(self) -> bool:
        return self._reporting

    @reporting.setter
    def reporting(self, report: bool):
        self._reporting = report

    @property
    def changeQueue(self):
        """
        The queue of readings for the line transitions of the input pins
        :return:
        """
        return self._changeQueue

    @abstractmethod
    def connect(self) -> bool:
        raise NotImplementedError()
        return True

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError()
        return True

    @abstractmethod
    def diagnostics(self):
        self._connected = False
        return 0

    @abstractmethod
    def registerCallback(self,callback):
        raise NotImplementedError()

    @abstractmethod
    def start(self):
        raise NotImplementedError()

    def stop(self):
        # Set the flag that will have the side effect of the odometry stopping.
        self._processing = False

    @property
    def source(self):
        return self._source


class VirtualOdometer(Odometer):
    def __init__(self, **kwargs):
        """
        An odometer that behaves as if the system is moving at a constant speed.
        :param kwargs: SPEED -- the speed expressed in KPH
        """
        super().__init__("")
        self._wheel_size = kwargs[constants.KEYWORD_WHEEL_CIRCUMFERENCE]
        self._encoder_clicks = kwargs[constants.KEYWORD_PPR]
        self._speed = kwargs[constants.KEYWORD_SPEED]

        self._speedInMMPerSecond = self._speed * 1000
        self._log = logging.getLogger(__name__)
        # For example, it takes 0.00224 seconds to travel 1cm at 44.704 cm per second/1mph
        self._timeToTravel1CM = 1 / self._speed

        self._distancePerDegree = self._wheel_size / 360

        self._start = 0
        self._elapsed_milliseconds = 0
        self._elapsed = 0

        self._source = constants.SOURCE_VIRTUAL

        self._log = logging.getLogger(__name__)

    # TODO: Move to superclass
    @property
    def distancePerDegree(self) -> float:
        return self._distancePerDegree

    # TODO: Move to superclass
    @property
    def encoderClicksPerRevolution(self) -> int:
        """
        The number of clicks per revolution
        :return:
        """
        return self._encoder_clicks

    def connect(self) -> bool:
        """
        Connect to the encoder -- meaningless on a virtual encoder
        :return: True
        """
        self._connected = True
        self.reporting = True
        return True

    def disconnect(self):
        """
        Disconnect from the encoder -- meaningless on a virtual encoder
        :return: True
        """
        self._connected = False
        self.reporting = False
        return True

    def diagnostics(self):
        """
        Perform diagnostics on the encoder -- meaningless on a virtual encoder
        :return: (True, "Passed")
        """
        self._connected = False
        return True, "Odometer diagnostics passed"

    def registerCallback(self,callback):
        return

    def start(self):

        self._log.debug("Begin virtual odometry")
        self._start = datetime.now()

        self._processing = True
        angle = 0.0
        # Determine how many wheel rotations we will have in an hour at that pace
        # Convert the speed (KM) per hour to MM per hour, as the wheel size is stated in mm
        rotationsInTargetSpeed = (self._speed * 1e6) / self._wheel_size
        # Determine how many pulses we will have in an hour at that pace
        pulsesAtTargetSpeed = self._encoder_clicks * rotationsInTargetSpeed

        # Determine the pulses in 1 second at target speed
        pulsesPerSecondAtTargetSpeed = pulsesAtTargetSpeed / (60 * 60)

        timeToMoveOnePulse = 1 / pulsesPerSecondAtTargetSpeed

        while self._processing:
            self._elapsed = datetime.now() - self._start
            self._elapsed_milliseconds = self._elapsed.total_seconds() * 1000

            if self.reporting:
                # put the angular change on the queue
                angle += (360 / self._encoder_clicks)
                try:
                    self.changeQueue.put(angle, block=False)
                    self._log.info("Queue size: {} Angle: {}".format(self.changeQueue.qsize(), angle))
                except queue.Full as full:
                    self._log.fatal("Odometry queue is full. This should not happen to a double ended queue")
                    self._log.fatal("Current queue size: {}".format(self.changeQueue.qsize()))

                # Call the processing routine every 1cm of travel
                #self._log.debug("Sleep for {:.5f} seconds".format(timeToMoveOnePulse))
            else:
                self._log.error("Movement not reported, so will not be enqueued")
            self._log.debug("Sleep: {}".format(timeToMoveOnePulse))
            sleep(timeToMoveOnePulse)
            self._start = datetime.now()

if __name__ == "__main__":
    import argparse
    import sys
    import os
    import threading
    import time

    from logging.config import fileConfig

    import constants
    from OptionsFile import OptionsFile


    running = False
    def userIO():
        global running
        input("Return to stop")
        running = False

    def nanoseconds() -> int:
        return time.time_ns()

    def serviceQueue(odometer : VirtualOdometer):
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

            log.debug("{:.4f} mm Total: {:.4f} Elapsed Time {} ns".format(distanceTraveled, totalDistanceTraveled, elapsed))

            i += 1
            # Determine if the wheel has undergone one rotation
            if i % odometer.encoderClicksPerRevolution == 0:
                log.debug("--- One revolution complete ---")




    parser = argparse.ArgumentParser("Virtual Odometer Utility")

    parser.add_argument('-s', '--speed', action="store", type=int, required=True, help="Speed in KPH")
    parser.add_argument('-w', '--wheel', action="store", default=0, type=int, required=False, help="Wheel circumference in mm")
    parser.add_argument('-e', '--encoder', action="store", default=0, type=int, required=False, help="Number of clicks per revolution")
    parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
    parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME,
                        help="Options INI")

    arguments = parser.parse_args()

    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    logging.config.fileConfig(arguments.logging)
    log = logging.getLogger("virtual-odometer")

    pulsesPerRotation = arguments.wheel
    wheelSize = arguments.encoder

    # Load up the options file.
    options = OptionsFile(arguments.ini)
    if not options.load():
        print("Failed to load options from {}.".format(arguments.ini))
        sys.exit(1)
    else:
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

    print("Starting virtual odometer with wheel size {} PPR {} Speed {}".format(wheelSize, pulsesPerRotation, arguments.speed))
    odometer = VirtualOdometer(WHEEL_SIZE = wheelSize, PULSES = pulsesPerRotation, SPEED = arguments.speed)


    # Start a thread to service the readings queue
    service = threading.Thread(target = serviceQueue, args=(odometer,))
    service.start()

    # Start a thread to handle user IO. This is not required in normal operation
    io = threading.Thread(target=userIO)
    io.start()

    # Connect the odometer and start.
    odometer.connect()
    # The start routines never return - this is executed in the main thread
    odometer.start()

    sys.exit(0)

