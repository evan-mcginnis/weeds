#
# O D O M E T E R
#

from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime
import queue

class Odometer(ABC):
    def __init__(self, options: str):
        self.options = options
        # This is where the readings go -- this size is way too big
        self._changeQueue = queue.Queue(maxsize=5000)

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


class VirtualOdometer(Odometer):
    def __init__(self, speed: int, processor: Callable):
        """
        A simulated odometer
        :param speed: Speed of movement in meters per second
        :param captureSize: Horizontal size of the image in millimeters
        :param processor: The image processing routine to callback at each processing step
        """
        self._speedInMMPerSecond = speed * 1000
        self._processor = processor
        self._log = logging.getLogger(__name__)
        # For example, it takes 0.00224 seconds to travel 1cm at 44.704 cm per second/1mph
        self._timeToTravel1CM = 1 / speed

        self._start = 0
        self._elapsed_milliseconds = 0
        self._elapsed = 0


    def connect(self) -> bool:
        return True

    def disconnect(self):
        return True

    def diagnostics(self):
        self._connected = False
        return True, "Odometer diagnostics passed"

    def registerCallback(self,callback):
        self._callback = callback

    def start(self):

        self._start = datetime.now()

        # Call the processor based on distance
        while self._processor(1):
            self._elapsed = datetime.now() - self._start
            self._elapsed_milliseconds = self._elapsed.total_seconds() * 1000


            # Call the processing routine every 1cm of travel
            self._log.debug("Sleep for {:.2f} seconds".format(self._timeToTravel1CM))
            sleep(self._timeToTravel1CM)
            self._start = datetime.now()
