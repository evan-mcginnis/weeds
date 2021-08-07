#
# O D O M E T E R
#

from abc import ABC, abstractmethod
from typing import Callable
import logging
from time import sleep
from datetime import datetime

class Odometer(ABC):
    def __init__(self, options: str):
        self.options = options

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


class VirtualOdometer(Odometer):
    def __init__(self, speed: int, captureSize: int, processor: Callable):
        """
        A simulated odometer
        :param speed: Speed of movement in meters per second
        :param captureSize: Horizontal size of the image in millimeters
        :param processor: The image processing routine to callback at each processing step
        """
        self._speedInMMPerSecond = speed * 1000
        self._processor = processor
        self._captureSize = captureSize
        self._log = logging.getLogger(__name__)

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

        # How long in milliseconds we have to process the image given the current speed
        timeBudget = (1 / (self._speedInMMPerSecond / self._captureSize))  * 1000

        self._start = datetime.now()
        while self._processor():
            self._elapsed = datetime.now() - self._start
            self._elapsed_milliseconds = self._elapsed.total_seconds() * 1000
            sleepTime = 0
            if self._elapsed_milliseconds > timeBudget:
                self._log.error("Processing exceeded budget. {:.4f} vs {:.4f}".format(timeBudget, self._elapsed_milliseconds))
            else:
                self._log.info("Processed image")
                sleepTime = timeBudget - self._elapsed_milliseconds

            self._log.debug("Sleep for {:.2f} seconds".format(sleepTime))
            sleep(sleepTime)
            self._start = datetime.now()


