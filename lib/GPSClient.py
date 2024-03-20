#
# G P S
#
import sys
import time

#from gpsd import *
import gpsd
import pyubx2
import logging
from abc import ABC, abstractmethod

import constants


class GPS(ABC):
    def __init___(self, **kwargs):
        self._connected = False
        self._latitude = 0.0
        self._longitude = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        return False

    @abstractmethod
    def getCurrentPosition(self) -> (float, float):
        return self._latitude, self._longitude

    @abstractmethod
    def isAvailable(self) -> bool:
        return False

class DirectlyConnected(GPS):
    def __init__(self, **kwargs):
        self._log = logging.getLogger(__name__)
        self._com = None
        super().__init__(**kwargs)

        try:
            self._com = kwargs[constants.KEYWORD_COM]
        except KeyError:
            self._log.error(f"COM port must be specified with {constants.KEYWORD_COM}")


    def isAvailable(self) -> bool:
        return False

    def connect(self) -> bool:
        return self._connected

    def getCurrentPosition(self) -> (float, float):
        return 0, 0

class GPSDConnected(GPS):
    def __init__(self, **kwargs):
        self._log = logging.getLogger(__name__)

        super().__init__(**kwargs)

    def isAvailable(self) -> bool:
        """
        Is the GPS data available?
        Note: this does not yet consider the number of satellites fixed.
        :return: True if GPS is available, False otherwise
        """
        return self.getCurrentPosition() is not None


    def connect(self) -> bool:
        """
        Connect to the GPS Daemon.  Required before any calls to get details such as position.
        :return: True on connection success, False otherwise
        """
        try:
            gpsd.connect()
            self._connected = True
        except ConnectionRefusedError as refused:
            self._log.error("Unable to connect to GPSD")
            self._connected = False
            return self._connected
        except UserWarning as warning:
            self._log.error("Unable to connect to GPS")
            self._log.error("Raw: {}".format(warning))
            self._connected = False
            return self._connected

        # See if we can get the current position
        packet = None
        try:
            packet = gpsd.get_current()
        except Exception as ex:
            self._log.error("Raw {}".format(ex))

        if packet is None:
            self._connected = False
        else:
            self._connected = True

        return self._connected

    def getCurrentPosition(self) -> (float, float):
        """
        Get the current position.  Returns (0,0) if not connected.
        :return: GpsResponse
        """
        position = None
        if not self._connected:
            self._log.error("Not connected to GPS")
        else:
            try:
                position = gpsd.get_current()
            except UserWarning as user:
                self._log.error("GPS Not active")
            except gpsd.NoFixError as fix:
                self._log.error("Needs a GPS 2D fix")
        return position

# Example for another library
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("GPS Client")

    parser.add_argument("-c", '--com', action="store", required=False, help="COM port")

    parser.parse_args()

    theGPS = GPSDConnected()
    if theGPS.connect():
        print("GPS Connection succeeded")
    else:
        print("GPS Connection failure")
        sys.exit(1)

    start = time.time() * 1000
    packet = theGPS.getCurrentPosition()
    finish = time.time() * 1000
    if packet is not None:
        try:
            print("Position: {}".format(packet.position()))
            print("Error: {}".format(packet.position_precision()))
            print("Fix: {}".format(packet.mode))
            print("Elapsed time {} seconds".format(finish - start))
        except gpsd.NoFixError as fix:
            print("Unable to obtain a 2D fix")
    else:
        print("No position data")