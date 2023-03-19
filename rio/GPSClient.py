#
# G P S
#

import time

#from gpsd import *
import gpsd
import logging

class GPS:
    def __init__(self):
        self._log = logging.getLogger(__name__)
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

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
        except UserWarning as warning:
            self._log.error("Unable to connect to GPS")
            self._log.error("Raw: {}".format(warning))
            self._connected = False

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

    # This is another

    theGPS = GPS()
    if theGPS.connect():
        print("GPS Connection succeeded")
    else:
        print("GPS Connection failure")

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