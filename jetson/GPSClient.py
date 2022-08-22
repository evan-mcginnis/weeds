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

    def connect(self) -> bool:
        """
        Connect to the GPS.  Required before any calls to get details such as position.
        :return: True on connection success, False otherwise
        """
        try:
            gpsd.connect()
            self._connected = True
        except ConnectionRefusedError as refused:
            self._log.error("Unable to connect to GPSD")
            self._connected = False
        return self._connected

    def getCurrentPosition(self):
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
        return position

# Example for another library
#from gps import *
# class GpsPoller(threading.Thread):
#
#    def __init__(self):
#        threading.Thread.__init__(self)
#        self.session = gps(mode=WATCH_ENABLE)
#        self.current_value = None
#
#    def get_current_value(self):
#        return self.current_value
#
#    def run(self):
#        try:
#             while True:
#                 self.current_value = self.session.next()
#                 time.sleep(0.2) # tune this, you might not get values that quickly
#        except StopIteration:
#             pass

if __name__ == '__main__':

    # # This is one way of doing things
    # gpsp = GpsPoller()
    # gpsp.start()
    # # gpsp now polls every .2 seconds for new data, storing it in self.current_value
    # while 1:
    #     # In the main thread, every 5 seconds print the current value
    #     time.sleep(5)
    #     print(gpsp.get_current_value())

    # This is another

    theGPS = GPS()
    if theGPS.connect():
        start = time.time() * 1000
        packet = theGPS.getCurrentPosition()
        finish = time.time() * 1000
        if packet is not None:
            print(packet.position())
            print("Elapsed time {} ms".format(finish - start))
    else:
        print("Error in connecting to GPS daemon")