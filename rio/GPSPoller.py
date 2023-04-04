from gps import *
import threading
import logging

import constants


#
# G P S P O L L E R
#
# To use this, create a poller and start it as a separate thread
#

class GPSPoller(threading.Thread):

    def __init__(self, refreshInterval: float) -> object:
        """
        GPSPoller -- create an instance and execute start.
        """
        threading.Thread.__init__(self)
        self.setName(constants.THREAD_NAME_GPS)
        self.setDaemon(True)
        self.session = gps(mode=WATCH_ENABLE)
        self._current_value = None
        self._tpv = None
        self._refreshInterval = refreshInterval
        self._log = logging.getLogger(__name__)

    def current(self):
        """
        The current value
        :return:
        """
        return self._current_value

    def latLong(self) -> (float, float):
        """
        The current latitude and longitude
        :return: (latitude. longitude)
        """
        if self._tpv is not None:
            position = (self._tpv['lat'], self._tpv['lon'])
        else:
            position = (0, 0)
        return position

    def run(self):
        """
        The run routine for the thread. This will not return.
        """
        try:
            while True:
                try:
                    self._current_value = self.session.next()
                    if self._current_value['class'] == 'TPV':
                        self._tpv = self._current_value
                    time.sleep(self._refreshInterval)
                except ConnectionResetError:
                    self._log.error("GPS reset connection -- will retry connection")
                    time.sleep(5)
                    self.session = gps(mode=WATCH_ENABLE)
        except StopIteration:
            pass


if __name__ == '__main__':
    # Example current_value
    # < dictwrapper: {'class': 'TPV', 'device': '/dev/ttyACM0', 'mode': 3, 'time': '2023-03-18T22:40:30.000Z',
    #                 'ept': 0.005, 'lat': 32.228964833, 'lon': -110.939808, 'alt': 756.7, 'epx': 9.547, 'epy': 11.263,
    #                 'epv': 35.42, 'track': 3.59, 'speed': 0.732, 'climb': 0.0, 'eps': 22.53} >

    gpsp = GPSPoller(0.5)
    gpsp.start()
    # gpsp now polls every .2 seconds for new data, storing it in self.current_value
    while 1:
        # In the main thread, every 5 seconds print the current value
        time.sleep(5)
        start = time.time()
        print('---')
        current = gpsp.current()
        print(current)
        print('---')
        finish = time.time()
        print("Elapsed time {} seconds".format(finish - start))
        print("Lat/Long {}".format(gpsp.latLong()))
