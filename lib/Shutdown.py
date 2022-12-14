#
# S I G N A L S
#

import signal

class Shutdown:

    def __init__(self):
        self.exitNow = False
        signal.signal(signal.SIGINT, self._exitGracefully)
        signal.signal(signal.SIGTERM, self._exitGracefully)

    def _exitGracefully(self, *args):
        """
        Action that is taken on the SIGINT or SIGTERM signal.
        :param args:
        """
        self.exitNow = True
