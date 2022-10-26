#
# X M P P S E R V E R
#

import xmpp.protocol
from xmpp import *

class XMPPServer():
    def __init__(self, ipAddress: str):
        self._serverReachable = False
        self._ipAddress = ipAddress

    @property
    def ipAddress(self) -> str:
        return self._ipAddress

    @property
    def serverReachable(self) -> bool:
        return self._serverReachable

    def connect(self, username: str, password: str) -> bool:
        return self._serverReachable

    def disconnect(self):
        pass