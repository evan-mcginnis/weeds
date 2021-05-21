#
# O D O M E T E R
#

from abc import ABC, abstractmethod

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
    def __init__(self, options: str):
        self.options = options

    def connect(self) -> bool:
        return True

    def disconnect(self):
        return True

    def diagnostics(self):
        self._connected = False
        return True, "Odometer diagnostics passed"

    def registerCallback(self,callback):
        self._callback = callback
