#
# C A M E R A
#
import numpy as np
from abc import ABC, abstractmethod

class Camera(ABC):
    def __init__(self, options : str):
        self.options = options
        super().__init__()

    @abstractmethod
    def connect(self):
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
    def capture(self) -> np.ndarray:
        self._connected = False
        return

    @abstractmethod
    def getResolution(self):
        self._connected = False
        return (0,0)


