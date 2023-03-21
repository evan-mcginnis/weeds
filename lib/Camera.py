#
#
#
import numpy as np
from typing import Callable
from abc import ABC, abstractmethod
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed
from ProcessedImage import ProcessedImage

import constants

class CameraState(StateMachine):
    new = State('New', initial=True)
    capturing = State('Capturing')
    claimed = State('Claimed')
    idle = State('Idle')
    failed = State('Failed')
    missing = State('Missing')

    toIdle = new.to(idle)
    toCapture = claimed.to(capturing)
    toStop = capturing.to(idle)
    toClaim = idle.to(claimed)
    toFailed = claimed.to(failed)
    toMissing = new.to(missing)


class Camera(ABC):
    cameras = list()
    cameraCount = 0
    def __init__(self, **kwargs):
        super().__init__()
        self._state = CameraState()

        # Register the camera on the global list so we can keep track of them
        # Even though there will probably be only one
        self.cameraID = Camera.cameraCount
        Camera.cameraCount += 1
        Camera.cameras.append(self)

        self._status = constants.OperationalStatus.UNKNOWN

        self._gsd = 0
        self._gsdAdjusted = 0

        # Where the camera is positioned
        self._position = constants.PositionWithEmitter.PRE

        # The current speed of travel
        self._speed = 0.0

        self._methodToGetSpeed = None

    @property
    def status(self) -> constants.OperationalStatus:
        return self._status

    @property
    def state(self) -> CameraState:
        return self._state
        return

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
    def capture(self) -> ProcessedImage:
        self._connected = False
        return

    @abstractmethod
    def getResolution(self):
        self._connected = False
        return (0,0)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, thePosition: constants.PositionWithEmitter):
        self._position = thePosition

    @property
    def gsdAdjusted(self):
        return self._gsdAdjusted

    @gsdAdjusted.setter
    def gsdAdjusted(self, _gsdAdjusted):
        self._gsdAdjusted = _gsdAdjusted

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, theSpeed: float):
        self._speed = theSpeed

    def methodToGetSpeed(self, speedMethod: Callable):
        self._methodToGetSpeed = speedMethod




