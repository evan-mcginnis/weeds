#
#
#
import numpy as np
from abc import ABC, abstractmethod
from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed

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
    def __init__(self, **kwargs):
        super().__init__()
        self._state = CameraState()


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
    def capture(self) -> np.ndarray:
        self._connected = False
        return

    @abstractmethod
    def getResolution(self):
        self._connected = False
        return (0,0)


