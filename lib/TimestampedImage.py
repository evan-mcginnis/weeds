#
# T I M E S T A M P E D I M A G E
#

import numpy as np

class TimestampedImage:
    def __init__(self, rgb: np.ndarray, depth: np.array, ir: np.array, timestamp: float):
        self._rgb = rgb
        self._depth = depth
        self._ir = ir
        self._timestamp = timestamp
        self._indexed = False

    @property
    def ir(self) ->np.ndarray:
        return self._ir
    @property
    def rgb(self) -> np.ndarray:
        return self._rgb

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    @property
    def timestamp(self) -> float:
        return self._timestamp
