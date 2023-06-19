
#
# L O G G E R
#
import os
import cv2 as cv
import numpy as np


class ImageLogger:
    def __init__(self):
        self.rootDirectory = ""
        self._sequence = 0
        self._autoIncrementMode = False

    @property
    def sequence(self):
        return self._sequence

    @sequence.setter
    def sequence(self, sequenceNo: int):
        self._sequence = sequenceNo

    @property
    def directory(self) -> str:
        return self.rootDirectory

    @property
    def autoIncrement(self):
        return self._autoIncrementMode

    @autoIncrement.setter
    def autoIncrement(self, theMode: bool):
        self._autoIncrementMode = theMode

    def increment(self):
        self._sequence += 1

    def connect(self, directoryName: str):
        try:
            os.makedirs(directoryName, exist_ok=True)
        except Exception as e:
            print("Cannot create directory: {}".format(directoryName))

        if os.path.isdir(directoryName):
            self.rootDirectory = directoryName
            return True
        else:
            return False

    def logImage(self, name: str, image: np.ndarray) -> str:
        pathname = "{}/{}-{:05d}.jpg".format(self.rootDirectory, name, self.sequence)
        filename = "{}-{:05d}.jpg".format(name, self.sequence)

        cv.imwrite(pathname, image)
        if self._autoIncrementMode:
            self.sequence += 1
        return filename
