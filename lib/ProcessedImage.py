#
# P R O C E S S E D  I M A G E
#

import numpy as np
import queue
from constants import Capture



class ProcessedImage:
    def __init__(self, captureType: Capture, image, timestamp: int):
        self._image = image
        self._timestamp = timestamp
        self._indexed = False
        self._exif = None
        self._filename = None
        self._urlFilename = ""
        self._exposure = 0
        self._latitude = 0.0
        self._longitude = 0.0
        self._speed = 0
        self._make = "Basler"
        self._model = "2500-14gc"
        self._software = "UofA Weed Imaging"
        self._copyright = "Evan McGinnis"
        self._captureType = captureType
        self._imageNumber = 0

    @property
    def urlFilename(self) -> str:
        return self._urlFilename

    @urlFilename.setter
    def urlFilename(self, filename):
        self._urlFilename = filename

    @property
    def imageNumber(self) -> int:
        return self._imageNumber

    @property
    def captureType(self) -> Capture:
        return self._captureType

    @property
    def copyright(self) -> str:
        return self._copyright

    @copyright.setter
    def copyright(self, theCopyright: str):
        self._copyright = theCopyright

    @property
    def exposure(self) -> int:
        return self._exposure

    @exposure.setter
    def exposure(self, theExposure: int):
        self._exposure = theExposure

    @property
    def latitude(self) -> float:
        return self._latitude

    @latitude.setter
    def latitude(self, theLatitude: float):
        self._latitude = theLatitude

    @property
    def longitude(self) -> float:
        return self._longitude

    @longitude.setter
    def longitude(self, theLongitude: float):
        self._longitude = theLongitude

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, theSpeed: float):
        self._speed = theSpeed

    @property
    def software(self) -> str:
        return self._software

    @software.setter
    def software(self, theSoftware):
        self._software = theSoftware

    @property
    def make(self) -> str:
        return self._make

    @make.setter
    def make(self, theMake: str):
        self._make = theMake

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, theModel: str):
        self._model = theModel

    def addEXIF(self, **kwargs):
        self._exif = kwargs

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, name: str):
        self._filename = name

    @property
    def exif(self):
        return self._exif

    @exif.setter
    def exif(self, exifData: str):
        self.exif = exifData

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def timestamp(self) -> int:
        return self._timestamp

class Images:
    def __init__(self):
        self._elements = queue.Queue()

    @property
    def queue(self):
        return self._elements

    def enqueue(self, element: ProcessedImage):
        """
        Add the processed image to the queue
        :param element:
        """
        self._elements.put(element)

    def dequeue(self) -> ProcessedImage:
        """
        Get the next image from the queue. This operation will block if the queue is empty.
        :return:
        """
        image = self._elements.get(block=True)
        return image
