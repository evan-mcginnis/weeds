#
# P R O C E S S E D  I M A G E
#

import numpy as np
import queue

import constants
from constants import Capture
from exif import Image
import logging
import logging.config
import math
import re



class ProcessedImage:
    def __init__(self, captureType: Capture, image: np.ndarray, timestamp: int):
        self._image = image
        self._timestamp = timestamp
        self._takenAt = ""
        self._indexed = False
        self._exif = None
        self._filename = None
        self._urlFilename = ""
        self._exposure = 0
        self._latitude = 0.0
        self._longitude = 0.0
        self._altitude = 0.0
        self._speed = 0
        self._make = "Basler"
        self._model = "2500-14gc"
        self._software = "UofA Weed Imaging"
        self._copyright = "Evan McGinnis"
        self._captureType = captureType
        self._imageNumber = 0
        self._type = constants.ImageType.RGB
        self._source = ""
        self._log = logging.getLogger(__name__)

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, theSource: str):
        self._source = theSource

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, theType: constants.ImageType):
        self._type = theType

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
    def altitude(self) -> float:
        return self._altitude

    @altitude.setter
    def altitude(self, theAltitude: float):
        self._altitude = theAltitude

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

    @image.setter
    def image(self, theImage):
        self._image = theImage

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def takenAt(self) -> str:
        return self._takenAt

    @takenAt.setter
    def takenAt(self, theTime: str):
        self._takenAt = theTime

    @classmethod
    def decdeg2dms(self, degs: float) -> ():
        neg = degs < 0
        degs = (-1) ** neg * degs
        degs, d_int = math.modf(degs)
        mins, m_int = math.modf(60 * degs)
        secs = 60 * mins
        return neg, d_int, m_int, secs

    @classmethod
    def dms2dd(self, degrees: float, minutes: float, seconds: float, direction: str) -> float:
        dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
        if direction == 'E' or direction == 'S':
            dd *= -1
        return dd

    def getMetadata(self, filename: str) -> {}:
        coordinates = "[0-9]+.[0-9]+"
        try:
            with open(filename, 'rb') as image_file:
                my_image = Image(image_file)
                if my_image.has_exif:
                    try:
                        if "gps_latitude" in my_image.list_all():
                            latitude = re.findall(coordinates, str(my_image.gps_latitude))
                            self._latitude = self.dms2dd(latitude[0], latitude[1], latitude[2], my_image.gps_latitude_ref)
                        else:
                            self._log.error(f"Unable to find latitude EXIF attribute")
                        if "gps_longitude" in my_image.list_all():
                            longitude = re.findall(coordinates, str(my_image.gps_longitude))
                            self._longitude = self.dms2dd(longitude[0], longitude[1], longitude[2], my_image.gps_longitude_ref)
                        else:
                            self._log.error(f"Unable to find longitude EXIF attribute")
                        if "gps_altitude" in my_image.list_all():
                            self._altitude = my_image.gps_altitude
                        else:
                            self._log.error(f"Unable to find altitude EXIF attribute")
                        if "datetime_original" in my_image.list_all():
                            self._takenAt = my_image.datetime_original
                        else:
                            self._log.error(f"Unable to find time EXIF attribute")
                    except AttributeError:
                        self._log.error("Unable to find expected EXIF: latitude, longitude, and altitude")
                else:
                    self._log.error("Image contains no EXIF data")
        except FileNotFoundError:
            self._log.fatal(f"Unable to access: {filename}")


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
