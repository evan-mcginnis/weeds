#
# C O N T E X T
#
# The context in which an image was taken
#
from exif import Image
import logging
import logging.config
import math
import re

class Context:
    def __init__(self):
        self._speed = 0.0
        self._make = "Basler"
        self._model = "2500"
        self._latitude = 0.0
        self._longitude = 0.0
        self._exposure = 0
        self._timestamp = ""
        self._altitude = 0.0
        self._log = logging.getLogger(__name__)
        self._datestamp = ""

    @property
    def datestamp(self) -> str:
        return self._datestamp

    @datestamp.setter
    def datestamp(self, theDate: str):
        self._datestamp = theDate

    @property
    def timestamp(self) -> str:
        """
        Timestamp of the image
        :return:
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, theTimestamp: str):
        """
        Set the timestamp of the image
        :param theTimestamp: string value
        """
        self._timestamp = theTimestamp

    @property
    def altitude(self) -> float:
        """
        Altitude in meters above sea level
        :return:
        """
        return  self._altitude

    @altitude.setter
    def altitude(self, theAltitude: float):
        self._altitude = theAltitude

    @property
    def exposure(self) -> int:
        return self._exposure

    @exposure.setter
    def exposure(self, theExposure: int):
        self.exposure = theExposure

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
    def make(self) -> str:
        return self._make

    @make.setter
    def make(self, theMake: str):
        self._make = theMake

    @property
    def model(self) -> str:
        return self._make

    @model.setter
    def model(self, theModel: str):
        self._make = theModel

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
                        latitude = re.findall(coordinates, str(my_image.gps_latitude))
                        self._latitude = self.dms2dd(latitude[0], latitude[1], latitude[2], my_image.gps_latitude_ref)
                        longitude = re.findall(coordinates, str(my_image.gps_longitude))
                        self._longitude = self.dms2dd(longitude[0], longitude[1], longitude[2], my_image.gps_longitude_ref)
                        self._altitude = my_image.gps_altitude
                        self._timestamp = my_image.datetime_original
                        pass
                    except AttributeError:
                        self._log.error("Unable to find expected EXIF: latitude, longitude, and altitude")
                else:
                    print("Image contains no EXIF data")
        except FileNotFoundError:
            self._log.error(f"Unable to access: {filename}")

if __name__ == "__main__":
    import argparse
    import sys
    import os

    import constants
    from OptionsFile import OptionsFile

    parser = argparse.ArgumentParser("EXIF/Context Utility")

    parser.add_argument("-i", "--input", action="store", required=False, help="Name of image")
    parser.add_argument("-lg", "--logging", action="store", default="info-logging.ini", help="Logging configuration file")
    parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME,
                        help="Options INI")

    arguments = parser.parse_args()

    # Confirm the logging config file exists
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    logging.config.fileConfig(arguments.logging)

    # Load up the options file.
    options = OptionsFile(arguments.ini)
    if not options.load():
        print("Failed to load options from {}.".format(arguments.ini))
        sys.exit(1)
    else:
        meta = Context()
        meta.getMetadata(arguments.input)
        print(f"Latitude: {meta.latitude}")
        print(f"Longitude: {meta.longitude}")
        print(f"Altitude: {meta.altitude}")
        print(f"Taken: {meta.timestamp}")

