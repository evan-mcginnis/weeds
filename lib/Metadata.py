#
# E X I F
#
# Enrich the image or image set by inserting GPS info and other data into the exif.
#

from exif import Image
import logging
import logging.config
import math
import os
import re
from exiftool import ExifToolHelper
from ProcessedImage import ProcessedImage
import constants


class Metadata:
    def __init__(self, image: str):
        """
        Extract metadata for all images in a directory
        :param image: path to the image
        """

        self._log = logging.getLogger(__name__)
        self._image = image
        self._meta = []
        self._lat = 0.0
        self._long = 0.0
        self._altitude = 0.0
        self._agl = 0.0
        self._taken = ""

    @property
    def gps(self) -> ():
        """
        GPS lat/long
        :return:
        """
        return self._lat, self._long

    @property
    def latitude(self) -> float:
        """
        Latitude
        :return: Float of decimal latitude
        """
        return self._lat

    @property
    def longitude(self) -> float:
        """
        Longitude
        :return: Float of decimal longitude
        """
        return self._long

    @property
    def altitude(self) -> float:
        """
        Altitude
        :return: Float of altitude
        """
        return self._altitude

    @property
    def taken(self) -> str:
        """
        Timestamp when image was taken
        :return: String of timestamp
        """
        return self._taken

    @property
    def agl(self) -> float:
        """
        Distance AGL -- DNG only
        :return: Float of AGL
        """
        return self._agl

    def getMetadata(self):
        """
        Get the EXIF for the image
        """
        coordinates = "[0-9]+.[0-9]+"
        # There are two ways to get the EXIF -- one for things like JPGs, and another for DNGs
        suffix = os.path.splitext(self._image)[1]
        if suffix is None:
            self._log.error(f"Unable to determine file type: {self._image}")
            return

        # Raw format image
        if suffix.upper() == ".DNG":
            with ExifToolHelper() as et:
                for d in et.get_metadata(self._image):
                    try:
                        altitude = d["Composite:GPSAltitude"]
                        latitude = d["Composite:GPSLatitude"]
                        longitude = d["Composite:GPSLongitude"]
                        taken = d["EXIF:DateTimeOriginal"]
                        # Not given in EXIF, but we can access it in DNG
                        agl = d["XMP:RelativeAltitude"]
                    except KeyError:
                        self._log.error("Unable to find expected EXIF in DNG: latitude, longitude, AGL, and altitude")
                        altitude = 0.0
                        latitude = 0.0
                        longitude = 0.0
                        agl = 0.0

                    self._altitude = float(altitude)
                    self._taken = taken
                    self._agl = float(agl)
                    # Confusingly enough, the EXIF in the DNS is in digital degrees, not dms, so no conversion
                    self._lat = float(latitude)
                    self._long = float(longitude)

                    # for k, v in d.items():
                    #     print(f"Dict: {k} = {v}")

        elif suffix.upper() == ".JPG":
            try:
                with open(self._image, 'rb') as image_file:
                    my_image = Image(image_file)
                    if my_image.has_exif:
                        try:
                            latitude = re.findall(coordinates, str(my_image.gps_latitude))
                            self._lat = self.dms2dd(latitude[0], latitude[1], latitude[2], my_image.gps_latitude_ref)
                            longitude = re.findall(coordinates, str(my_image.gps_longitude))
                            self._long = self.dms2dd(longitude[0], longitude[1], longitude[2], my_image.gps_longitude_ref)
                            self._altitude = my_image.gps_altitude
                            self._taken = my_image.datetime_original
                        except AttributeError:
                            self._log.error("Unable to find expected EXIF in JPG: latitude, longitude, and altitude")
                    else:
                        self._log.error("JPG Image contains no EXIF data")
            except FileNotFoundError:
                self._log.error(f"Unable to access: {self._image}")
        else:
            self._log.error(f"Unknown file type: {self._image}")


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

    def printEXIF(self, image: str):
        print("Show current EXIF for {}".format(image))
        with open(image, 'rb') as image_file:
            my_image = Image(image_file)
            if my_image.has_exif:
                exifTags = my_image.get_all()
                for tag in exifTags:
                    print(f"{tag}: {my_image.get(tag)}")
                # print(my_image.gps_latitude)
                # print(my_image.gps_speed)
            else:
                print("Image contains no EXIF data")

if __name__ == "__main__":
    import argparse
    import sys
    import os

    import constants
    from OptionsFile import OptionsFile

    parser = argparse.ArgumentParser("EXIF/Metadata Utility")

    parser.add_argument("-i", "--input", action="store", required=False, help="Name of image")
    parser.add_argument("-lg", "--logging", action="store", default="info-logging.ini", help="Logging configuration file")
    parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME,
                        help="Options INI")
    parser.add_argument("-all", "--all", action="store_true", required=False, default=False)

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
        meta = Metadata(arguments.input)
        if arguments.all:
            meta.printEXIF(arguments.input)
        else:
            meta.getMetadata()
            print(f"Latitude: {meta.latitude}")
            print(f"Longitude: {meta.longitude}")
            print(f"Altitude: {meta.altitude}")
            print(f"Taken: {meta.taken}")


