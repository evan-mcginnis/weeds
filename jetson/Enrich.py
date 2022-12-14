#
# E N R I C H
#
# Enrich the image or image set by inserting GPS info and other data into the exif.
#

import exif
from exif import Image
import logging
import logging.config
import math
from pathlib import Path

from ProcessedImage import ProcessedImage

class Enrich:
    def __init__(self, **kwargs):
        """
        Enrich an image or a set of images
        :param kwargs: LAT, LONG, EXPOSURE, PULSES, SPEED
        """

        self._log = logging.getLogger(__name__)

    # TODO: Move this to a GPS utility
    def decdeg2dms(self, degs: float) -> ():
        neg = degs < 0
        degs = (-1) ** neg * degs
        degs, d_int = math.modf(degs)
        mins, m_int = math.modf(60 * degs)
        secs = 60 * mins
        return neg, d_int, m_int, secs

    def addExifToImagesInDirectory(self, imageSet: str, **kwargs):
        """
        Add EXIF information to all JPG images in directory.
        :param imageSet: Fully qualified target directory
        :param kwargs: keywords to add to EXIF
        """
        self._log.debug("Add EXIF keywords to images in directory: {}".format(imageSet))

        for key, value in kwargs.items():
            self._log.debug("Keyword: {}/{}".format(key, value))

    def addEXIFToImageAndWriteToDisk(self, image: ProcessedImage):
        """
        Adds the EXIF data to the image and writes out the result.
        :param image: the target image, already on disk without EXIF
        """
        filename = image.filename
        self._log.debug("Enriching {} with EXIF data".format(filename))
        with open(filename, 'rb') as image_file:
            self._log.debug("Reading image")
            image_bytes = image_file.read()
            img = Image(image_bytes)
            self._log.debug("Assigning EXIF data to image")
            img.software = image.software
            img.copyright = image.copyright
            img.make = image.make
            img.model = image.model
            img.gps_speed = image.speed
            img.gps_speed_ref = "K"
            img.exposure_time = image.exposure / 1e+6
            (negative, degrees, minutes, seconds) = self.decdeg2dms(image.latitude)
            img.gps_latitude = (degrees, minutes, seconds)
            img.gps_latitude_ref = "N"
            (negative, degrees, minutes, seconds) = self.decdeg2dms(image.longitude)
            img.gps_longitude = (degrees, minutes, seconds)
            img.gps_longitude_ref = "W"

            newFilename = filename.replace(constants.FILENAME_RAW, constants.FILENAME_FINISHED)

            self._log.debug("Write out enriched file {}".format(newFilename))
            with open(newFilename, 'wb') as new_image_file:
                new_image_file.write(img.get_file())

    def addEXIFToFile(self, image: str, **kwargs):
        """
        Add EXIF information to a single image.
        :param image: Fully qualified image name
        :param kwargs: keywords to add to exif
        """
        with open(image, 'rb') as image_file:
            img = Image(image_file)
            for key, value in kwargs.items():
                self._log.debug("Add {}/{}".format(key, value))
                if key == constants.KEYWORD_SOFTWARE:
                    img.software = value
                elif key == constants.KEYWORD_COPYRIGHT:
                    img.copyright = value
                elif key == constants.KEYWORD_MAKE:
                    img.make = value
                elif key == constants.KEYWORD_MODEL:
                    img.model = value
                elif key == constants.KEYWORD_SPEED:
                    img.gps_speed = value
                    img.gps_speed_ref = "K"
                elif key == constants.KEYWORD_EXPOSURE:
                    img.exposure_time = value / 1e+6
                elif key == constants.KEYWORD_LAT:
                    (negative, degrees, minutes, seconds) = self.decdeg2dms(value)
                    img.gps_latitude = (degrees, minutes, seconds)
                    img.gps_latitude_ref = "N"
                elif key == constants.KEYWORD_LONG:
                    (negative, degrees, minutes, seconds) = self.decdeg2dms(value)
                    img.gps_longitude = (degrees, minutes, seconds)
                    img.gps_longitude_ref = "W"

        modifiedImage = "modified.jpg"
        with open(modifiedImage, 'wb') as new_image_file:
            new_image_file.write(img.get_file())

    def printEXIF(self, image: str):
        print("Show current EXIF for {}".format(image))
        with open(image, 'rb') as image_file:
            my_image = Image(image_file)
            if my_image.has_exif:
                print("{}".format(my_image.list_all()))
                print(my_image.gps_latitude)
                print(my_image.gps_speed)
            else:
                print("Image contains no EXIF data")

if __name__ == "__main__":
    import argparse
    import sys
    import os

    import constants
    from OptionsFile import OptionsFile

    parser = argparse.ArgumentParser("File Enrichment Utility")

    parser.add_argument("-i", "--input", action="store", required=False, help="Name of image")
    parser.add_argument("-lg", "--logging", action="store", default="info-logging.ini", help="Logging configuration file")
    parser.add_argument('-ini', '--ini', action="store", required=False, default=constants.PROPERTY_FILENAME,
                        help="Options INI")
    parser.add_argument("-s", "--show", action="store_true", required=False, default=False, help="Show current tags")

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
        enricher = Enrich()
        if arguments.show:
            enricher.printEXIF(arguments.input)
        else:
            processed = ProcessedImage(None, 0)
            processed.filename = arguments.input
            processed.model = "2500"
            processed.make = "basler"
            processed.copyright = "Evan McGinnis"
            processed.software = "UofA Weeds"
            processed.latitude = 32.228995
            processed.longitude = -110.9398582
            enricher.addEXIFToImageAndWriteToDisk(processed)
            # enricher.addEXIFToFile(arguments.input,
            #                        SPEED=3.0,
            #                        SOFTWARE="UofA Weeds",
            #                        COPYRIGHT="Evan McGinnis",
            #                        EXPOSURE=385,
            #                        MAKE="basler",
            #                        MODEL="2500",
            #                        LATITUDE=32.228995,
            #                        LONGITUDE=-110.9398582)

