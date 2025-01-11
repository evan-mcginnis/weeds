#
#
#

import argparse
import sys
import os
import logging
import logging.config
import shutil
import pandas as pd
import cv2 as cv
from pathlib import Path
import math

from Metadata import Metadata
import constants
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger
from Classifier import Classifier

def closestAGL(possibleAGLs: [], recordedAGL: float) -> float:
    """
    Determine the AGL closest to one in the list given
    :param possibleAGLs: List of possible AGLs
    :param recordedAGL: AGL recorded
    :return: one of the AGLs in the possibleAGLs list
    """
    closest = possibleAGLs[0]
    for i in possibleAGLs:
        if abs(i - recordedAGL) < closest:
            closest = i
    # print(f"{recordedAGL} Closest AGL {closest}")
    return closest

parser = argparse.ArgumentParser("Analyse Altitude Metadata from images")

parser.add_argument("-i", "--input", action="store", nargs="+", required=True, help="Source directory for images")
parser.add_argument("-o", "--output", action="store", required=False, default=".", help="Output directory")
parser.add_argument("-a", "--agl", action="store", type=float, nargs="+", required=False, help="List of AGL captured")
parser.add_argument("-lg", "--logging", action="store", default="logging.ini", help="Logging configuration file")
parser.add_argument("-f", "--force", action="store_true", required=False, default=False, help="Force overwrite of existing files")
parser.add_argument("-m", "--msl", type=float, action="store", required=False, help="The altitude above MSL")
parser.add_argument("-b", "--blob", action="store", required=False, type=float, help="Blob size in mm")
parser.add_argument("-c", "--calculate", action="store_true", required=False, default=False, help="Calculate from the image")
parser.add_argument("-v", "--verbose", action="store_true", required=False, default=False, help="Enable verbose output")
arguments = parser.parse_args()

COLUMN_PATH = "path"
COLUMN_ALTITUDE = "altitude"
COLUMN_AGL = "agl"

if arguments.calculate:
    if arguments.blob is None:
        print(f"Blob size (--blob) must be specified if calculate is (--calculate)")
        exit(-1)

# Confirm the INI exists
if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(1)

# Initialize logging
logging.config.fileConfig(arguments.logging)

log = logging.getLogger('altitude')

logging.getLogger('exif._image').setLevel(logging.CRITICAL)

allDeviations = []

for imageDirectory in arguments.input:
    if arguments.verbose:
        print(f"Image set: {imageDirectory}")
    # Assume a flat source directory, and find all the files
    allFiles = []
    allAltitudes = []
    correctedAGL = []
    # Ignore DNG files, as I haven't sorted through support for them yet for the VIs.
    # The metadata bit is handled incmpletely, but does work
    included_extensions = ['jpg', 'JPG']
    if not os.path.isdir(imageDirectory):
        print(f"Unable to access directory: {imageDirectory}")
        sys.exit(-1)

    file_names = [fn for fn in os.listdir(imageDirectory)
                  if any(fn.endswith(ext) for ext in included_extensions)]

    # Create the lista and dataframe
    for aFile in file_names:
        # check if current file_path is a file
        filePath = os.path.join(imageDirectory, aFile)
        if os.path.isfile(filePath):
            # add filename to list
            allFiles.append(os.path.join(imageDirectory, aFile))

            # Determine things by the bucket lid in the image
            if arguments.calculate:
                # Mask the image
                vi = VegetationIndex()
                vi.Load(filePath)
                rawMask = vi.bucketLid()
                #"Bucket Lid": {"short": "BLI", "create": utility.bucketLid, "negate": False, "threshold": threholds["TGI"], "direction": 1, "normalize": False},
                threshold = 0.0
                negate = False
                direction = 1
                mask, thresholdUsed = vi.createMask(rawMask, negate, direction, threshold)
                vi.applyMask()
                image = vi.GetImage()
                # Debug
                cv.imwrite(f"{arguments.output}/{Path(aFile).stem}-bi.jpg", image)

                logger = ImageLogger()
                if not logger.connect(arguments.output):
                    log.error("Unable to connect to logging. {} does not exist.".format(arguments.output))
                manipulatedImage = ImageManipulation(image, 0, logger)

                # Find the blobs in the image -- there should be only the lid
                # For now, this only works with a single lid
                # The threshold for area is completely arbitrary -- as we really only care about the bucket lid, this may
                # need to be changed to reflect what is seen at higher distances AGL
                contours, hierarchy, blobs, largest = manipulatedImage.findBlobs(10000, constants.Strategy.CARTOON)


                heuristicClassifier = Classifier()
                heuristicClassifier.blobs = blobs
                heuristicClassifier.classifyAs(constants.TYPE_WEED)
                #manipulatedImage.drawContours()
                manipulatedImage.drawBoxes("Bucket Lid", heuristicClassifier.blobs, [constants.NAME_BOX_DIMENSIONS])
                imageWithContours = manipulatedImage.image
                cv.imwrite(f"{arguments.output}/{Path(aFile).stem}-contours.jpg", imageWithContours)

                # if len(blobs) > 1:
                #     log.error(f"Image contains {len(blobs)} blobs. Unable to determine which to use")
                #     exit(-1)
                if len(blobs) == 0:
                    log.error(f"Unable to find blob in image")
                    exit(-1)

                # Interested in the lens focal length, the make, and the pixel size
                meta = Metadata(filePath)
                meta.getMetadata()
                # Set these from the image attributes, as the iphone images don't have the required attribute
                meta.x = image.shape[0]
                meta.y = image.shape[1]

                for name, blob in blobs.items():
                    print(f"Examine: {name}")
                    #disk = blobs['blob-0']
                    # The size of the disk in the image in mm
                    diskPixelWidth = (blob['location'][3] + blob['location'][2]) / 2

                    sizeOfDiskOnSensor = diskPixelWidth * meta.pixelSize(meta.model, meta.x, meta.y) / 1000
                    # Opposite in the trigonometry sense
                    opposite = sizeOfDiskOnSensor / 2
                    # The tan of the angle will be equal to this
                    tanOfAngle = opposite / meta.focalLength

                    # Determine the angle furthest away from sensor
                    # Note that math.atan() returns radians
                    angle = math.degrees(math.atan(tanOfAngle))

                    # Determine the flight height
                    flightHeight = (arguments.blob / 2) / tanOfAngle

                    log.debug(f"Angle at focal length: {angle}")
                    log.debug(f"Flight height: {flightHeight} mm")

            # Use the image metadata to determine altitude
            else:
                meta = Metadata(filePath)
                meta.getMetadata()
                if arguments.verbose:
                    print(f"{aFile}: Altitude {meta.altitude}")
                allAltitudes.append(meta.altitude)
                # Determine AGL, as we don't really care about the altitude
                #correctedAGL.append(meta.altitude - arguments.msl)
                correctedAGL.append(meta.altitude)


    images = pd.DataFrame(list(zip(allFiles, allAltitudes, correctedAGL)),
                          columns=[COLUMN_PATH, COLUMN_ALTITUDE, COLUMN_AGL])

    # Determine the standard deviation of the observation
    deviations = images.std(numeric_only=True)
    allDeviations.append(deviations[0])

print(f"Standard deviations: {allDeviations}")
sys.exit(0)

for index, row in images.iterrows():
    images.at[index, COLUMN_AGL] = closestAGL(list(arguments.agl), row[COLUMN_AGL])
    # print(f"{row[COLUMN_PATH]}: Raw AGL: {row[COLUMN_AGL]} Closest AGL: {takeClosest(list(arguments.agl), row[COLUMN_AGL])}")

directory = os.path.dirname(__file__)

for index, row in images.iterrows():
    # Create the directory
    filename = os.path.join(directory, arguments.output)
    destinationDir = os.path.join(filename, "AGL-" + str(images.at[index, COLUMN_AGL]) + constants.DASH + arguments.crop)
    if not os.path.exists(destinationDir):
        try:
            os.makedirs(destinationDir, exist_ok=True)
        except OSError as e:
            print(f"Unable to create: {destinationDir}. {e.strerror}")
            sys.exit(-1)

    # Copy the file there, refusing to overwrite a file
    destinationFile = os.path.join(destinationDir, os.path.split(images.at[index, COLUMN_PATH])[1])
    if not arguments.force and os.path.isfile(destinationFile):
        print(f"File exists: {destinationFile}. Use -f to force overwrite")
        sys.exit(-1)
    else:
        try:
            print(f"{images.at[index, COLUMN_PATH]} -> {destinationDir}")
            shutil.copy2(images.at[index, COLUMN_PATH], destinationDir)
        except OSError as e:
            print(f"Unable to copy to {destinationFile}. {e.strerror}")
            sys.exit(-1)

sys.exit(0)
