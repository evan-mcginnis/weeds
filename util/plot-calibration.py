#
# P L O T - C A L I B R A T I O N
#
# Plot the average value for a color for a set of images.

import os
import sys
import glob
from datetime import datetime

import argparse

import constants
from ImageManipulation import ImageManipulation
from VegetationIndex import VegetationIndex
from Metadata import Metadata
from ImageLogger import ImageLogger

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Plot the average color for a set of images")

parser.add_argument("-c", "--corrected", action="store", required=True, help="Corrected image directory")
parser.add_argument("-u", "--uncorrected", action="store", required=True, help="Uncorrected image directory")
parser.add_argument("-d", "--debug", action="store_true", required=False, default=False, help="Debug")
parser.add_argument("-s", "--scratch", action="store", required=False, help="Scratch area for debug output")
parser.add_argument("-o", "--output", action="store", required=False, help="Output for plot")

arguments = parser.parse_args()

if not os.path.isdir(arguments.corrected):
    print(f"Unable to access directory: {arguments.corrected}")
    sys.exit(-1)

if arguments.scratch is not None and not os.path.isdir(arguments.scratch):
    print(f"Unable to access directory: {arguments.scratch}")
    sys.exit(-1)

logger = ImageLogger()
if arguments.scratch is not None:
    logger.connect(arguments.scratch)

def processDirectory(directory: str, scratch: str) -> pd.DataFrame:

    data = []
    images = glob.glob(directory + "/*.jpg")
    for image in images:
        print(f"Process: {image}")

        # Determine date of capture
        exif = Metadata(image)
        exif.getMetadata()
        exifDate = exif.taken

        # Convert the timestamp to just a ISO date
        dateTimeFormat = '%Y:%m:%d %H:%M:%S'
        dateTime = datetime.strptime(exifDate, dateTimeFormat)
        acquisitionDate = dateTime.strftime('%Y-%m-%d')
        print(f"Image was acquired on {acquisitionDate}")

        # Segment out the blue squares
        vi = VegetationIndex()
        vi.Load(image, equalize=False)
        # This maek captures the blue squares -- there are two on the calibration plate that will match
        rawMask = vi.SI()
        threshold = 0.0
        negate = False
        direction = 1
        mask, thresholdUsed = vi.createMask(rawMask, negate, direction, threshold)
        vi.applyMask()
        result = vi.GetImage()

        if arguments.scratch is not None:
            print(f"Write to: {scratch}\{os.path.basename(image)}")
            cv2.imwrite(f"{scratch}\{os.path.basename(image)}", result)

        # Find the blobs
        finalImage = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        manipulated = ImageManipulation(finalImage, 0, logger)
        strategy = constants.Strategy.CARTOON
        # Find the blobs -- use a minimum size of 100 pixels (completely arbitrary)
        contours, hierarchy, blobs, largest = manipulated.findBlobs(500, strategy)
        print("Found blobs: {}".format(len(blobs)))
        if len(blobs) == 0:
            print(f"Something is wrong with {image}. Unable to find blobs")
            continue
            #sys.exit(-1)

        # if len(blobs) != 2:
        #     print(f"Something is wrong with {image}. Failed to find two blobs")
        #     sys.exit(-1)

        # Calculate the average value of them and store the result
        manipulated.toHSI()
        manipulated.extractImagesFrom(manipulated.hsi, 0, constants.NAME_HUE, np.nanmean)
        totalHue = 0.0
        averageHue = 0.0
        for blobName, blobAttributes in blobs.items():
            if blobName == "blob-0":
                averageHue = blobAttributes[constants.NAME_HUE]
            print(f"Blob: {blobName}: {blobAttributes[constants.NAME_HUE]}")
        #     totalHue += blobAttributes[constants.NAME_HUE]
        # averageHue = totalHue / len(blobs)
        print(f"Hue average: {averageHue}")
        if averageHue > 0.0:
            data.append([acquisitionDate, averageHue, os.path.basename(image)])

    calibrations = pd.DataFrame(data, columns=["Date", "Hue", "Image"])
    calibrations = calibrations.set_index("Image")

    return calibrations

# The uncorrected readings
uncorrected = processDirectory(arguments.uncorrected, arguments.scratch + "/" + "uncalibrated")
uncorrected.sort_index(inplace=True)
uncorrected['acquired'] = uncorrected["Date"]

# The corrected readings
corrected = processDirectory(arguments.corrected, arguments.scratch + "/" + "calibrated")
corrected.sort_index(inplace=True)
corrected['acquired'] = corrected["Date"]

combined = uncorrected
combined['Corrected'] = corrected['Hue']
combined.reset_index(inplace=True)
combined = combined.set_index("Date")
combined.sort_index(inplace=True)
plt.style.use('ggplot')
plt.scatter(combined.index, combined['Hue'], label='Uncorrected', color='blue')
plt.scatter(combined.index, combined['Corrected'], label='Corrected', color='green')

# ax = combined.plot(kind='scatter', x='acquired', y='Hue')
# ax = combined.plot(kind='scatter', x='acquired', y='Corrected', color='green')
plt.xlabel("Acquisition Date")
plt.ylabel("Hue")
plt.title("Blue Square of Calibration Plate")
legend = plt.legend()
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.0)

if arguments.output is None:
    plt.show()
else:
    plt.savefig(arguments.output)

sys.exit(0)

