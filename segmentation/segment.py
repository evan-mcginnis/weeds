import argparse
import os.path
import glob
import logging.config
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.color import rgb2yiq

from Classifier import Classifier
from Classifier import ClassificationTechniques
from Classifier import classifierFactory
from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger

classifications = [c.name for c in ClassificationTechniques]
classificationChoices = classifications
classificationChoices.append("all")

# Names of the color spaces that can be used
spaces = ["BGR", "RGB", "YIQ", "YUV", "HSI", "HSV", "YCBCR", "CIELAB"]
spaceChoices = spaces
spaceChoices.append("all")

parser = argparse.ArgumentParser("ML Image Segmentation")

parser.add_argument('-i', '--input', action="store", required=True,  help="Input Image or directory")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory")
parser.add_argument('-t', '--training', action="store", required=True, help="Training file")
parser.add_argument('-l', '--logging', action="store", required=True, help="Logging configuration")
parser.add_argument('-a', '--algorithm', action="store", required=False, default="KNN", choices=classificationChoices, help="ML Algorithm")
parser.add_argument('-c', '--color', action="store", required=False, default="BGR", choices=spaceChoices, help="Color Spaces")
parser.add_argument('-s', '--score', action="store_true", required=False, default=True, help="Score the model")


arguments = parser.parse_args()


# Input files -- find all jpgs if a directory is specified
files = []
if os.path.isfile(arguments.input):
    files = [arguments.input]
elif os.path.isdir(arguments.input):
    files = glob.glob(arguments.input + "/*.jpg")
    if len(files) == 0:
        print(f"Unable to find JPGs in {arguments.input}")
        exit(-1)

# Output should be to an existing directory
if os.path.isfile(arguments.output):
    print(f"Output must be an existing directory: {arguments.output}")
    exit(-1)
if arguments.input == arguments.output:
    print(f"Input directory and output directory must differ")
    exit(-1)

# The image logger used
logger = ImageLogger()
logger.connect(arguments.output)
logger.autoIncrement = True

# Read training data
if not os.path.isfile(arguments.training):
    print(f"Unable to access training file: {arguments.training}")
    exit(-1)
else:
    try:
        # Easier to read it with pandass and then convert it to numpy
        trainingDF = pd.read_csv(arguments.training)
        training = trainingDF.to_numpy()
    except Exception as e:
        print(f"Unable to read CSV: {e}")
        exit(-1)

# Logging configuration
if os.path.isfile(arguments.logging):
    try:
        logging.config.fileConfig(arguments.logging)
    except Exception as e:
        print("Unable to configure logging: {e}")
    log = logging.getLogger("segment")
else:
    print(f"Unable to access logging configuration: {arguments.logging}")

def process(algorithm: str, color: str, source: str, logger: ImageLogger):
    pass

classifier = classifierFactory(arguments.algorithm)

# The columns to be read from the training file
band0 = f"{arguments.color}-band-0"
band1 = f"{arguments.color}-band-1"
band2 = f"{arguments.color}-band-2"
features = [band0, band1, band2]
columns = ["type", band0, band1, band2]

# Set up the classifier
classifier.selections = features
# TODO: stratify should be false for random forest
classifier.load(arguments.training, stratify=False)
classifier.createModel(arguments.score)

log.debug(f"Reading {columns} from {arguments.training}")

for file in files:
    log.debug(f"Processing {file} using {classifier.name}")
    # Read file and convert to color spaces
    imgAsBGR = cv.imread(file, cv.IMREAD_COLOR)

    # The target image to be manipulated
    targetImage = np.copy(imgAsBGR)

    #cv.imwrite(arguments.output + "/before-" + os.path.basename(file), targetImage)
    logger.logImage("before-" + Path(file).stem, targetImage)

    manipulated = ImageManipulation(imgAsBGR, 0, logger)
    # OpenCV treats images as BGR, not RGB
    imgAsRGB = cv.cvtColor(imgAsBGR.astype(np.uint8), cv.COLOR_BGR2RGB)
    imgAsHSV = cv.cvtColor(imgAsBGR.astype(np.uint8), cv.COLOR_BGR2HSV)
    imgAsYCBCR = cv.cvtColor(imgAsBGR.astype(np.uint8), cv.COLOR_BGR2YCR_CB)
    imgAsCIELAB = cv.cvtColor(imgAsBGR.astype(np.uint8), cv.COLOR_BGR2Lab)
    imgAsYUV = cv.cvtColor(imgAsBGR.astype(np.uint8), cv.COLOR_BGR2YUV)

    # This method takes RGB as input, not BGR
    imgAsYIQ = rgb2yiq(imgAsRGB)

    imgAsHSI = manipulated.toHSI()

    allBands = {
        "BGR": {"readings": imgAsBGR},
        "RGB": {"readings": imgAsRGB},
        "YIQ": {"readings": imgAsYIQ},
        "YUV": {"readings": imgAsYUV},
        "HSI": {"readings": imgAsHSI},
        "HSV": {"readings": imgAsHSV},
        "YCBCR": {"readings": imgAsYCBCR},
        "CIELAB": {"readings": imgAsCIELAB}
    }

    # For each pixel, predict the class
    height, width, bands = imgAsBGR.shape

    for h in range(height):
        log.debug(f"Processing row {h}")
        for w in range(width):
            # Retrieve the colorspace requested
            bandInformation = allBands[arguments.color]
            band = bandInformation["readings"]

            # Predict based on the band data
            predicted = classifier.classifyPixel(band[h, w, 0], band[h, w, 1], band[h, w, 1])

            # For the ground, set all the bands to 0 to indicate black
            if predicted == 1:
                targetImage[h, w, 0] = 0
                targetImage[h, w, 1] = 0
                targetImage[h, w, 2] = 0

    logger.logImage("after-" + Path(file).stem, targetImage)
