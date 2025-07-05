import argparse
import os.path
import glob
import logging.config
import sys
import threading
from pathlib import Path
import time
from threading import Thread, Semaphore
from enum import Enum

import numpy as np
import pandas as pd
import cv2 as cv
from skimage.color import rgb2yiq

from Classifier import Classifier
from Classifier import ClassificationTechniques
from Classifier import classifierFactory
from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger
from Performance import Performance

# This is temporary to debug
class _ClassificationTechniques(Enum):
    KNN = 0
    RANDOMFOREST = 1
    DECISIONTREE = 2
    SVM = 3
    LDA = 4
    #MLP = 5
    LOGISTIC = 6
    GRADIENT = 7
    EXTRA = 8

    def __str__(self):
        return self.name
# End debug

classifications = [c.name for c in ClassificationTechniques]
classificationChoices = classifications.copy()
classificationChoices.append("all")

# Names of the color spaces that can be used
spaces = ["RGB", "YIQ", "YUV", "HSI", "HSV", "YCBCR", "CIELAB"]
spaceChoices = spaces.copy()
spaceChoices.append("all")

parser = argparse.ArgumentParser("ML Image Segmentation")

parser.add_argument('-i', '--input', action="store", required=True,  help="Input Image or directory")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory")
parser.add_argument('-t', '--training', action="store", required=True, help="Training file")
parser.add_argument('-l', '--logging', action="store", required=True, help="Logging configuration")
parser.add_argument('-a', '--algorithm', action="store", required=False, default="KNN", choices=classificationChoices, nargs='+', help="ML Algorithm")
parser.add_argument('-c', '--color', action="store", required=False, default="BGR", choices=spaceChoices, help="Color Spaces")
parser.add_argument('-s', '--score', action="store_true", required=False, default=True, help="Score the model")
parser.add_argument('-b', '--build', action="store_true", required=False, default=False, help="Build model and summarize -- no actual processing")


arguments = parser.parse_args()

#
# Perfomance
#
performanceResults = Performance("performance.csv")
(performanceOK, performanceDiagnostics) = performanceResults.initialize()
if not performanceOK:
    print(performanceDiagnostics)
    sys.exit(1)

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

# Process all the images using a specific algorithm

def process(algorithm: str, source: str, logger: ImageLogger):

    for color in spaces:
        classifier = classifierFactory(algorithm)

        # The columns to be read from the training file
        band0 = f"{color}-band-0"
        band1 = f"{color}-band-1"
        band2 = f"{color}-band-2"
        features = [band0, band1, band2]
        columns = ["type", band0, band1, band2]

        # Set up the classifier
        classifier.selections = features
        stratify = False
        if algorithm == ClassificationTechniques.RANDOMFOREST.name:
            stratify = True
        classifier.load(arguments.training, stratify=stratify)
        try:
            classifier.createModel(arguments.score)
        except Exception as e:
            log.fatal(f"Unable to create model: {e}")
            threadStatus[threading.current_thread().name] = -1
            sys.exit()

        log.debug(f"Reading {columns} from {arguments.training}")

        for file in files:
            log.debug(f"Processing {file} using {classifier.name}")
            # Read file and convert to color spaces
            imgAsBGR = cv.imread(file, cv.IMREAD_COLOR)

            # The target image to be manipulated
            targetImage = np.copy(imgAsBGR)

            #cv.imwrite(arguments.output + "/before-" + os.path.basename(file), targetImage)
            logger.logImage(f"{Path(file).stem}-before", targetImage)

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

            performanceResults.start()
            for h in range(height):
                log.debug(f"Processing row {h}")
                for w in range(width):
                    # Retrieve the colorspace requested
                    bandInformation = allBands[color]
                    band = bandInformation["readings"]

                    # Predict based on the band data
                    try:
                        predicted = classifier.classifyPixel(band[h, w, 0], band[h, w, 1], band[h, w, 2])
                    except Exception as e:
                        log.fatal(f"Unable to classify: {e}")
                        threadStatus[threading.current_thread().name] = -1
                        sys.exit()

                    # For the ground, set all the bands to 0 to indicate black
                    if predicted == 1:
                        targetImage[h, w, 0] = 0
                        targetImage[h, w, 1] = 0
                        targetImage[h, w, 2] = 0

            performanceResults.stopAndRecord(f"{algorithm}-{color}")
            logger.logImage(f"{Path(file).stem}-{algorithm}-{color}", targetImage)

if arguments.algorithm[0] == "all":
    allTechniques = classifications
else:
    allTechniques = arguments.algorithm

log.debug(f"Examine using {allTechniques}")

threads = []
threadStatus = {}

# Just build the classifiers
if arguments.build:
    allResults = []

    for algorithm in allTechniques:
        for color in spaces:
            log.debug(f"Building {algorithm}-{color}")
            classifier = classifierFactory(algorithm)

            # The columns to be read from the training file
            band0 = f"{color}-band-0"
            band1 = f"{color}-band-1"
            band2 = f"{color}-band-2"
            features = [band0, band1, band2]
            columns = ["type", band0, band1, band2]

            # Set up the classifier
            classifier.selections = features
            stratify = False
            if algorithm == _ClassificationTechniques.RANDOMFOREST.name:
                stratify = True
            classifier.load(arguments.training, stratify=stratify)
            classifier.createModel(arguments.score)
            classifier.assess()
            result = [algorithm, color, classifier.auc, classifier.precision, classifier.recall, classifier.f1]
            allResults.append(result)

    # The results from building the models go here
    columns = ["Technique", "Color", "AUC", "Precision", "Recall", "F1"]
    results = pd.DataFrame(columns=columns, data=allResults)

    shortCaption = "Machine Learning Segmentation"
    longCaption = "Machine Learning Segmentation"
    print("---------- begin latex ---------------")
    print(f"{results.to_latex(longtable=True, index_names=False, index=False, caption=(longCaption, shortCaption), float_format='%.2f', label='table:ml-segmentation', header=columns)}")
    print("---------- end latex ---------------")

    results.to_csv(f"{arguments.output}/results.csv")

# Process images
else:
    for technique in allTechniques:
        search = Thread(name=f"{technique}",
                        target=process,
                        args=(technique, arguments.input, logger,))

        search.daemon = True
        threads.append(search)
        threadStatus[search.name] = 0
        search.start()
        # This is arbitrary but required to avoid errors in startup, it would seem.
        time.sleep(2)

    # Wait for the threads to finish
    log.info(f"Wait for {len(threads)} threads to finish")
    finished = False
    while not finished:
        finishedThreads = 0
        for x in threads:
            if not x.is_alive():
                log.debug(f"{x.name} finished")
                status = threadStatus[x.name]
                if status != 0:
                    log.error(f"Thread {x.name} terminated with an error.")
                    finished = True
                finishedThreads += 1
        if finishedThreads == len(threads):
            finished = True
            time.sleep(5)

