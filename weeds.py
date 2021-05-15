#
# W E E D S
#

import argparse
import sys
import numpy as np
import cv2 as cv
from CameraFile import CameraFile, CameraPhysical
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Classifier import Classifier

veg = VegetationIndex()

parser = argparse.ArgumentParser("Weed recognition system")

parser.add_argument('-i', '--input', action="store", help="Images directory", default="./images")
parser.add_argument('-o', '--output', action="store", help="Output directory for processed images")
parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms(),
                    default="ngrdi")
parser.add_argument("-t", "--threshold", action="store", type=int, default=0)
parser.add_argument("-v", "--verbose", action="store_true")

results = parser.parse_args()

#
# C A M E R A
#

def startupCamera():
    if results.input != None:
        # Get the images from a directory
        camera = CameraFile(results.input)
    else:
        # Get the images from an actual camera
        camera = CameraPhysical("")

    if not camera.connect():
        print("Unable to connect to camera.")
        sys.exit(1)

    # Test the camera
    diagnosticResult, diagnosticText = camera.diagnostics()
    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)
    return camera

def startupOdometer():
    return

def startupLogger() -> Logger:
    logger = Logger()
    if not logger.connect("./output"):
        print("Unable to connect to logging")
        sys.exit(1)
    return logger

#
# Start up various subsystems
#

camera = startupCamera()
startupOdometer()
logger = startupLogger()
#
# G R A N D  L O O P
#
try:
    # Loop and process images until requested to stop
    # TODO: Accept signal to stop processing
    while True:
        image = camera.capture()
        #ImageManipulation.show("Source",image)
        veg.SetImage(image)

        manipulated = ImageManipulation(image)
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        index = veg.Index(results.algorithm)

        # Get the mask
        mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)

        veg.applyMask()
        #image = veg.GetMaskedImage()
        image = veg.GetImage()
        #ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(image)

        # Find the plants in the image
        contours, hierarchy, blobs, largest = manipulated.findBlobs(500)

        classifier = Classifier(blobs)

        classifier.classifyByRatio(largest, 5)

        classifiedBlobs = classifier.blob

        # Draw boxes around the images we found
        #manipulated.drawBoundingBoxes(contours)
        manipulated.drawBoxes(classifiedBlobs)

        # Write out the processed image
        #cv.imwrite("processed.jpg", manipulated.image)
        logger.logImage("processed", manipulated.image)
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

except IOError:
    print("I/O error")
    sys.exit(1)
except EOFError:
    print("End of input")
    sys.exit(0)

