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
from Odometer import VirtualOdometer

veg = VegetationIndex()

parser = argparse.ArgumentParser("Weed recognition system")

parser.add_argument('-i', '--input', action="store", help="Images directory")
parser.add_argument('-o', '--output', action="store", help="Output directory for processed images")
parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms(),
                    default="ngrdi")
parser.add_argument("-t", "--threshold", action="store", type=int, default=0)
parser.add_argument("-s", "--stitch", action="store_true", help="Stitch images together")
parser.add_argument("-v", "--verbose", action="store_true", default=False)

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

    (w, h) = camera.getResolution()

    # Test the camera
    diagnosticResult, diagnosticText = camera.diagnostics()
    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)
    return camera

#
# O D O M E T E R
#

def startupOdometer():
    odometer = VirtualOdometer("")
    if not odometer.connect():
        print("Unable to connect to odometer")
        sys.exit(1)

    diagnosticResult, diagnosticText = odometer.diagnostics()

    if not diagnosticResult:
        print(diagnosticText)
        sys.exit(1)

    return odometer

#
# L O G G E R
#

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
odometer = startupOdometer()
logger = startupLogger()

mmPerPixel = camera.getMMPerPixel()

previousImage = None
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
        manipulated.mmPerPixel = mmPerPixel
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

        # TODO: Simply this to just imsge=brg.GetMaskedImage(results.algorithm)
        # Compute the index using the requested algorithm
        index = veg.Index(results.algorithm)
        ImageManipulation.show("index", index)
        cv.imwrite("index.jpg", index)

        # Get the mask
        mask, threshold = veg.MaskFromIndex(index, True, 1, results.threshold)

        veg.applyMask()
        #image = veg.GetMaskedImage()
        image = veg.GetImage()
        #ImageManipulation.show("Masked", image)

        manipulated = ImageManipulation(image)
        manipulated.mmPerPixel = mmPerPixel

        # Find the plants in the image
        contours, hierarchy, blobs, largest = manipulated.findBlobs(500)

        manipulated.identifyOverlappingVegetation()

        classifier = Classifier(blobs)

        classifier.classifyByRatio(largest, size=manipulated.image.shape, ratio=5)

        classifiedBlobs = classifier.blob

        # Draw boxes around the images we found
        #manipulated.drawBoundingBoxes(contours)
        manipulated.drawBoxes(classifiedBlobs)

        # Crop row processing
        manipulated.identifyCropRowCandidates()
        manipulated.substituteRectanglesForVegetation()

        #logger.logImage("cropline", manipulated.croplineImage)
        manipulated.detectLines()
        manipulated.drawCropline()
        logger.logImage("crop-line", manipulated.croplineImage)
        manipulated.drawContours()

        # Just a test of stitching. This needs some more thought
        # we can't stitch things where there is nothing in common between the two images
        # even if there is overlap. It may just be black background after the segmentation.
        # One possibility here is to put something in the known overlap area that can them be used
        # to align the images.
        # The alternative is to use the original images and use the soil as the element that is common between
        # the two.  The worry here is computational efficiency

        if results.stitch:
            if previousImage is not None:
                manipulated.stitchTo(previousImage)
            else:
                previousImage = image

        # Write out the processed image
        #cv.imwrite("processed.jpg", manipulated.image)
        logger.logImage("processed", manipulated.image)
        #ImageManipulation.show("Greyscale", manipulated.toGreyscale())

except IOError:
    print("There was a problem communicating with the camera")
    sys.exit(1)
except EOFError:
    print("End of input")
    sys.exit(0)

