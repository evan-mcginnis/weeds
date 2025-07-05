#
#
# NOTE:
# This will not process large images unless opencl is disabled:
# OPENCV_OPENCL_DEVICE=disabled

import argparse
import sys
import numpy as np
import cv2 as cv
import imutils
from imutils import paths

from CameraFile import CameraFile, CameraPhysical
from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from ImageLogger import ImageLogger

ALG_INCREMENTAL = "incremental"
ALG_ALL = "all"

parser = argparse.ArgumentParser("Row image processsor")

parser.add_argument('-i', '--input', action="store", required=True, help="Images directory")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-v", "--verbose", action="store_true", default=False)
parser.add_argument("-n", "--number", action="store", default=2, type=int, help="the number of images to stitch")
parser.add_argument("-a", "--algorithm", action="store", default=ALG_ALL, help="all or incremental")

results = parser.parse_args()

# Read all the images
images = []
print("Loading images...")
imagePaths = sorted(list(paths.list_images(results.input)))
panorama = None
# loop over the image paths, load each one, and add them to our
# images to stitch list
for imagePath in imagePaths:
    image = cv.imread(imagePath)
    images.append(image)



stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)
#stitcher.setRegistrationResol(-1)
#stitcher.setSeamEstimationResol(-1)
#stitcher.setCompositingResol(-1)
#stitcher.setPanoConfidenceThresh(-1)
#stitcher.setWaveCorrection(True)
#stitcher.setWaveCorrectKind(cv.detail.WAVE_CORRECT_HORIZ)
stitchedBaseName = "stiched-"
if results.algorithm == ALG_INCREMENTAL:
    i = 0
    for image in images:
        for j in range(results.number):

            print("Adding image: " + str(i))
            i = i + 1
            if panorama is None:
                panorama = image
            else:
                (status, stitched) = stitcher.stitch([panorama, image])
                panorama = stitched
                # if the status is '0', then OpenCV successfully performed image
                # stitching
                if status == 0:
                    # write the output stitched image to disk
                    print(results.output + "/" + stitchedBaseName + str(i) + ".jpg")
                    cv.imwrite(results.output + "/" + stitchedBaseName + str(i) + ".jpg", panorama)
                else:
                    print("[INFO] image stitching failed ({})".format(status))
                    sys.exit(1)
else:
    i = 0
    # This approach runs out of memory after 6 images or so taken with my phone
    for j in range(0, len(images),2):
        (status, stitched) = stitcher.stitch([images[j],images[j+1]])
        panorama = stitched
        # if the status is '0', then OpenCV successfully performed image
        # stitching
        if status == 0:
            # write the output stitched image to disk
	        cv.imwrite(results.output + "/" + stitchedBaseName + str(j) + ".jpg", panorama)
        else:
            print("[INFO] image stitching failed ({})".format(status))

sys.exit(0)

# This method is unable to stitch photos the previous method handled without complaint.

