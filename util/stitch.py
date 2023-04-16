#!/usr/bin/env wpython
import cv2
import argparse

parser = argparse.ArgumentParser("Image stitcher")

parser.add_argument('images', metavar='image', type=str, nargs='+')
parser.add_argument('-o', '--output', action="store", required=False, default="stitched.jpg", help="Output file")
arguments = parser.parse_args()

#stitcher = cv2.createStitcher(False)
stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
stitcher = cv2.Stitcher_create(0)


allImages = []
for image in arguments.images:
    print("Loading {}".format(image))
    rgb = cv2.imread(image)
    allImages.append(rgb)

status, pano = stitcher.stitch(allImages)
if status == 0:
    cv2.imwrite(arguments.output, pano)
else:
    print("Unable to stitch images. Return code: {}".format(status))

