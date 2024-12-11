import os
import sys
import argparse
import glob
import cv2
from pathlib import Path
from tqdm import tqdm

from Mask import Mask

parser = argparse.ArgumentParser("Apply masks")
parser.add_argument("-ie", "--image-encoding", action="store", required=False, default="jpg", choices=["jpg", "png"], help="Image encoding")
parser.add_argument("-me", "--mask-encoding", action="store", required=False, default="jpg", choices=["jpg", "png"], help="Mask encoding")
parser.add_argument("-i", "--image", action="store", required=False, help="Source image(s)")
parser.add_argument("-m", "--mask", action="store", required=False, help="Source mask(s)")
parser.add_argument("-o", "--output", action="store", required=True, help="Altered image or directory")


arguments = parser.parse_args()

processingMultipleImages = False

# The images we will process
imagesToProcess = []
masksToProcess = []

# Did user specify a directory of images or a single file?
if os.path.isdir(arguments.image):
    if arguments.image_encoding is None or arguments.mask_encoding is None:
        print(f"The encoding for images and masks must be specified for processing directories")
        sys.exit(-1)

    images = glob.glob(arguments.image + "/*." + arguments.image_encoding)
    masks = glob.glob(arguments.mask + "/*." + arguments.mask_encoding)
    imagesToProcess = images
    masksToProcess = masks
    if len(images) == 0 or len(masks) == 0:
        print(f"Unable to access images ({arguments.image}) and masks ({arguments.mask})")
        sys.exit(-1)
    elif len(images) != len(masks):
        print(f"The number of images ({len(images)} must be the same as masks ({len(masks)})")
        sys.exit(-1)
else:
    if not os.path.isfile(arguments.image):
        print(f"Unable to access image: {arguments.image}")
        sys.exit(-1)
    else:
        imagesToProcess.append(arguments.images)
    if not os.path.isfile(arguments.mask):
        print(f"Unable to access mask: {arguments.mask}")
        sys.exit(-1)
    else:
        masksToProcess.append(arguments.mask)

if not os.path.isdir(arguments.output):
    print("Unable to access output directory")
    sys.exit(-1)

# Process each of the images
for i in tqdm(range(len(imagesToProcess))):
    imageName = imagesToProcess[i]
    maskName = masksToProcess[i]

    theMask = Mask()

    theMask.load(maskName)
    maskedImage = theMask.apply(imageName)
    cv2.imwrite(arguments.output + "/" + Path(imageName).stem + "-masked." + arguments.image_encoding, maskedImage)

sys.exit(0)





