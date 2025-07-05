import glob
import os.path
import math
import sys

import cv2
import argparse
from tqdm import tqdm

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def calculate_aspect(width: int, height: int) -> str:
    r = math.gcd(width, height)
    x = int(width / r)
    y = int(height / r)
    return f"{x}:{y}"

parser = argparse.ArgumentParser("Image Operations")
parser.add_argument("-e", "--encoding", action="store", required=False, default="jpg", choices=["jpg", "png"], help="Image encoding")
parser.add_argument("-i", "--input", action="store", required=True, help="Source image(s)")
parser.add_argument("-o", "--output", action="store", required=True, help="Altered image or directory")
parser.add_argument("-r", "--resize", action="store", type=tuple_type, required=False, help="Resize to these dimensions: (Y, X)")
parser.add_argument("-or", "--orientation", action="store", required=False, default="unchanged", choices=["unchanged", "portrait", "landscape"], help="Preferred orientation")
parser.add_argument("-t", "--type", action="store", required=False, default="image", choices=["image", "mask"], help="Source type")
parser.add_argument("-d", "--drop", action="store", type=tuple_type, required=False, help="Drop pixels to these dimensions: (Y, X)")


arguments = parser.parse_args()

processingMultipleImages = False
files = []

# Operate on a single file
if os.path.isfile(arguments.input):
    files = [arguments.input]
# Operate on a directory
elif os.path.isdir(arguments.input):
    targetFiles = arguments.input + "/*." + arguments.encoding
    files = glob.glob(targetFiles)
    if len(files) == 0:
        print(f"Unable to find images using {targetFiles}")
        exit(-1)
    processingMultipleImages = True
else:
    print(f"Unable to access: {arguments.input}")
    sys.exit(-1)

# Make certain that an output directory is specified if we are processing multiple files
if processingMultipleImages:
    if os.path.isfile(arguments.output):
        print(f"Output must be a directory when processing multiple files")
        exit(-1)
    if not os.path.isdir(arguments.output):
        print(f"Output must be an existing directory")
        exit(-1)

if arguments.input == arguments.output:
    print(f"Unable to read and write to the same directory.")
    exit(-1)

for i in tqdm(range(len(files))):
    file = files[i]
    if arguments.type == "image":
        img = cv2.imread(file)
    elif arguments.type == "mask":
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    else:
        print(f"Unknown type: {arguments.type}")
        sys.exit(-1)

    # Get original height and width
    #print(f"Original Dimensions : {img.shape}")
    #print(f"Required: {arguments.resize}")

    # Determine if image is portrait or landscape
    # and perform the transform to match the desired orientation

    # Portrait
    if img.shape[0] > img.shape[1]:
        if arguments.orientation == "landscape":
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # Landscape
    elif img.shape[0] < img.shape[1]:
        if arguments.orientation == "portrait":
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Square
    else:
        pass

    # if arguments.orientation == "portrait":
    #     img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # elif arguments.orientation == "landscape":
    #     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # elif arguments.orientation == "unchanged":
    #     pass
    # else:
    #     print(f"Unknown orientation: {arguments.orientation}")
    #     sys.exit(-1)

    #print(f"Aspect Ratio of {file}: {calculate_aspect(img.shape[0], img.shape[1])}")
    if arguments.drop is not None:
        if arguments.type == "image":
            assert(len(img.shape) == 3)
            newImage = img[0:arguments.drop[0], 0:arguments.drop[1], :]
        elif arguments.type == "mask":
            assert(len(img.shape) == 2)
            newImage = img[0:arguments.drop[0], 0:arguments.drop[1]]

        cv2.imwrite(arguments.output + "/" + os.path.basename(file), newImage)

    if arguments.resize is not None:
        #print(f"Aspect Ratio specified: {calculate_aspect(arguments.resize[0], arguments.resize[1])}")

        # resize image by specifying custom width and height
        resized = cv2.resize(img, arguments.resize)

        #print(f"Resized {file} to {resized.shape}")
        if processingMultipleImages:
            cv2.imwrite(arguments.output + "/" + os.path.basename(file), resized)
        else:
            cv2.imwrite(arguments.output, resized)
