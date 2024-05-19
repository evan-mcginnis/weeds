import glob
import os.path

import cv2
import argparse

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

parser = argparse.ArgumentParser("Image Operations")
parser.add_argument("-i", "--input", action="store", required=True, help="Source image(s)")
parser.add_argument("-o", "--output", action="store", required=True, help="Altered image or directory")
parser.add_argument("-r", "--resize", action="store", type=tuple_type, required=False, help="Resize to these dimensions: (Y, X)")


arguments = parser.parse_args()

processingMultipleImages = False

# Operate on a single file
if os.path.isfile(arguments.input):
    files = [arguments.input]
# Operate on a directory
elif os.path.isdir(arguments.input):
    targetFiles = arguments.input + "/*.jpg"
    files = glob.glob(targetFiles)
    if len(files) == 0:
        print(f"Unable to find images using {targetFiles}")
        exit(-1)
    processingMultipleImages = True

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


for file in files:
    img = cv2.imread(file)

    # Get original height and width
    #print(f"Original Dimensions : {img.shape}")
    #print(f"Required: {arguments.resize}")

    # resize image by specifying custom width and height
    resized = cv2.resize(img, arguments.resize)

    print(f"Resized {file} to {resized.shape}")
    if processingMultipleImages:
        cv2.imwrite(arguments.output + "/" + os.path.basename(file), resized)
    else:
        cv2.imwrite(arguments.output, resized)
