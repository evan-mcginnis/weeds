import argparse
import sys
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import logging
import logging.config

import pandas as pd

from Mask import Mask
from Mask import Rate


parser = argparse.ArgumentParser("Evaluate Mask Differences", epilog="Plot error rates for mask production")

parser.add_argument('-i', '--input', action="store", required=True, help="Input mask or directory")
parser.add_argument('-s', '--source', action="store", required=True, help="Original source image or directory")
#parser.add_argument('-p', '--processing', action="store", required=False, default="mask", help="Operation")
parser.add_argument('-t', '--target', action="store", required=True, help="Directory of reference masks")
parser.add_argument('-a', '--after', action="store", required=False, help="Directory where masks produced from improved images live. Produces a dumbbell plot.")
parser.add_argument('-o', '--output', action="store", required=False, default="results.csv", help="Results csv")
parser.add_argument('-l', '--logging', action="store", required=False, default="logging.ini", help="Logging configuration file")

arguments = parser.parse_args()

if not os.path.isfile(arguments.logging):
    print("Unable to access logging configuration file {}".format(arguments.logging))
    sys.exit(1)

# Initialize logging
logging.config.fileConfig(arguments.logging)
# The masks to evaluate
masksToEvaluate = []

if os.path.isdir(arguments.input):
    evaluateSingleImage = False
    # If the masks to evaluate is a directory, the sources and reference mask should be as well
    if not os.path.isdir(arguments.source):
        print(f"Source images must be directory if target is")
        sys.exit(-1)
    if not os.path.isdir(arguments.target):
        print(f"Reference masks must be directory if target is")
        sys.exit(-1)

    # Check access to source images
    sourceFiles = glob.glob(arguments.source + "*.jpg")
    if len(sourceFiles) == 0:
        print(f"Unable to find source images in {arguments.source}")
        sys.exit(-1)

    # Find all the masks for all the images
    masksToEvaluate = glob.glob(arguments.input + "*-mask-*.jpg")
    if len(masksToEvaluate) == 0:
        print(f"Unable to find masks in {arguments.input}")
        sys.exit(-1)

    # Walk through all the sources, and make sure there is at least one mask for it
    for source in sourceFiles:
        baseSource = Path(source).stem
        # Check to see if corresponding masks are there
        expectedMasks = glob.glob(arguments.input + baseSource + "-mask-*.jpg")
        if len(expectedMasks) == 0:
            print(f"Unable to find masks for image: {baseSource}")
            sys.exit(-1)
else:
    # We are evaluating a single image
    evaluateSingleImage = True

    # Find the masks associated with that image
    masksToEvaluate = glob.glob(arguments.input + "-mask-*.jpg")
    if len(masksToEvaluate) == 0:
        print(f"Unable to access: {arguments.input}-mask-*.jpg")
        sys.exit(-1)

    # Find the source image -- this could be specified as the source directory, and we should figure it out,
    # or specified directly

    # Check access to source image if specified directly
    if os.path.isfile(arguments.source):
        sourceFiles = [arguments.source]
    # or if the directory was specified. just use the input specified
    elif os.path.isdir(arguments.source):
        sourceFile = arguments.source + Path(arguments.input).stem + ".jpg"
        if not os.path.isfile(sourceFile):
            print(f"Unable to access source image: {sourceFile}")
            sys.exit(-1)
        sourceFiles = [sourceFile]

# For each of the masks to evaluate, there must be a reference
if os.path.isdir(arguments.target):
    referenceMasks = glob.glob(arguments.target + "*-mask.jpg")
    if len(referenceMasks) == 0:
        print(f"Unable to access reference masks in directory: {arguments.target}")
        sys.exit(-1)
elif os.path.isfile(arguments.target):
    referenceMasks = [arguments.target]

for mask in masksToEvaluate:
    # Get the basename of the file
    baseFileName = Path(mask).stem
    # Should be in the form IMG_1110-mask-COM2, so just take the first bit
    fileNameWithoutTechnique = baseFileName.split('-')[0]
    referenceMask = fileNameWithoutTechnique + "-mask.jpg"
    # There are two choices here -- the reference mask is directly specified or the directory is
    # and we just figure it out
    if os.path.isdir(arguments.target):
        referenceMask = arguments.target + fileNameWithoutTechnique + "-mask.jpg"
    else:
        referenceMask = arguments.target
    if not os.path.isfile(referenceMask):
        print(f"Unable to access reference mask: {referenceMask}")
        sys.exit(-1)

if os.path.isfile(arguments.output):
    print(f"Output file exists: {arguments.output}")
    sys.exit(-1)

# Find all the masks with a basename specified
#masks = glob.glob(arguments.target + "-mask-*.jpg")

results = []

# For each one of the files, there should be 1 reference and N techniques
for source in sourceFiles:
    baseSource = Path(source).stem
    if os.path.isdir(arguments.target):
        referenceMask = arguments.target + baseSource + "-mask.jpg"
    elif os.path.isfile(arguments.target):
        referenceMask = arguments.target
    else:
        print(f"Unable to access reference mask in: {arguments.target}")
        sys.exit(-1)

    if os.path.isdir(arguments.input):
        masksToEvaluate = glob.glob(arguments.input + baseSource + "-mask-*.jpg")
    else:
        masksToEvaluate = glob.glob(arguments.input + "-mask-*.jpg")

    if len(masksToEvaluate) == 0:
        print(f"Unable to locate evaluation masks: {arguments.input}")
        sys.exit(-1)

    for mask in masksToEvaluate:
        theMask = Mask()
        theMask.load(mask)
        print(f"\nEvaluating {mask} vs {referenceMask}")
        theMask.compare(referenceMask)
        print(f"{theMask}")
        technique = Path(mask).stem
        print(f"Technique: {technique.split('-')[2]} FPR: {theMask.rate(Rate.FPR)} FNR: {theMask.rate(Rate.FNR)} Total {theMask.differences}")
        #results[technique.split('-')[2]] = [theMask.rate(Rate.FPR), theMask.rate(Rate.FNR), ((theMask.fp + theMask.fn) / (theMask.mask.shape[0] * theMask.mask.shape[1])) * 100]
        results.append([Path(source).stem, technique.split('-')[2], theMask.rate(Rate.FPR), theMask.rate(Rate.FNR), theMask.rate(Rate.FPR) + theMask.rate(Rate.FNR)])

df = pd.DataFrame(results, columns=['source', 'technique', 'FPR', 'FNR', 'Total'])
df.to_csv(arguments.output)
sys.exit(0)

for mask in masksToEvaluate:
    theMask = Mask()
    theMask.load(mask)
    # For each mask, compare it to the reference one produced manually
    theMask.compare(mask)
    # The base filename without the .jpg extension
    technique = Path(mask).stem
    # Without the -mask- bit
    print(f"Technique: {technique.split('-')[2]} FPR: {theMask.rate(Rate.FPR)} FNR: {theMask.rate(Rate.FNR)} Total {theMask.differences}")
    #results[technique.split('-')[2]] = (theMask.differences / (theMask.mask.shape[0] * theMask.mask.shape[1])) * 100
    #results[technique.split('-')[2]] = ((theMask.fp + theMask.fn) / (theMask.mask.shape[0] * theMask.mask.shape[1])) * 100
    results[technique.split('-')[2]] = [theMask.rate(Rate.FPR), theMask.rate(Rate.FNR), ((theMask.fp + theMaskY.fn) / (theMask.mask.shape[0] * theMask.mask.shape[1])) * 100]

# The detaframe now contains the result from the unimproved image/mask combination
df = pd.DataFrame.from_dict(results, orient='index', columns=['I', 'II', 'Total'])


# If the user has specified a set of masks produced from an improved image
if arguments.after is not None:
    print(f"Masks from {arguments.after}")
    masks = glob.glob(arguments.after + "-mask-*.jpg")
    typeI = []
    typeII = []
    totalError = []
    for mask in masks:
        theMask.compare(mask)
        technique = Path(mask).stem
        print(f"Technique: {technique.split('-')[2]} FPR: {theMask.rate(Rate.FPR)} FNR: {theMask.rate(Rate.FNR)} Total {theMask.differences}")
        typeI.append(theMask.rate(Rate.FPR))
        typeII.append(theMask.rate(Rate.FNR))
        totalError.append(theMask.rate(Rate.FPR) + theMask.rate(Rate.FNR))
    df['I-Improved'] = typeI
    df['II-Improved'] = typeII
    df['Total-Improved'] = totalError
    print(f"{pd.DataFrame.to_latex(df)}")
else:
    df = df.sort_values('Total')
    df = df.drop('Total', axis=1)
    df.plot.barh(stacked=True)
    plt.ylabel("Segmentation Algorithm")
    plt.xlabel("% Different from Manual Mask")
    plt.title("Error Rates of Segmentation Algorithms")
    plt.show()

#plt.barh(list(results.keys()), list(results.values()))
#plt.barh('technique', 'percentage', data=df)



sys.exit(0)

