import argparse
import sys
import os.path
import glob
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import pandas as pd

from Mask import Mask
from Mask import Rate


parser = argparse.ArgumentParser("Evaluate Mask Differences", epilog="Plot error rates for mask production")

parser.add_argument('-i', '--input', action="store", required=True, help="Input mask or directory")
parser.add_argument('-s', '--source', action="store", required=True, help="Original source image directory")
#parser.add_argument('-p', '--processing', action="store", required=False, default="mask", help="Operation")
parser.add_argument('-t', '--target', action="store", required=True, help="Directory of reference masks")
parser.add_argument('-a', '--after', action="store", required=False, help="Directory where masks produced from improved images live. Produces a dumbbell plot.")
parser.add_argument('-o', '--output', action="store", required=False, default=".", help="Output Directory for csv")

arguments = parser.parse_args()

# The masks to evaluate
if os.path.isfile(arguments.input):
    masksToEvaluate = [arguments.input]
    # theMask = Mask()
    # if os.path.isfile(arguments.input):
    #     theMask.load(arguments.input)
    # else:
    #     print(f"Unable to access {arguments.input}")
    #     sys.exit(-1)
elif os.path.isdir(arguments.input):
    # The base names to expect
    sourceFiles = glob.glob(arguments.source + "*.jpg")
    if len(sourceFiles) == 0:
        print(f"Unable to find source images in {arguments.source}")
        sys.exit(-1)

    # Find all the masks for all the images
    masksToEvaluate = glob.glob(arguments.input + "*-mask-*.jpg")
    if len(masksToEvaluate) == 0:
        print(f"Unable to find masks in {arguments.input}")

    # Walk through all the sources, and make sure there is at least one mask for it
    for source in sourceFiles:
        baseSource = Path(source).stem
        # Check to see if corresponding masks are there
        expectedMasks = glob.glob(arguments.input + baseSource + "-mask-*.jpg")
        if len(expectedMasks) == 0:
            print(f"Unable to find masks for image: {baseSource}")
            sys.exit(-1)
else:
    print(f"Unable to access: {arguments.input}")
    sys.exit(-1)

# For each of the masks to evaluate, there must be a reference
referenceMasks = glob.glob(arguments.target + "*-mask.jpg")
for mask in masksToEvaluate:
    # Get the basename of the file
    baseFileName = Path(mask).stem
    # Should be in the form IMG_1110-mask-COM2, so just take the first bit
    fileNameWithoutTechnique = baseFileName.split('-')[0]
    referenceMask = fileNameWithoutTechnique + "-mask.jpg"
    if not os.path.isfile(arguments.target + referenceMask):
        print(f"Unable to access reference mask: {referenceMask}")
        sys.exit(-1)


# Find all the masks with a basename specified
#masks = glob.glob(arguments.target + "-mask-*.jpg")

results = []

# For each one of the files, there should be 1 reference and N techniques
for source in sourceFiles:
    baseSource = Path(source).stem
    referenceMask = arguments.target + baseSource + "-mask.jpg"
    masksToEvaluate = glob.glob(arguments.input + baseSource + "-mask-*.jpg")
    for mask in masksToEvaluate:
        theMask = Mask()
        theMask.load(mask)
        print(f"Evaluating {mask} vs {referenceMask}")
        theMask.compare(referenceMask)
        technique = Path(mask).stem
        print(f"Technique: {technique.split('-')[2]} FPR: {theMask.rate(Rate.FPR)} FNR: {theMask.rate(Rate.FNR)} Total {theMask.differences}")
        #results[technique.split('-')[2]] = [theMask.rate(Rate.FPR), theMask.rate(Rate.FNR), ((theMask.fp + theMask.fn) / (theMask.mask.shape[0] * theMask.mask.shape[1])) * 100]
        results.append([Path(source).stem, technique.split('-')[2], theMask.rate(Rate.FPR), theMask.rate(Rate.FNR), ((theMask.fp + theMask.fn) / (theMask.mask.shape[0] * theMask.mask.shape[1])) * 100])

df = pd.DataFrame(results, columns=['source', 'technique', 'I', 'II', 'Total'])
df.to_csv("results.csv")
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

