#
# Organize image files
#

import argparse
import sys
import os
import shutil
import pandas as pd
from Metadata import Metadata
import constants

def closestAGL(possibleAGLs: [], recordedAGL: float) -> float:
    """
    Determine the AGL closest to one in the list given
    :param possibleAGLs: List of possible AGLs
    :param recordedAGL: AGL recorded
    :return: one of the AGLs in the possibleAGLs list
    """
    closest = possibleAGLs[0]
    for i in possibleAGLs:
        if abs(i - recordedAGL) < closest:
            closest = i
    # print(f"{recordedAGL} Closest AGL {closest}")
    return closest

parser = argparse.ArgumentParser("Organize photos by altitude")

parser.add_argument("-i", "--input", action="store", required=True, help="Source directory for images")
parser.add_argument("-o", "--output", action="store", required=True, help="Output directory")
parser.add_argument("-a", "--agl", action="store", type=float, nargs="+", required=True, help="List of AGL captured")
parser.add_argument("-c", "--crop", action="store", required=True, help="crop")
parser.add_argument("-f", "--force", action="store_true", required=False, default=False, help="Force overwrite of existing files")
parser.add_argument("-m", "--msl", type=float, action="store", required=True, help="The altitude above MSL")
parser.add_argument("-r", "--range", action="store", required=False, default=1.0, help="MSL range")
arguments = parser.parse_args()

COLUMN_PATH = "path"
COLUMN_ALTITUDE = "altitude"
COLUMN_AGL = "agl"

# Assume a flat source directory, and find all the files
allFiles = []
allAltitudes = []
correctedAGL = []
included_extensions = ['JPG', 'DNG']
file_names = [fn for fn in os.listdir(arguments.input)
              if any(fn.endswith(ext) for ext in included_extensions)]

# Create the lista and dataframe
for aFile in file_names:
    # check if current file_path is a file
    filePath = os.path.join(arguments.input, aFile)
    if os.path.isfile(filePath):
        # add filename to list
        allFiles.append(os.path.join(arguments.input, aFile))
        meta = Metadata(filePath)
        meta.getMetadata()
        allAltitudes.append(meta.altitude)
        # Determine AGL, as we don't really care about the altitude
        correctedAGL.append(meta.altitude - arguments.msl)

images = pd.DataFrame(list(zip(allFiles, allAltitudes, correctedAGL)),
                      columns=[COLUMN_PATH, COLUMN_ALTITUDE, COLUMN_AGL])

# Correct the AGL to a discrete set -- there is much variance in what was recorded, so this is just a way
# to group observations together.

for index, row in images.iterrows():
    images.at[index, COLUMN_AGL] = closestAGL(list(arguments.agl), row[COLUMN_AGL])
    # print(f"{row[COLUMN_PATH]}: Raw AGL: {row[COLUMN_AGL]} Closest AGL: {takeClosest(list(arguments.agl), row[COLUMN_AGL])}")

directory = os.path.dirname(__file__)

for index, row in images.iterrows():
    # Create the directory
    filename = os.path.join(directory, arguments.output)
    destinationDir = os.path.join(filename, "AGL-" + str(images.at[index, COLUMN_AGL]) + constants.DASH + arguments.crop)
    if not os.path.exists(destinationDir):
        os.makedirs(destinationDir, exist_ok=True)

    # Copy the file there, refusing to overwrite a file
    destinationFile = os.path.join(destinationDir, os.path.split(images.at[index, COLUMN_PATH])[1])
    if not arguments.force and os.path.isfile(destinationFile):
        print(f"File exists: {destinationFile}. Use -f to force overwrite")
        sys.exit(-1)
    else:
        shutil.copy2(images.at[index, COLUMN_PATH], destinationDir)

sys.exit(0)
