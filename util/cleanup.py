import os
import sys
import argparse
import tqdm
from pathlib import Path

#
# Clean up the specified directory, eliminating JPGs if there is a DNG file with the same name
#

parser = argparse.ArgumentParser("Disk Cleanup")

parser.add_argument("-d", "--directory", action="store", required=True, help="Directory to clean")
parser.add_argument("-t", "--type", action="store", required=True, help="Extension of files to clean")
parser.add_argument("-duplicate", "--duplicate", action="store", required=False, help="Delete only if another file exists with the same name but with this extension")

arguments = parser.parse_args()

if not os.path.isdir(arguments.directory):
    print(f"Unable to access: {arguments.directory}")
    sys.exit(-1)

def cleanDirectory(directory: str, extension: str, extensionRequired: str):
    root = Path(directory)

    for dirPath, dirNames, fileNames in os.walk(directory):
        for dir in dirNames:
            print(f"Directory: {os.path.join(dirPath, dir)}")
            cleanDirectory(os.path.join(directory, dir), extension, extensionRequired)

        for file in fileNames:
            if extensionRequired is not None:
                if file.lower().endswith(extension.lower()):
                    requiredFile = Path(file).stem + "." + extensionRequired
                    if requiredFile.upper() in fileNames:
                        print(f"Remove: {file}")
                        os.remove(os.path.join(dirPath, file))
                    else:
                        pass
                        #print(f"Retain: {os.path.join(dirPath, file)}")

            #print(f"File: {os.path.join(dirPath, file)}")

cleanDirectory(arguments.directory, arguments.type, arguments.duplicate)

sys.exit(0)





