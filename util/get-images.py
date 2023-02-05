#!/bin/env python
# G E T  I M A G E S
#
# Get the images from the jetsons

import argparse
import os

parser = argparse.ArgumentParser("Get images from local systems")

parser.add_argument('-s', '--system', action="append", required=True, nargs='*', help="System")
parser.add_argument('-o', '--output', action="store", required=False, default=".", help="Output directory")
parser.add_argument('-u', '--user', action="store", required=False, default="weeds", help="User name")
parser.add_argument('-d', '--directory', action="store", required=False, default="output", help="Input directory")
arguments = parser.parse_args()

for system in arguments.system:
    print("Retrieve from: {}".format(system))
    copyCommand = "scp -r {}@{}:{} {}".format(arguments.user, system, arguments.directory, arguments.output)
    os.system(copyCommand)


