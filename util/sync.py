#!/bin/env python
# S Y N C H  I M A G E S
#

import argparse
import os
import sys

def syncSystem(user: str, system: str):
    synchCommand = "c:\\cygwin64\\bin\\rsync.exe -avz -e ssh -r {}@{}:output .".format(user, system)
    print("Synch using {}\n".format(synchCommand))
    os.system(synchCommand)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get images from local systems")

    parser.add_argument('-s', '--systems', required=True, nargs='+', help="Systems")
    parser.add_argument('-o', '--output', action="store", required=False, default=".", help="Output directory")
    parser.add_argument('-u', '--user', action="store", required=False, default="weeds", help="User name")
    parser.add_argument('-d', '--directory', action="store", required=False, default="/tmp", help="Input directory")
    parser.add_argument('-l', '--log', action="store", required=False, default="/tmp/sync.log", help="Stdout file")
    parser.add_argument('-e', '--error', action="store", required=False, default="/tmp/sync.err", help="Stderr file")
    arguments = parser.parse_args()

    os.chdir(arguments.directory)

    sys.stdout = open(arguments.log, "w")
    sys.stderr = open(arguments.error, "w")

    for system in arguments.systems:
        synchCommand = "c:\\cygwin64\\bin\\rsync.exe -avz -e ssh -r {}@{}:output .".format(arguments.user, system)
        print("Synch using {}\n".format(synchCommand))
        syncSystem(arguments.user, system)
    print("Finished with synchronization\n")
    sys.exit(0)

