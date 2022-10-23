import argparse
import sys
import time

import numpy as np
import scipy.ndimage

parser = argparse.ArgumentParser("Resample Depth Data")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-o", "--output", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-a", "--algorithm", action="store", required=False, default="nearest-neighbor", help="Algorithm: nearest-neighbor, bilinear, biquadratic, or bicubic")
arguments = parser.parse_args()

depth = None

try:
    depth = np.load(arguments.input)
except FileNotFoundError:
    print("Unable to find the file: {}".format(arguments.input))
    sys.exit(-1)

if depth is not None:
    print("Max X is: {}".format(depth.max()))
    print("Shape is: {}".format(np.shape(depth)))

# Hardcode this for now -- depth is 1280x720, and we want 1920x1080

z = (1920/1280, 1080/720)

if arguments.algorithm == "ALL":
    methods=['nearest-neighbor', 'bilinear', 'biquadratic', 'bicubic']
    for o in range(4):
        starttime = time.time()
        transformed = scipy.ndimage.zoom(depth, z, order=o)
        stoptime = time.time()
        print("Transformed via {} in {} ms".format(methods[o], stoptime - starttime))
        np.save(methods[o], transformed)
else:
    starttime = time.time()
    transformed = scipy.ndimage.zoom(depth, z, order=arguments.algorithm)
    stoptime = time.time()
    print("Transformed via {} in {} ms".format(arguments.algorithm, stoptime - starttime))
    np.save(arguments.output, transformed)