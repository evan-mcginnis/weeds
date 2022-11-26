import argparse
import sys
import time

import numpy as np
import scipy.ndimage
from PIL import Image

parser = argparse.ArgumentParser("Resample Depth Data")
parser.add_argument("-i", "--input", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-o", "--output", action="store", required=True, help="Depth data file in npy format")
parser.add_argument("-a", "--algorithm", action="store", required=False, default="nearest-neighbor", help="Algorithm: nearest-neighbor, bilinear, biquadratic, bicubic, or ALL")
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

def resized_dims(width, height, ratio):
   t = (height + ratio * width) / (ratio**2 + 1)
   new_width = int(t*ratio + 0.5)
   new_height = int(t + 0.5)
   return new_width, new_height
def cropped_dims(width, height, ratio):
    if width > height*ratio:
        width = int(height*ratio + 0.5)
    else:
        height = int(width/ratio + 0.5)
    return (width, height)

# Determine the shape needed to match the image ratio from the Basler 2500

croppedShape = cropped_dims(1280, 720, 4/3)
print("Cropped dimensions of (1280x720) to 4:3: {}".format(croppedShape))
#print("Resize of (1280x720): {}".format(resized_dims(1280, 720, 4/3)))

# Pull out an image that is the same aspect ratio

# The offset is where we want to start the extraction so the depth data overlaps with the RGB

offset = 200
resized = depth[0:croppedShape[1],0+offset:croppedShape[0]+offset]
#np.save("resized-4-3.npy", resized)

# Pull out the bits that correspond to the aspect ratio we want:

# Hardcode this for now -- depth is 1280x720, and we want 1920x1080
#z = (1920/1280, 1080/720)
# Hardcode this for now -- depth is 960x720, and we want 2590x1942
z = (2590/960, 1942/720)

if arguments.algorithm == "ALL":
    methods=['nearest-neighbor', 'bilinear', 'biquadratic', 'bicubic']
    for o in range(4):
        starttime = time.time()
        transformed = scipy.ndimage.zoom(resized, z, order=o)
        stoptime = time.time()
        print("Transformed via {} in {} ms".format(methods[o], stoptime - starttime))
        np.save(methods[o], transformed)
else:
    starttime = time.time()
    transformed = scipy.ndimage.zoom(resized, z, order=0)
    stoptime = time.time()
    print("Transformed via {} in {} ms".format(arguments.algorithm, stoptime - starttime))
    np.save(arguments.output, transformed)