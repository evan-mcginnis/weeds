import sys

import numpy as np
import laspy
import argparse
import sys

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import logging
import logging.config
import yaml
import os

from VegetationIndex import VegetationIndex
from ImageManipulation import ImageManipulation
from Logger import Logger
from Performance import Performance
import constants

ALG_ALL = "all"
# Used in command line processing so we can accept thresholds that are tuples
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

# This is here so we can extract the supported algorithms

veg = VegetationIndex()

parser = argparse.ArgumentParser("Process Point Cloud with Vegetation Index")

parser.add_argument('-i', '--input', action="store", required=True, help="Point Cloud (.Las) to process")
parser.add_argument('-o', '--output', action="store", required=True, help="Output directory for processed images")
parser.add_argument("-a", '--algorithm', action="store", help="Vegetation Index algorithm",
                    choices=veg.GetSupportedAlgorithms().append(ALG_ALL),
                    default="mexg")
parser.add_argument("-P", "--performance", action="store", type=str, default="performance.csv", help="Name of performance file")
parser.add_argument("-p", '--plot', action="store_true", help="Show 3D plot of index", default=False)
parser.add_argument("-n", "--nonegate", action="store_true", default=False, help="Negate image mask")
parser.add_argument("-t", "--threshold", action="store", type=tuple_type, default="(0,0)", help="Threshold tuple (x,y)")
parser.add_argument("-m", "--mask", action="store_true", default=False, help="Mask only -- no processing")
parser.add_argument("-d", "--direction", action="store", default=1, type=int, help="Direction 1 or -1")

arguments = parser.parse_args()

ALG_NDI="ndi"
ALG_TGI="tgi"
ALG_EXG="exg"
ALG_EXR="exr"
ALG_EXGEXR="exgexr"
ALG_CIVE="cive"
ALG_NGRDI="ngrdi"
ALG_VEG="veg"
ALG_COM1="com1"
ALG_MEXG="mexg"
ALG_COM2="com2"
ALG_RGD="rgd"

thresholds = {ALG_NDI: (130,0),
              ALG_EXG: (18, 0),
              ALG_EXR: (35, 5),
              ALG_CIVE: (30,0),
              ALG_EXGEXR: (0,0),
              ALG_NGRDI: (10,0),
              ALG_VEG: (1.25,0),
              ALG_COM1: (-100,0),
              ALG_MEXG: (6000,-9999),
              ALG_COM2: (15,0),
              ALG_TGI: (300,-15)} # Revisit

indices = {ALG_NDI: {"short": ALG_NDI, "create": veg.NDI, "negate": True, "threshold": thresholds[ALG_NDI], "direction": 1},
           ALG_EXG: {"short": ALG_EXG, "create": veg.ExG, "negate": True, "threshold": thresholds[ALG_EXG], "direction": 1},
           ALG_EXR: {"short": ALG_EXR, "create": veg.ExR, "negate": False, "threshold": thresholds[ALG_EXR], "direction": 1},
           ALG_CIVE: {"short": ALG_CIVE, "create": veg.CIVE, "negate": False, "threshold": thresholds[ALG_CIVE], "direction": 1},
           ALG_EXGEXR: {"short": ALG_EXGEXR, "create": veg.ExGR, "negate": True, "threshold": thresholds[ALG_EXGEXR], "direction": 1},
           ALG_NGRDI: {"short": ALG_NGRDI, "create": veg.NGRDI, "negate": True, "threshold": thresholds[ALG_NGRDI], "direction": 1},
           ALG_VEG: {"short": ALG_VEG, "create": veg.VEG, "negate": True, "threshold": thresholds[ALG_VEG], "direction": 1},
           ALG_COM1: {"short": ALG_COM1, "create": veg.COM1, "negate": False, "threshold": None, "direction": 1} ,
           ALG_MEXG: {"short": ALG_MEXG, "create": veg.MExG, "negate": True, "threshold": thresholds[ALG_MEXG], "direction": 1},
           ALG_COM2: {"short": ALG_COM2, "create": veg.COM2, "negate": False, "threshold": thresholds[ALG_COM2], "direction": 1},
           ALG_TGI: {"short": ALG_TGI, "create": veg.TGI, "negate": True, "threshold": thresholds[ALG_TGI], "direction": 1}}

_algorithms = [ALG_NDI,
              ALG_EXG,
              ALG_EXR,
              ALG_CIVE,
              ALG_EXGEXR,
              ALG_NGRDI,
              ALG_VEG,
              ALG_COM1,
              ALG_MEXG,
              ALG_COM2,
              ALG_TGI]

#_algorithms = [ALG_TGI]

def startupPerformance() -> Performance:
    """
    Start up the performance subsystem.
    :return:
    """
    performance = Performance(arguments.performance)
    (performanceOK, performanceDiagnostics) = performance.initialize()
    if not performanceOK:
        print(performanceDiagnostics)
        sys.exit(1)
    return performance

#
# L O G G E R
#

def startupLogger(outputDirectory: str) -> Logger:
    """
    Initializes two logging systems: the image logger and python centric logging.
    :param outputDirectory: The output directory for the images
    :return: The image logger instance
    """

    # The command line argument contains the name of the YAML configuration file.

    # Confirm the YAML file exists
    if not os.path.isfile(arguments.logging):
        print("Unable to access logging configuration file {}".format(arguments.logging))
        sys.exit(1)

    # Initialize logging
    with open(arguments.logging, "rt") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        #logger = logging.getLogger(__name__)

    logger = Logger()
    if not logger.connect(outputDirectory):
        print("Unable to connect to logging. ./output does not exist.")
        sys.exit(1)
    return logger

def plot3D(index, title):
    yLen,xLen = index.shape
    x = np.arange(0, xLen, 1)
    y = np.arange(0, yLen, 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,10))
    axes = fig.gca(projection ='3d')
    plt.title(title)
    axes.scatter(x, y, index, c=index, cmap='BrBG', s=0.25)
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Index Value')
    plt.show()
    cv.waitKey()

def plotTransect(image, title):
    yLen, xLen = image.shape
    x = np.arange(0, yLen, 1)
    #plt.style.use('ggplot')
    plt.figure(figsize=(20,10))
    plt.title("Transect across river")
    plt.xlabel('Distance')
    plt.ylabel('Pixel Value')
    plt.plot(x, image[:,0], color='red', label="Red")
    plt.plot(x, image[:,1], color='green', label="Green")
    plt.plot(x, image[:,2], color='blue', label="Blue")
    plt.legend()
    plt.show()
    cv.waitKey()



log = logging.getLogger(__name__)
performance = startupPerformance()
imageNumber = 0
mmPerPixel = 0
try:
    performance.start()
    # Read the point cloud
    las = laspy.read(arguments.input)

    # Construct the image we will use.  This really isn't an image, of course
    rPoints = las.red
    gPoints = las.green
    bPoints = las.blue
    rawImage = np.dstack([bPoints,gPoints, rPoints])

    veg.SetImageBands(las.red, las.green, las.blue)
    index = veg.Index(arguments.algorithm)
    mask = veg.makeMask(index,
                        indices.get(arguments.algorithm).get("negate"),
                        indices.get(arguments.algorithm).get("direction"),
                        indices.get(arguments.algorithm).get("threshold"))
    #mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, 1, arguments.threshold)
    veg.applyLargeMask(True)
    #image = veg.GetImage()
    las.red = veg.redMasked
    las.green = veg.greenMasked
    #las.green = np.full_like(image[0,:,1],0)
    las.blue = veg.blueMasked
    las.write("manipulated.las")

    sys.exit(0)

    if arguments.algorithm == ALG_ALL:
        veg.SetImage(rawImage)
        results = [0] * (len(thresholds) + 1)
        #algorithms = [0] * (len(thresholds) + 1)
        algorithms = []
        i = 0
        # Debug line
        #for algorithm in _algorithms:
        for algorithm in veg.GetSupportedAlgorithms():
            # This is a workaround for what is probably a bug (or a something I don't understand)
            # A call to GetSupportedAlgorithms() returns the "all" appended to it above
            # This doesn't seem quite right...
            if algorithm != ALG_ALL and algorithm != ALG_RGD:
                print("Algorithm: {}".format(algorithm))
                veg = VegetationIndex()
                veg.SetImage(rawImage)
                index = veg.Index(algorithm)
                mask, threshold = veg.MaskFromIndex(index,
                                                    indices.get(algorithm).get("negate"),
                                                    indices.get(algorithm).get("direction"),
                                                    indices.get(algorithm).get("threshold"))
                #mask, threshold = veg.MaskFromIndex(index, not arguments.nonegate, 1, arguments.threshold)
                veg.applyMask()
                image = veg.GetImage()
                # Uncomment this line to show
                #ImageManipulation.show(algorithm, image)
                #ImageManipulation.save(image, algorithm + ".jpg")
                cv.imwrite(arguments.output + "/" + "processed-with-" + algorithm + ".jpg", image)
                veg.ShowStats(image)
                # The results hold the area calculation
                results[i] = veg.GetImageStats(image) * arguments.gsd
                i += 1
                algorithms.append(algorithm)

        xs = np.arange(0, len(results), 1)
        #x_pos = [i for i, _ in enumerate(algorithms)]
        plt.figure(figsize=(10,5))
        plt.title("Vegetated Area")
        plt.xlabel('Algorithm')
        plt.ylabel('Vegetated Area (CM)')
        plt.bar(algorithms, results)
        plt.savefig(arguments.output + "/" + "bar-counts.jpg")
        plt.show()

        sys.exit(0)


except IOError as e:
    print("There was a problem communicating with the camera")
    print(e)
    sys.exit(1)

# las = laspy.read('./final_group0_densified_point_cloud.las')
# print(las)
# print('Points from data:', len(las.points))
# points = las.points
# format = las.point_format
# print('Format:', format.id)
# rPoints = las.red
# gPoints = las.green
# bPoints = las.blue
# las.red = np.where(las.red > -1,0,0)
# #for point in rPoints:
# #    point = -1
# las.red = rPoints
# for point in las.green:
#     point = -1
# las.green = gPoints
# for point in bPoints:
#     point = -1
# las.blue = bPoints
# las.write("manipulated.las")
# sys.exit(-1)



