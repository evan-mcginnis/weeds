from PIL import Image

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import datetime
import argparse

import viplab_lib as vip
from mpl_toolkits import mplot3d

from ImageManipulation import ImageManipulation




class VegetationIndex:
    """
    Compute Vegetation Indices for a number of methods and create masks
    """
    def __init__(self):
        self.initialized = False
        self.images = []
        self.redBand = np.empty([0,0], dtype=np.uint8)
        self.greenBand = np.empty([0,0], dtype=np.uint8)
        self.blueBand = np.empty([0,0], dtype=np.uint8)

        self.redBandMasked = np.empty([0,0], dtype=np.uint8)
        self.greenBandMasked = np.empty([0,0], dtype=np.uint8)
        self.blueBandMasked = np.empty([0,0], dtype=np.uint8)

        self.mask = np.empty([0,0])

        self.HSV_COLOR_THRESHOLD = 20

        # Band positions for openCV (BGR)
        self.CV_BLUE = 0
        self.CV_GREEN = 1
        self.CV_RED = 2

        # Lambda values for the vegetation indices
        self.lambdaRed = 670 #nm
        self.lambdaGreen = 550 #nm
        self.lambdaBlue = 480 #nm

        self.ALG_NDI="ndi"
        self.ALG_TGI="tgi"
        self.ALG_EXG="exg"
        self.ALG_EXR="exr"
        self.ALG_EXGEXR="exgexr"
        self.ALG_CIVE="cive"
        self.ALG_NGRDI="ngrdi"
        self.ALG_VEG="veg"
        self.ALG_COM1="com1"
        self.ALG_MEXG="mexg"
        self.ALG_COM2="com2"
        self.ALG_RGD="rgd"

        self.algorithms = [self.ALG_NDI,
                           self.ALG_TGI,
                           self.ALG_NGRDI,
                           self.ALG_EXGEXR,
                           self.ALG_EXG,
                           self.ALG_EXR,
                           self.ALG_CIVE,
                           self.ALG_VEG,
                           self.ALG_COM1,
                           self.ALG_MEXG,
                           self.ALG_COM2,
                           self.ALG_TGI,
                           self.ALG_RGD]

        thresholds = {"NDI": 0,
                      "EXG": 200,
                      "EXR": 40,
                      "CIVE": 0,
                      "EXGEXR": 170,
                      "NGRDI": 0,
                      "VEG": 0,
                      "COM1": 300,
                      "MEXG": 10,
                      "COM2": 40,
                      "TGI": 0}

        self.computations = {self.ALG_NDI     : {"description": "Normalized Difference", "create": self.NDI, "threshold": thresholds["NDI"]},
                             self.ALG_EXG     : {"description": "Excess Green", "create": self.ExG, "threshold": thresholds["EXG"]},
                             self.ALG_EXR     : {"description": "Excess Red", "create": self.ExR, "threshold": thresholds["EXR"]},
                             self.ALG_CIVE    : {"description": "Color Index of Vegetation Extraction", "create": self.CIVE, "threshold": thresholds["CIVE"]},
                             self.ALG_EXGEXR  : {"description": "Excess Green - Excess Red", "create": self.ExGR, "threshold": thresholds["EXGEXR"]},
                             self.ALG_NGRDI   : {"description": "Normalized Green Red Difference", "create": self.NGRDI, "threshold": thresholds["NGRDI"]},
                             self.ALG_VEG     : {"description": "Vegetative Index", "create": self.VEG, "threshold": thresholds["VEG"]},
                             self.ALG_COM1    : {"description": "Combined Index 1", "create": self.COM1, "threshold": thresholds["COM1"]} ,
                             self.ALG_MEXG    : {"description": "Modified Excess Green", "create": self.MExG, "threshold": thresholds["MEXG"]},
                             self.ALG_COM2    : {"description": "Combined Index 2", "create": self.COM2, "threshold": thresholds["COM2"]},
                             self.ALG_TGI     : {"description": "TGI", "create": self.TGI, "threshold": thresholds["TGI"]},
                             self.ALG_RGD     : {"description": "Red Green Dots", "create": self.redGreenDots, "threshold": 0}}


    def Index(self, name: str)-> np.ndarray:
        """
        Compute the named index
        :param name:
        The name of the algorithm.  Use GetSupportedAlgorithms() for a list.
        :return:
        The index as an ndarray
        """
        algorithm = self.computations[name]
        return(algorithm["create"]())

    def GetImageStats(self, target: np.ndarray):
        nonZeroCells = np.count_nonzero(target > 0, keepdims=False)
        count = (target != 0.0).sum()
        return nonZeroCells

    def ShowStats(self, target: np.ndarray):
        nonZeroCells = np.count_nonzero(target)

        print("Nonzero cells:", nonZeroCells)

    def SaveIndexToFile(self,name: str, index: np.ndarray):
        cv.imwrite(name, index)

    def GetImage(self) -> np.ndarray:
        return self.img

    # The VIP routine is quite slow
    def GetMaskedImage(self) -> np.ndarray:
        maskedRGB = vip.Image_getBGR(self.redBandMasked, self.greenBandMasked, self.blueBandMasked, 8000)
        return maskedRGB

    # The opencv routine is fast, but seems to introduce some noise in the image
    # that must be cleaned up later.
    def GetImage(self):
        maskedRGB = cv.merge((self.blueBandMasked, self.greenBandMasked, self.redBandMasked))
        return maskedRGB

    def ReplaceNonZeroPixels(self, image: np.ndarray, value : float) -> np.ndarray:
        cartooned = np.where(image > 0, value, image)
        return cartooned

    #TODO: Move to utility
    def SaveImage(self, image: np.ndarray, name: str):
        data = Image.fromarray((image * 255).astype(np.uint8))
        data.save(name)

    #TODO: Move to utility -- nothing to do with this
    def SaveMaskedImage(self, name: str):
        """
        Save the RGB image to a file
        :type file: str
        :type RGBImage: np.ndarray
        """
        data = Image.fromarray((self.GetMaskedImage() * 255).astype(np.uint8))
        data.save(name)

    def SetImage(self, image):
        normalized = np.zeros_like(image)
        finalImage = cv.normalize(image,  normalized, 0, 255, cv.NORM_MINMAX)
        self.img = finalImage

        self.imgNP = np.array(self.img).astype(dtype=np.int16)
        self.redBand = self.imgNP[:, :, self.CV_RED]
        self.greenBand = self.imgNP[:, :, self.CV_GREEN]
        self.blueBand = self.imgNP[:, :, self.CV_BLUE]

    def Load(self, location: str):
        # TODO: Make this work for URLs
        # s = requests.Session()
        # s.mount('file://', FileAdapter())
        #
        # resp = s.get(location)
        # print(resp)

        self.img = cv.imread(location,cv.IMREAD_COLOR)

        self.imgNP = np.array(self.img).astype(dtype=np.int16)
        self.redBand = self.imgNP[:, :, self.CV_RED]
        self.greenBand = self.imgNP[:, :, self.CV_GREEN]
        self.blueBand = self.imgNP[:, :, self.CV_BLUE]

        self.images.append(self.img)

        # Display the image
        #plt.imshow(self.img)
        #plt.show()

        # This hangs, not sure what is going on here.
        #cv.imshow("Image", img)

    def Reload(self):
        """
        Reload the current image from the memory buffer, not from disk
        """
        self.imgNP = np.array(self.img)
        self.redBand = self.imgNP[:, :, self.CV_RED]
        self.greenBand = self.imgNP[:, :, self.CV_GREEN]
        self.blueBand = self.imgNP[:, :, self.CV_BLUE]


    def GetSupportedAlgorithms(self) -> []:
        return self.algorithms


    # def Stitch(self, location: str) -> Image:
    #
    #     return None

    def TGI(self) -> np.ndarray:
        #img = self.images[0]

        # Avoid conversion problems
        # shape = self.redBand.shape
        # TGI = np.empty(shape,dtype=np.int16)
        # TGI_Numerator = np.empty(shape,dtype=np.int16)
        # TGI_Denominator = np.empty(shape,dtype=np.int16)
        #
        # TGI_Numerator = (self.lambdaRed - self.lambdaBlue)\
        #                 * (self.redBand - self.greenBand) - (self.lambdaRed - self.lambdaGreen) * \
        #                 (self.redBand - self.blueBand)
        # TGI_Denominator = 2
        #
        #
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     TGI =np.true_divide(TGI_Numerator,TGI_Denominator)
        #     TGI[TGI == np.inf] = 0
        #     TGI = np.nan_to_num(TGI)

        alpha = (2 * (self.lambdaBlue - self.lambdaGreen)) / (self.lambdaBlue - self.lambdaRed)
        beta = (2 * (self.lambdaGreen - self.lambdaRed)) / (self.lambdaBlue - self.lambdaRed)
        alpha = 0.52
        beta = 0.59

        TGI = self.greenBand - alpha * self.redBand - beta * self.blueBand

        #MaskTGI = TGI > threshold #0.100


        return TGI

    def MaskFromIndex(self, index: np.ndarray, negate: bool, direction: int, threshold: () = None) -> np.ndarray:
        """
        Create a mask based on the index
        :param index:
        The vegetation index
        :param threshold:
        The threshold value to use. If not specified, Otsu's Binarization is used.
        :return:
        The mask as a numpy array with RGB channels
        """

        # TODO: This routine is a mess. Rewrite
        # If a threshold is not supplied, use Otsu
        if threshold == None:
            # Convert to a grayscale#
            greyScale = index.astype(np.uint8)
            #ret, thresholdedImage = cv.threshold(greyScale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            ret, thresholdedImage = cv.threshold(greyScale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
            threshold = ret
            threshold1 = ret
            threshold2 = -9999
        else:
            # Thresholds are (up, down)
            threshold1 = threshold[0]
            threshold2 = threshold[1]

        # If the direction is 1, interpret the two thresholds as up, down
        # To make this go back to the original implementation, set the second threshold to -9999

        if direction > 0:
            thresholdedIndex = index > threshold1
            if threshold2 != -9999:
                thresholdIndex2 = index < threshold2
                finalMask = np.logical_or(thresholdedIndex, thresholdIndex2)
            else:
                finalMask = thresholdedIndex
            negated = finalMask
        else:
            thresholdedIndex = index < threshold1
            #negated = np.logical_not(thresholdedIndex)
            negated = thresholdedIndex
            finalMask = negated

        if negate:
            finalMask = np.logical_not(thresholdedIndex)
            #negated = finalMask

        # Touch up the mask with some floodfill
        filledMask = finalMask.copy().astype(np.uint8)
        cv.floodFill(filledMask, None, (0,0),255)
        filledMaskInverted = cv.bitwise_not(filledMask)
        # finalMask = cv.bitwise_not(filledMaskInverted)
        # #plt.imshow(finalMask, cmap='gray', vmin=0, vmax=1)
        # #plt.show()
        # As we apply this mask as multiplying, convert 255s to 1
        filledMaskInverted = np.where(filledMaskInverted > 0, 1, filledMaskInverted)

        self.mask = filledMaskInverted
        finalMask = filledMaskInverted

        negated = filledMaskInverted
        #negated = np.logical_not(filledMaskInverted)
        # End floodfill touch up


        self.mask = negated


        # FYI: the mask value returned here is not used
        return finalMask, threshold

    @property
    def imageMask(self):
        return self.mask

    def applyMask(self):
        self.redBandMasked = self.redBand * self.mask # DataRED * negated
        self.greenBandMasked = self.greenBand * self.mask  # DataGREEN * negated
        self.blueBandMasked = self.blueBand * self.mask  # DataBLUE * negated

        return

    def GetMask(self):
        return np.dstack([self.redBandMasked, self.greenBandMasked, self.blueBandMasked])

    def ShowImage(self, title : str, index : np.ndarray):

        plt.title(title)
        #plt.imshow(index, cmap='gray', vmin=0, vmax=255)
        plt.imshow(index, cmap='gray', vmin=-1, vmax=1)
        plt.show()

    def ExR(self) -> np.ndarray:
        """
        The Excessive Red Index
        1.3*R - G
        :return:
        The index as a numpy array
        """
        excessRed = 1.3*self.redBand - self.greenBand

        return excessRed

    def ExG(self) -> np.ndarray:
        """
        The Excessive Green Index
        2*G - R - B
        :return:
        The index as a numpy array
        """
        # 2g-r-b

        excessGreen = 2*self.greenBand - self.redBand - self.blueBand

        return excessGreen

    def ExGR(self) -> np.ndarray:
        """
        The Excess Green - Excessive Red Index
        :return:
        The index as a numpy array
        """
        exgr = np.empty_like(self.redBand)

        exgr = self.ExG() - self.ExR()

        return exgr

    def CIVE(self) -> np.ndarray:
        """
        The Color Index of Vegetation Extraction
        :return:
        The index as a numpy array
        """
        cive = 0.441*self.redBand - 0.811*self.greenBand + 0.385*self.blueBand + 18.78745

        return cive

    def NGRDI(self) -> np.ndarray:
        """
        The Normalized Green-Red Difference Index
        :return:
        The index as a numpy array
        """
        ngrdi = np.zeros_like(self.img)
        ngrdi_numerator = self.greenBand - self.redBand
        ngrdi_denominator = self.greenBand + self.redBand

        #ngrdi = (self.greenBand - self.redBand) / (self.greenBand + self.redBand)

        with np.errstate(divide='ignore', invalid='ignore'):
            ngrdi =np.true_divide(ngrdi_numerator,ngrdi_denominator)
            ngrdi[ngrdi == np.inf] = 0
            ngrdi = np.nan_to_num(ngrdi)

        return ngrdi

    def VEG(self) -> np.ndarray:

        #veg = self.greenBand / self.redBand ** 0.667 * self.blueBand ** (1-0.667)

        veg_numerator = self.greenBand
        veg_denominator = self.redBand ** 0.667 * self.blueBand ** (1-0.667)

        with np.errstate(divide='ignore', invalid='ignore'):
            veg =np.true_divide(veg_numerator,veg_denominator)
            veg[veg == np.inf] = 0
            veg = np.nan_to_num(veg)

        return veg

    def COM1(self) -> np.ndarray:

        com1 = self.ExG() + self.CIVE() + self.ExGR() + self.VEG()

        return com1

    def MExG(self) -> np.ndarray:

        mexg = 1.262 * self.greenBand - 0.884 * self.redBand - 0.311 * self.blueBand

        return mexg

    def COM2(self) -> np.ndarray:

        com2 = (0.36 * self.ExG()) + (0.47 * self.CIVE()) + (0.17 * self.VEG())

        return com2

    def NDI(self) -> np.ndarray:

        #img = self.images[0]


        # Avoid conversion problems and set the dtype explicitly
        # The problem here is that the bands are all 8 bit, so the intermediate values
        # are incorrect when things exceed 8 bits.  As always, probably a more elegant way of doing things
        # here
        shape = self.redBand.shape

        # NDI = np.empty(shape,dtype=np.int16)
        # NDI_Numerator = np.empty(shape, dtype=np.int16)
        # NDI_Denominator = np.empty(shape, dtype=np.int16)
        # NDI = (self.greenBand.astype(dtype=np.int16)
        #        - self.redBand.astype(dtype=np.int16)) \
        #       / (self.greenBand.astype(dtype=np.int16)
        #          + self.redBand.astype(dtype=np.int16))

        NDI_Numerator=(self.greenBand.astype(dtype=np.int16)
                       - self.redBand.astype(dtype=np.int16)) #DataGREEN-DataRED
        NDI_Denominator=(self.greenBand.astype(dtype=np.int16)
                         + self.redBand.astype(dtype=np.int16)) #DataGREEN+DataRED

        with np.errstate(divide='ignore', invalid='ignore'):
            NDI =np.true_divide(NDI_Numerator,NDI_Denominator) + 1
            NDI= NDI * 128
            NDI[NDI == np.inf] = 0
            NDI = np.nan_to_num(NDI)

        #MaskNDI = NDI > 0.100

        # The mask we computed above is the opposite of what is needed
        # We want the green bits to show through.
        # TODO: Make this a 3d mask.  Seems like that should be possible
        #negated = np.logical_not(MaskNDI)
        #DataREDMasked = self.redBand * negated #DataRED * negated
        #DataGREENMasked = self.greenBand * negated #DataGREEN * negated
        #DataBLUEMasked = self.blueBand * negated #DataBLUE * negated

        #RGBImage=vip.Image_getRGB(DataREDMasked,DataGREENMasked,DataBLUEMasked,8000)

        return NDI
        #result = cv.bitwise_and(RGBImage,MaskNDI)

    #
    # Not really an index for things found in nature, but intended for the colored dots
    # technique where green is for crop and red is for weeds.
    #
    def redGreenDots(self):
        # Convert the image to HSV
        manipulation = ImageManipulation(self.img)
        hsv = manipulation.toHSV()

        # Pure colors
        green = np.uint8([[[0,255,0 ]]])
        red = np.uint8([[[0,0,255]]])

        # Get the HSV for the color green
        hsv_green = cv.cvtColor(green,cv.COLOR_BGR2HSV)
        # Compute the lower and upper bound of what we consider green
        lowerGreen = np.array([int(hsv_green[0,0,0]) - self.HSV_COLOR_THRESHOLD,100,100])
        upperGreen = np.array([int(hsv_green[0,0,0]) + self.HSV_COLOR_THRESHOLD, 255,255])

        hsv_red = cv.cvtColor(red,cv.COLOR_BGR2HSV)
        # Compute the lower and upper bound of what we consider green
        lowerRed = np.array([int(hsv_red[0,0,0]) - 5,100,100])
        upperRed = np.array([int(hsv_red[0,0,0]) + 5, 255,255])
        # lower boundary RED color range values; Hue (0 - 10)

        # Another red attempt
        # This looks odd, but eliminates the hue of red in the floor
        lower1 = np.array([0, 100, 20])
        #upper1 = np.array([10, 255, 255])
        upper1 = np.array([0, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])

        lower_mask = cv.inRange(hsv, lower1, upper1)
        upper_mask = cv.inRange(hsv, lower2, upper2)

        maskRed = lower_mask + upper_mask

        #result = cv2.bitwise_and(result, result, mask=full_mask)




        maskGreen = cv.inRange(hsv, lowerGreen, upperGreen)
        #maskRed = cv.inRange(hsv, lowerRed, upperRed)

        # The final mask where either of the two colors are present
        mask = cv.bitwise_or(maskGreen, maskRed)

        # Crap, this does not work well on the test data -- the floor has too much red in it.
        # So the results have way too much in them to be useful.
        # I'll try this with blue, but in the meantime, just use the green mask
        #mask = maskGreen

        # Bitwise-AND mask and original image
        #res = cv.bitwise_and(self.img,self.img, mask= mask)
        return mask

if __name__ == "__main__":
    utility = VegetationIndex()
    parser = argparse.ArgumentParser("Show various vegetation indices")

    parser.add_argument('-i', '--input', action="store", required=True, type=str, help="Image to process")
    parser.add_argument('-o', '--output', action="store", help="Output directory for processed images")
    parser.add_argument('-p', '--plot', action="store_true", default=False, help="Plot the index")
    parser.add_argument('-s', '--show', action="store_true", default=False, help="Show intermediate images")
    parser.add_argument('-t', '--threshold', action="store_true", default=False, help="Calculate thresholds")

    arguments = parser.parse_args()


    utility.Load(arguments.input)

    # Load the target -- this must be done first
    #utility.Load('./20190118_124612.jpg')
    #utility.Load('./overhead.jpg')
    #utility.Load('./LettuceBed.jpg')
    #utility.Load('./drone-pictures/DJI_0074.jpg')
    #utility.Load("./Oasis_Area.png")
    #utility.Load("./test-images/purslane.png")
    #Not required.  Just show our starting point
    #utility.ShowImage("Source", utility.GetImage())
    from mpl_toolkits import mplot3d
    import numpy as np
    import matplotlib.pyplot as plt


    # The thresholds are probably dependant on the image selected, so
    # we can't use fixed values in practice.
    # For the overhead:
    threshOasis = {"NDI"   : 0,
                   "EXG"   : 200,
                   "EXR"   : 75,
                   "CIVE"  : 20,
                   "EXGEXR": 100,
                   "NGRDI" : 200,
                   "VEG"   : 0,
                   "COM1"  : 400,
                   "MEXG"  : 0,
                   "COM2"  : 80,
                   "TGI"   : 0}

    # Original thresholds
    threshOverhead = {"NDI"   : 125,
                      "EXG"   : 100,
                      "EXR"   : 20,
                      "CIVE"  : -10,
                      "EXGEXR": 60,
                      "NGRDI" : 0,
                      "VEG"   : 1,
                      "COM1"  : 5, # Originally 25
                      "MEXG"  : 10, # Orignally 40
                      "COM2"  : 15,
                      "TGI"   : 0}

    # Thresholds where we need two sides (like red stems we want to pick up)
    threshOverhead = {"NDI"   : (130,0),
                      "EXG"   : (50,0),
                      "EXR"   : (20,0),
                      "CIVE"  : (40,5),
                      "EXGEXR": (50,-9999),
                      "NGRDI" : (0,-9999),
                      "VEG"   : (2,-9999),
                      "COM1"  : (5,-9999), # Originally 25
                      "MEXG"  : (30,-9999), # Originally 40
                      "COM2"  : (15,-9999),
                      "TGI"   : (20,-9999)}
    threholds = threshOverhead

    # All of the indices
    indices = {"Normalized Difference": {"short": "NDI", "create": utility.NDI, "negate": True, "threshold": threholds["NDI"], "direction": 1},
               "Excess Green": {"short": "ExG", "create": utility.ExG, "negate": True, "threshold": threholds["EXG"], "direction": 1},
               "Excess Red": {"short": "ExR", "create": utility.ExR, "negate": False, "threshold": threholds["EXR"], "direction": -1},
               "Color Index of Vegetation Extraction": {"short": "CIVE", "create": utility.CIVE, "negate": True, "threshold": threholds["CIVE"], "direction": 1},
               "Excess Green - Excess Red": {"short": "ExGR", "create": utility.ExGR, "negate": True, "threshold": threholds["EXGEXR"], "direction": 1},
               "Normalized Green Red Difference": {"short": "NGRDI", "create": utility.NGRDI, "negate": True, "threshold": threholds["NGRDI"], "direction": 1},
               "Vegetative Index": {"short": "VEG", "create": utility.VEG, "negate": True, "threshold": threholds["VEG"], "direction": 1},
               #"Combined Index 1": {"short": "COM1", "create": utility.COM1, "negate": False, "threshold": threholds["COM1"], "direction": -1} ,
               "Combined Index 1": {"short": "COM1", "create": utility.COM1, "negate": False, "threshold": None, "direction": 1} ,
               "Modified Excess Green": {"short": "MexG", "create": utility.MExG, "negate": True, "threshold": threholds["MEXG"], "direction": 1},
               #"Combined Index 2": {"short": "COM2", "create": utility.COM2, "negate": True, "threshold": threholds["COM2"], "direction": -1},
               "Combined Index 2": {"short": "COM2", "create": utility.COM2, "negate": False, "threshold": None, "direction": 1},
               "Triangulation Greenness Index": {"short": "TGI", "create": utility.TGI, "negate": False, "threshold": threholds["TGI"], "direction": 1}}

    # Debug the implementations:
    indices = {
        "Normalized Difference": {"short": "NDI", "create": utility.NDI, "negate": True, "threshold": threholds["NDI"],
                                  "direction": 1},
    }


    # Step through the indices and show the result on the target image
    for indexName, indexData in indices.items():
        creationMethod = indexData["create"]

        showImages = arguments.show
        plot = arguments.plot

        startTime = datetime.datetime.now()
        vegIndex = creationMethod()
        finishTime = datetime.datetime.now()
        computeTime = finishTime - startTime
        indexData["time"] = computeTime.microseconds
        print("Index: " + indexName + " " + str(computeTime))

        threshold = indexData["threshold"]
        #threshold=None
        direction = indexData["direction"]
        negate = indexData["negate"]

        if showImages:
            utility.ShowImage("Index: " + indexName + " Threshold: " + str(threshold), vegIndex)

        indexData["index"] = vegIndex

        # if plot:
        #     transectRow = vegIndex[100,:]
        #     plt.figure()
        #     plt.plot(transectRow)
        #     plt.title(indexName)
        #     plt.show()

        # Force otsu
        if arguments.threshold:
            threshold = None

        mask, thresholdUsed = utility.MaskFromIndex(vegIndex, negate, direction, threshold)
        # Determing threshold from image
        #utility.MaskFromIndex(vegIndex)
        utility.applyMask()
        image = utility.GetMaskedImage()

        indexData["masked"] = image

        indexStats = utility.GetImageStats(image)
        # For now, the only stat we care about is the vegetative pixel count
        indexData["vegetativePixels"] = indexStats


        if showImages:
            utility.ShowImage(indexName + " mask applied to source with threshold: " + str(thresholdUsed), image)

        if plot:
            Ylen,Xlen = vegIndex.shape
            x = np.arange(0, Xlen, 1)
            y = np.arange(0, Ylen, 1)
            x, y = np.meshgrid(x, y)
            fig = plt.figure(figsize=(10,10))
            axes = fig.gca(projection ='3d')
            #axes.plot_surface(x, y, vegIndex)
            plt.title(indexName)
            axes.scatter(x, y, vegIndex, c=vegIndex, cmap='BrBG', s=0.25)
            plt.show()

        #print(indexName + ":" + str(indexData["vegetativePixels"]))
        # Evaluate how much information was discarded from image
        #utility.ShowStats(image)

    # Commment this out for now.
    #
    # # Side by side plots of everything
    # fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(8,6))
    # xpos = 0
    # ypos = 0
    # for indexName, indexData in indices.items():
    #     im = ax[xpos, ypos].imshow(indexData["masked"])
    #     ax[xpos, ypos].set_title(indexData["short"])
    #     if ypos == 3:
    #         ypos = 0
    #         xpos = xpos + 1
    #     else:
    #         ypos = ypos + 1
    # # As we have an open spot
    # ax[xpos, ypos].imshow(utility.GetImage())
    # ax[xpos, ypos].set_title("None")
    # plt.show()
    #
    # fig = plt.figure(figsize=(5,5))
    # plt.style.use('ggplot')
    #
    # xs = np.arange(len(indices))
    # pixelCounts = []
    # labels = []
    # pixelCount = utility.GetImageStats(utility.GetImage())
    # indexName = "None"
    # pixelCounts.append(pixelCount)
    # labels.append(indexName)
    # for indexName, indexData in indices.items():
    #     pixelCount = indexData["vegetativePixels"]
    #     pixelCounts.append(pixelCount)
    #     indexName = indexData["short"]
    #     labels.append(indexName)
    #
    # xpos = [i for i, _ in enumerate(labels)]
    # plt.barh(labels, pixelCounts)
    # plt.ylabel("Algorithms")
    # plt.xlabel("Vegetated Pixels")
    # plt.title("Color Index Algorithms Pixel Counts")
    # plt.show()
    #
    # times = []
    # labels = []
    #
    # for indexName, indexData in indices.items():
    #     timeTaken = indexData["time"]
    #     times.append(timeTaken)
    #     indexName = indexData["short"]
    #     labels.append(indexName)
    #
    # plt.barh(labels, times)
    # plt.ylabel("Algorithms")
    # plt.xlabel("Time Taken in microseconds")
    # plt.title("Color Index Algorithms Compute Times")
    # plt.show()
    #
    #
    # # E D G E  D E T E C T I O N
    # # Rather than wrestle with this, just write out the image for now.
    # image = utility.ReplaceNonZeroPixels(utility.GetMaskedImage(), 0.575)
    # utility.ShowImage("Cartooned", image)
    #
    # utility.SaveImage(image, "cartooned.jpg")
    # #utility.SaveMaskedImage("mask-applied.jpg")
    # #img_float32 = np.float32(utility.GetMaskedImage())
    # img = cv.imread("cartooned.jpg")
    # gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # edges = cv.Canny(gray, 20, 30)
    # plt.subplot(121),plt.imshow(utility.GetMaskedImage(),cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    # plt.show()
    #
    # #Converting to RGB
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # gray = cv.medianBlur(gray, 5)
    # edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)
    # utility.ShowImage("Edges", edges)
    # pixelCount = utility.GetImageStats(edges)
    # print("Pixel Count for edges:", pixelCount)