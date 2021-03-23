

#from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import viplab_lib as vip

#import validators
#import requests
#from requests_file import FileAdapter


class ImageUtilities:

    def __init__(self):
        self.initialized = False
        self.images = []
        self.redBand = np.empty([0,0])
        self.greenBand = np.empty([0,0])
        self.blueBand = np.empty([0,0])

        # Band positions for openCV (BGR)
        self.CV_BLUE = 0
        self.CV_GREEN = 1
        self.CV_RED = 2

        # Lambda values for the vegetation indices
        self.lambdaRed = 670 #nm
        self.lambdaGreen = 550 #nm
        self.lambdaBlue = 480 #nm

    def Load(self, location: str):
        # TODO: Make this work for URLs
        # s = requests.Session()
        # s.mount('file://', FileAdapter())
        #
        # resp = s.get(location)
        # print(resp)

        self.img = cv.imread(location,cv.IMREAD_COLOR)

        self.imgNP = np.array(self.img)
        self.redBand = self.imgNP[:, :, self.CV_RED]
        self.greenBand = self.imgNP[:, :, self.CV_GREEN]
        self.blueBand = self.imgNP[:, :, self.CV_BLUE]

        self.images.append(self.img)

        # Display the image
        plt.imshow(self.img)
        plt.show()

        # This hangs, not sure what is going on here.
        #cv.imshow("Image", img)

    # def Stitch(self, location: str) -> Image:
    #
    #     return None

    def TGIMask(self, threshold: int) -> np.ndarray:
        img = self.images[0]

        TGI_Numerator = (self.lambdaRed - self.lambdaBlue)*(self.redBand - self.greenBand) - (self.lambdaRed - self.lambdaGreen) * \
                        (self.redBand - self.blueBand)
        TGI_Denominator = 2
        with np.errstate(divide='ignore', invalid='ignore'):
            TGI =np.true_divide(TGI_Numerator,TGI_Denominator)
            TGI[TGI == np.inf] = 0
            TGI = np.nan_to_num(TGI)

        MaskTGI = TGI > threshold #0.100

        self.mask = MaskTGI

        return MaskTGI

    def applyMask(self):
        #negated = np.logical_not(self.mask)
        negated = self.mask
        self.redBandMasked = self.redBand * negated  # DataRED * negated
        self.greenBandMasked = self.greenBand * negated  # DataGREEN * negated
        self.blueBandMasked = self.blueBand * negated  # DataBLUE * negated

    def getImage(self):
        return vip.Image_getRGB(self.redBandMasked,self.greenBandMasked,self.blueBandMasked,8000)

    def applyNDI(self):

        img = self.images[0]

        NDI_Numerator=self.greenBand - self.redBand #DataGREEN-DataRED
        NDI_Denominator=self.greenBand - self.redBand #DataGREEN+DataRED
        with np.errstate(divide='ignore', invalid='ignore'):
            NDI =np.true_divide(NDI_Numerator,NDI_Denominator)
            NDI[NDI == np.inf] = 0
            NDI = np.nan_to_num(NDI)

        MaskNDI = NDI > 0.100
        plt.imshow(MaskNDI)
        plt.show()

        # The mask we computed above is the opposite of what is needed
        # We want the green bits to show through.
        # TODO: Make this a 3d mask.  Seems like that should be possible
        negated = np.logical_not(MaskNDI)
        DataREDMasked = self.redBand * negated #DataRED * negated
        DataGREENMasked = self.greenBand * negated #DataGREEN * negated
        DataBLUEMasked = self.blueBand * negated #DataBLUE * negated

        RGBImage=vip.Image_getRGB(DataREDMasked,DataGREENMasked,DataBLUEMasked,8000)

        #result = cv.bitwise_and(RGBImage,MaskNDI)

        plt.imshow(RGBImage)
        plt.imsave('crop.jpg', RGBImage)
        plt.show()

if __name__ == "__main__":
    utility = ImageUtilities()

    utility.Load('./overhead.jpg')
    #utility.Load('./drone-pictures/DJI_0074.jpg')
    mask = utility.TGIMask(110)
    plt.imshow(mask)
    plt.show()
    utility.applyMask()
    image = utility.getImage()
    plt.title("Masked image")
    plt.imshow(image)
    plt.show()

    #utility.applyNDI()


