

#from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import viplab_lib as vip


class VegetationIndex:

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

    def GetImage(self):
        return self.img

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
        #plt.imshow(self.img)
        #plt.show()

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

    def GetImage(self):
        return(self.img)

    def ShowImage(self, title : str, index : np.ndarray):

        plt.title(title)
        plt.imshow(index, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def ExcessRedMask(self) -> np.ndarray:

        excessRed = 1.3*self.redBand - self.greenBand

        return excessRed

    def ExcessGreenMask(self) -> np.ndarray:
        # 2g-r-b

        excessGreen = 2*self.greenBand - self.redBand - self.blueBand

        return excessGreen

    def ExGR(self):
        exgr = np.empty_like(self.redBand)

        exgr = self.ExcessGreenMask() - self.ExcessRedMask()

        return exgr

    def CIVEMask(self):

        cive = 0.441*self.redBand - 0.811*self.greenBand + 0.385*self.blueBand + 18.78745

        return cive

    def NGRDI(self):

        ngrdi = (self.greenBand - self.redBand) / (self.greenBand + self.redBand)

        return ngrdi

    def VEGMask(self):

        veg = self.greenBand / self.redBand ** 0.667 * self.blueBand ** (1-0.667)

        return veg

    def COM1Mask(self):

        com1 = self.ExcessGreenMask() + self.CIVEMask() + self.ExGR() + self.VEGMask()

        return com1

    def MEXGMask(self):

        mexg = 1.262 * self.greenBand - 0.884 * self.redBand - 0.311 * self.blueBand

        return mexg

    def COM2Mask(self):

        com2 = (0.36 * self.ExcessGreenMask()) + (0.47 * self.CIVEMask()) + (0.17 * self.VEGMask())

        return com2

    def NDIMask(self):

        img = self.images[0]

        NDI_Numerator=self.greenBand - self.redBand #DataGREEN-DataRED
        NDI_Denominator=self.greenBand + self.redBand #DataGREEN+DataRED
        with np.errstate(divide='ignore', invalid='ignore'):
            NDI =np.true_divide(NDI_Numerator,NDI_Denominator) + 1
            NDI= NDI * 128
            NDI[NDI == np.inf] = 0
            NDI = np.nan_to_num(NDI)

        MaskNDI = NDI > 0.100

        # The mask we computed above is the opposite of what is needed
        # We want the green bits to show through.
        # TODO: Make this a 3d mask.  Seems like that should be possible
        negated = np.logical_not(MaskNDI)
        DataREDMasked = self.redBand * negated #DataRED * negated
        DataGREENMasked = self.greenBand * negated #DataGREEN * negated
        DataBLUEMasked = self.blueBand * negated #DataBLUE * negated

        RGBImage=vip.Image_getRGB(DataREDMasked,DataGREENMasked,DataBLUEMasked,8000)

        return NDI
        #result = cv.bitwise_and(RGBImage,MaskNDI)

if __name__ == "__main__":
    utility = VegetationIndex()

    #utility.Load('./IMG_1302.jpg')
    utility.Load('./overhead.jpg')
    #utility.Load('./drone-pictures/DJI_0074.jpg')

    utility.ShowImage("Source", utility.GetImage())

    indices = {"NDI": utility.NDIMask,
               "EGr": utility.ExcessGreenMask,
               "ExR": utility.ExcessRedMask,
               "CIVE": utility.CIVEMask,
               "ExGR": utility.ExGR,
               "NGRDI": utility.NGRDI,
               "VEG": utility.VEGMask,
               "COM1": utility.COM1Mask,
               "MExG": utility.MEXGMask,
               "COM2": utility.COM2Mask}

    for indexName, createIndex in indices.items():
        vegIndex = createIndex()
        utility.ShowImage(indexName, vegIndex)

    #utility.NDIMask()
    #utility.ExcessGreenMask()
    mask = utility.TGIMask(100)
    plt.imshow(mask)
    plt.show()
    utility.applyMask()
    image = utility.getImage()
    plt.title("Masked image")
    plt.imshow(image)
    plt.show()

    #utility.applyNDI()


