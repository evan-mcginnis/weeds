#
# I M A G E
#
# Image manipulation
import uuid

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import constants

# Colors for the bounding boxes
COLOR_WEED = (255,0,0)
COLOR_CROP = (0,0,255)
COLOR_UNKNOWN = (0,255,0)

# How far outside the midline of the image vegetation should be considered the cropline
MIDDLE_THRESHOLD = 200

# The lines for the enclosing rectangle
BOUNDING_BOX_THICKNESS = 2

class ImageManipulation:
    def __init__(self, img : np.ndarray):
        self._image = img
        self._rectangles = []
        self._largestName = ""
        self._largestArea = 0
        self._blobs = {}
        self._cropRowCandidates = {}
        self._mmPerPixel = 0
        self._stitcher = cv.Stitcher.create(cv.Stitcher_PANORAMA)

    def init(self):
        self._cvimage = cv.cvtColor(self._image)

    @property
    def mmPerPixel(self) -> float:
        return self._mmPerPixel

    @mmPerPixel.setter
    def mmPerPixel(self, mm : float):
        self._mmPerPixel = mm

    @property
    def image(self):
        return self._image

    @property
    def croplineImage(self):
        return self.cropline_image

    @classmethod
    def show(self, title : str, index : np.ndarray):
        plt.title(title)
        plt.imshow(index, cmap='gray', vmin=0, vmax=255)
        plt.show()

    @classmethod
    def statistics(self, target: np.ndarray):
        nonZeroCells = np.count_nonzero(target > 0, keepdims=False)
        count = (target != 0.0).sum()
        return nonZeroCells

    @classmethod
    def save(self, image: np.ndarray, name: str):
        data = Image.fromarray((image * 255).astype(np.uint8))
        data.save(name)

    def write(self, image: np.ndarray, name: str):
        cv.imwrite(name, image)

    def toGreyscale(self) -> np.ndarray:
        """
        The current image converted to greyscale
        :return:
        The greyscale image as a numpy array
        """
        # This method of converting to greyscale is a complete hack.
        #self.save(self._image, "temporary.jpg")
        #utility.SaveMaskedImage("mask-applied.jpg")
        #img_float32 = np.float32(utility.GetMaskedImage())
        #img = cv.imread("temporary.jpg")
        #self._imgAsGreyscale = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        #self._imgAsGreyscale = self._image.astype(np.uint8)
        # If the conversion to uint8 is not there, opencv complains when we try to find the
        # contours.  Strictly speaking this is not required for just the greyscale conversion

        img = cv.cvtColor(self._image.astype(np.uint8), cv.COLOR_BGR2GRAY)
        #cv.imwrite("converted.jpg", img)
        self._imgAsGreyscale = img

        return self._imgAsGreyscale

    def findEdges(self, image: np.ndarray):
        self._edges = cv.Canny(self._imgAsGreyscale, 20, 30)
        return

    def cartoon(self):
        self._cartooned = np.where(self._imgAsGreyscale > 0, 255, self._imgAsGreyscale)
        return self._cartooned

    def mmBetweenPoints(self, point1: (), point2: (), mmPerPixel: float) -> int:
        distance = 0
        (x1, y1) = point1
        (x2, y2) = point2
        distance = int((x2 - x1) * mmPerPixel)
        return distance

    def findBlobs(self, threshold : int) -> ([], np.ndarray, {}, str):
        """
        Find objects within the current image
        :return: (contours, hierarchy, bounding rectangles, name of largest object)
        """
        self.toGreyscale()
        #self.show("grey", self._imgAsGreyscale)
        #self.write(self._imgAsGreyscale, "greyscale.jpg")

        self.cartoon()
        #self.show("cartooned", self._cartooned)
        #self.write(self._image, "original.jpg")
        #self.write(self._cartooned, "cartooned.jpg")

        #self.write(self._image, "index.jpg")

        # Convert to binary image
        # Works
        #ret,thresh = cv.threshold(self._cartooned,127,255,0)
        ret,thresh = cv.threshold(self._imgAsGreyscale, 127,255,cv.THRESH_BINARY)
        #self.write(thresh, "threshold.jpg")
        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(self._cartooned, kernel,iterations = 3)
        #self.write(erosion, "erosion.jpg")
        erosion = cv.dilate(erosion,kernel,iterations = 3)

        closing = cv.morphologyEx(erosion, cv.MORPH_CLOSE, kernel)
        #self.write(closing, "closing.jpg")
        #self.show("binary", erosion)
        #self.write(erosion, "binary.jpg")

        # Originally
        # candidate = erosion
        candidate = closing
        # find contours in the binary image
        #im2, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        # We don't need the hierarchy at this point, so the RETR_LIST seems faster
        #contours, hierarchy = cv.findContours(erosion,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv.findContours(candidate,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        self._contours = contours

        # Calculate the area of each box
        i = 0
        largest = 0
        for c in contours:
            M = cv.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            x,y,w,h = cv.boundingRect(c)
            area = w*h
            type = constants.TYPE_UNKNOWN
            location = (x,y,w,h)
            center = (cX,cY)
            infoAboutBlob = {constants.NAME_LOCATION: location,
                             constants.NAME_CENTER: center,
                             constants.NAME_AREA: area,
                             constants.NAME_TYPE: type}
            name = "blob" + str(i)
            # Ignore items in the image that are smaller in area than the
            # threshold.  Things in shadow and noise will be identified as shapes
            if area > threshold:
                self._blobs[name] = infoAboutBlob
            i = i + 1

            # Determine the largest blob in the image
            if area > largest:
                largest = area
                largestName = name

        self._largestName = largestName
        self._largestArea = area

        return contours, hierarchy, self._blobs, largestName

    def identifyCropRowCandidates(self):
        """
        Create a list of the vegetation likely to be in the crop row.
        For now, use a very simple approach -- if it is roughly in the center of the image
        it is likely part of the crop
        """
        (height, width, depth) = self.image.shape

        # This is approximately the midline of the image
        middleY = int(height/2)
        for blobName, blobAttributes in self._blobs.items():
            (x, y, w, h) = blobAttributes[constants.NAME_LOCATION]
            (x, y) = blobAttributes[constants.NAME_CENTER]
            # Items near the midline are more likely to be crop
            # ?? Perhaps we should also consider equal spacing?
            if(y > middleY - MIDDLE_THRESHOLD and y < middleY + MIDDLE_THRESHOLD):
                self._cropRowCandidates[blobName] = blobAttributes

    def substituteRectanglesForVegetation(self):
        """
        Using the identified centers of where the vegetation, draw low height rectangles
        that are later used for crop line detection
        """
        self.cropline_image = np.zeros(self._image.shape, np.uint8)
        for blobName, blobAttributes in self._cropRowCandidates.items():
            (x,y,w,h) = blobAttributes[constants.NAME_LOCATION]
            (x,y) = blobAttributes[constants.NAME_CENTER]
            #cv.circle(self.blank_image,(x,y),1, (255,255,255), -1)
            self.cropline_image = cv.rectangle(self.cropline_image, (x, y), (x + 20, y + 10), (255, 255, 255), 2)

        #filename = "candidates-" + str(uuid.uuid4()) + ".jpg"

        #cv.imwrite(filename, self.cropline_image)

    def detectLines(self):

        dst = cv.Canny(self.cropline_image, 50, 200, None, 3)
        cv.imwrite("edges.jpg", dst)
        #self.linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, minLineLength=100, maxLineGap=1000)
        # This allows for some plants in the crop line to be slightly offset from other plants
        # in the same row
        # What is needed is to detect roughly horizontal lines
        self.linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 1, None, minLineLength=100, maxLineGap=1000)
        if self.linesP is not None:
            for i in range(0, len(self.linesP)):
                l = self.linesP[i][0]
                cv.line(self.cropline_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        #cv.imwrite("crop-lines.jpg", self.cropline_image)

    def drawCropline(self):
        """
        Draw a cropline on the current image if one has been found.
        """
        if self.linesP is not None:
            for i in range(0, len(self.linesP)):
                l = self.linesP[i][0]
                cv.line(self._image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
                cv.putText(self._image,
                           str(self.mmBetweenPoints((l[0],l[1]), (l[2], l[3]), self._mmPerPixel)) + " mm",
                           (l[0] + 25, l[1] + 25),
                           cv.FONT_HERSHEY_SIMPLEX,
                           0.75,
                           (255,255,255),
                           2)


    def drawContours(self):
        self._contours_image = np.zeros(self._image.shape, np.uint8)
        cv.drawContours(self._contours_image, self._contours, contourIdx=-1, color=(255,0,0),thickness=30)
        cv.imwrite("contours.jpg", self._contours_image)

    def drawBoxes(self, rectangles: []):
        """
        Draw colored boxes around the blobs based on what type they are
        :param rectangles: A list of rectangles surrounding the blobs
        """
        for rectName, rectAttributes in rectangles.items():
            (x,y,w,h) = rectAttributes[constants.NAME_LOCATION]
            (cX,cY) = rectAttributes[constants.NAME_CENTER]
            type = rectAttributes[constants.NAME_TYPE]
            if type == constants.TYPE_UNKNOWN:
                color = COLOR_UNKNOWN
            elif type == constants.TYPE_UNDESIRED:
                color = COLOR_WEED
            else:
                color = COLOR_CROP
            self._image = cv.rectangle(self._image,(x,y),(x+w,y+h),color,2)
            cv.circle(self._image, (cX, cY), 5, (255, 255, 255), -1)
            location = "(" + str(cX) + "," + str(cY) + ")"
            cv.putText(self._image, location, (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            #asRGB = self._image * 255

        #asRGB = Image.fromarray(self._image.astype(np.float32))
        #self.save(self._image, "centers.jpg")
        #cv.imwrite("opencv-centers.jpg", self._image)
        #self.show("centers", self._image)
        #cv.waitKey()

    def stitchTo(self, previous):
        # Stitch the current image to the previous one
        status, pano = self._stitcher.stitch([previous.astype(np.uint8),self._image.astype(np.uint8)])
        return pano
        #cv.imwrite("stitched.jpg", pano)

    def drawBoundingBoxes(self, contours: []):
        for c in contours:
            # calculate moments for each contour
            M = cv.moments(c)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            # calculate x,y coordinate of center
            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            cv.circle(self._image, (cX, cY), 5, (255, 255, 255), -1)
            cv.putText(self._image, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Do not consider rotation -- just bound the object
            x,y,w,h = cv.boundingRect(c)
            self._image = cv.rectangle(self._image,(x,y),(x+w,y+h),COLOR_CROP,2)

            # Minimum area bounding
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(self._image,[box],0,COLOR_UNKNOWN,BOUNDING_BOX_THICKNESS)

            # Append the current rectangle
            self._rectangles.append(rect)

            #cv.imwrite("centers.jpg", cv.cvtColor(self._image,))
        self.show("centers", self._image)
        cv.waitKey()
        return
