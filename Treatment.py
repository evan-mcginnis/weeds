#
# T R E A T M E N T
#
# Generate a treatment plan given the classified blob list
#
import os

import numpy as np
import pandas as pd
import cv2 as cv
import logging

import constants


# Size of the cells in treatment grid
GRID_SIZE = 40

# Number of treatment sprayers
TREATMENT_HEADS = 12

# Minimum percentage of vegetation in a cell to be considered a treatment target
MIN_VEGETATION_FACTOR = 0.75

class Treatment:
    def __init__(self, image: np.ndarray, binary: np.ndarray):
        self._image = image
        self._binary = binary
        (self._maxRows, self._maxColumns, self._depth) = self._image.shape
        rows, cols = (int(self._maxRows / GRID_SIZE), int(self._maxColumns/ GRID_SIZE))
        self._treatmentGrid = [[0]*cols]*rows
        self._grid = pd.DataFrame(dtype=int)
        self._treatmentPlan = pd.DataFrame(dtype=int)
        self.log = logging.getLogger(__name__)

        # Determine the number of pixels and individual sprayer head can cover
        self._sprayerCoverage = int(self._maxRows / TREATMENT_HEADS)
        self._gridSize = self._sprayerCoverage

    @property
    def image(self):
        """
        The current image
        :return:
        """
        return self._image

    @property
    def grid(self):
        """
        The treatment grid
        :return:
        """
        return self._treatmentGrid

    @property
    def plan(self):
        """
        The treatment plan for the images arranged as lines.
        :return:
        The treatment plan as a dataframe
        """
        return self._treatmentPlan



    def overlayTreatmentLanes(self):
        """
        Draw a grid on the current image. The cell size is statically defined.
        This is purely cosmetic. No calculations or manipulations performed.
        """
        (maxRows, maxColumns, depth) = self._image.shape
        # The vertical lines aren't needed anymore
        for i in range(0,maxRows, self._gridSize):
            cv.line(self._image,(0,i), (maxColumns,i),constants.COLOR_TREATMENT_GRID, 1, cv.LINE_AA)
        sprayerLane = 0
        for j in range(0,maxColumns,self._gridSize):
            laneText = constants.SPRAYER_NAME + " " + str(sprayerLane)
            # TODO: This is a bit of a mess that needs to be sorted out.
            if j > 0:
                adjustment = int(self._gridSize / 2)
            else:
                adjustment = 0

            cv.putText(self._image, laneText, (125, j - adjustment), cv.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 2)
            #cv.line(self._image,(j,0), (j,maxRows),constants.COLOR_TREATMENT_GRID, 1, cv.LINE_AA)
            sprayerLane = sprayerLane + 1

    def _gridNumber(self, center: ()) -> ():
        """
        The grid location given the center coordinates.
        :param center: The center
        :return: a tuple indicating the grid row and column
        """
        (centerX, centerY) = center
        gridRow = int((self._maxRows - centerY) / self._gridSize) + 1
        #gridColumn = int((self._maxColumns - centerX) / self._gridSize)
        gridColumn = int(centerX / self._gridSize)
        return gridRow, gridColumn

    def _gridToUpperLeftCoordinates(self, location: ()) -> ():
        """
        Get the coordinates of the upper left corner corresponding to the cell in the treatment matrix
        :param location: The location in the treatment matrix
        :return: The corresponding location in the image
        """
        (gridRow, gridColumn) = location
        upperLeftColumn = gridColumn * self._gridSize
        upperLeftRow = self._maxRows - (gridRow * self._gridSize)
        return upperLeftColumn, upperLeftRow

    def _hasVegetationInGrid(self, location: ()) -> bool:
        """
        Determines if the given grid position is sufficiently vegetated

        :param location: A tuple of the grid position
        :return: True if vegetated, False otherwise
        """
        #(gridRow, gridColumn) = location
        (upperLeftColumn, upperLeftRow) = self._gridToUpperLeftCoordinates(location)
        gridContents = self._binary[upperLeftRow:upperLeftRow + self._gridSize, upperLeftColumn:upperLeftColumn+self._gridSize]
        #cv.imwrite("grid-" + str(upperLeftColumn) + "-" + str(upperLeftRow) + ".jpg", gridContents)

        # If anything is the grid is a vegetated pixel, return true
        return np.any(gridContents)

        # This is a threshold based scheme we don't want to use
        # The average of a fully vegetated cell is 255
        average = np.average(gridContents)
        #print("Average in cell: {:.2f}".format(average))
        return average > (MIN_VEGETATION_FACTOR * 255)


    def generatePlan(self, classified: {}) -> []:
        """
        Generate a treatment plan given the classified blobs.
        :param classified: The blobs in the image that have already classified
        :return: a 2D matrix. Cells with the constant UNDESIRABLE are slated for treatment
        """
        self.log.info("Generating plan")
        for classifiedName, classifiedAttributes in classified.items():
            if classifiedAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED:
                center = classifiedAttributes[constants.NAME_CENTER]
                (x, y, w, h) = classifiedAttributes[constants.NAME_LOCATION]
                self.log.debug("Undesired vegetation at xy = ({:},{:}) w = {:} h = {:}".format(x, y, w, h))
                (gridRow, gridColumn) = self._gridNumber(center)
                self.log.debug("Center " + str(center) +
                               " is at grid " + str((gridRow, gridColumn)) +
                               " " + str(self._gridToUpperLeftCoordinates((gridRow, gridColumn))))
                #self._treatmentGrid[gridRow][gridColumn] = center
                #self._treatmentGrid[gridRow][gridColumn] = center
                cv.rectangle(self._image,
                             (gridColumn * self._gridSize, self._maxRows - (gridRow * self._gridSize)),
                             ((gridColumn * self._gridSize) + self._gridSize, (self._maxRows - (gridRow * self._gridSize)) + self._gridSize),
                             constants.COLOR_TREATMENT_WEED,
                            4)
                # Find all the cells within the bounding box of the undesirable vegetation
                for i in range(x,x+w, self._gridSize):
                    for j in range(y, y+h, self._gridSize):
                        (gridRow, gridColumn) = self._gridNumber((i, j))
                        #print("Mark to be treated: " + str(gridNumber))
                        if self._hasVegetationInGrid((gridRow, gridColumn)):
                            cv.rectangle(self._image,
                                         (gridColumn * self._gridSize, self._maxRows - (gridRow * self._gridSize)),
                                         ((gridColumn * self._gridSize) + self._gridSize, (self._maxRows - (gridRow * self._gridSize)) + self._gridSize),
                                         constants.COLOR_TREATMENT_WEED,
                                        4)
                            self._treatmentGrid[gridRow][gridColumn] = constants.TYPE_UNDESIRED

        return self._treatmentGrid


    def drawTreatmentPlan(self, classified : {}):
        """
        Deprecated. Draw the treatment plan on the current image
        :param classified:  The previously classified blobs
        """

        self.log.error("Deprecated method called")
        raise NotImplementedError

        # load image
        img = self._image

        # define undercolor region in the input image
        for classifiedName, classifiedAttributes in classified.items():
            if classifiedAttributes[constants.NAME_TYPE] == constants.TYPE_UNDESIRED:
                center = classifiedAttributes[constants.NAME_CENTER]
                (x, y, w, h) = classifiedAttributes[constants.NAME_LOCATION]
                #x,y,w,h = 66,688,998,382

                # define text coordinates in the input image
                xx,yy = 250,800

                # compute text coordinates in undercolor region
                xu = xx - x
                yu = yy - y

                # crop undercolor region of input
                sub = self._image[y:y+h, x:x+w]

                # create black image same size
                black = np.zeros_like(sub)

                # blend the two
                blend = cv.addWeighted(sub, 0.25, black, 0.75, 0)

                self._image[y:y+h, x:x+w] = blend



    def _drawTreatmentLanes(self):
        """
        Deprecated. Draw the lanes for the treatment nozzles on the image.  No computation performed here.
        """

        # Not used anymore, but keep around to show how to overlay blended images

        self.log.error("Deprecated method called")
        raise NotImplementedError

        image = self._image

        # 2. Get tags
        # General label format 　bbox = [x, y, w, h]
        bbox = [192, 364, 267, 37]

        # 3. Draw the mask
        zeros = np.zeros((image.shape), self._image.dtype)
        bbox = [int(b) for b in bbox]
        bbox[2] = bbox[2] + bbox[0]
        bbox[3] = bbox[3] + bbox[1]
        zeros_mask = cv.rectangle(zeros,
                                  (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                                  color=(0, 0, 255),
                                  thickness=-1)  # thickness=-1 indicates the color fill in the rectangular frame
        # 4. Save the drawn mask and read it again.
        # The following cv2.addWeighted cannot directly use the mask to merge with the original picture,
        # As for why, I don't know, anyway, it will report an error if used directly.
        cv.imwrite('zeros_mask.jpg', zeros_mask)
        mask = cv.imread('zeros_mask.jpg', cv.IMREAD_ANYDEPTH)

        try:
            # alpha is the transparency of the first picture
            alpha = 1
            # beta is the transparency of the second picture
            beta = 0.5
            gamma = 0
            # cv2.addWeighted merge the original image with the mask
            mask_img = cv.addWeighted(image, alpha, mask, beta, gamma)
            #cv.imwrite(os.path.join(output_fold, 'mask_img.jpg'), mask_img)
        except Exception as e:
            print(e)
