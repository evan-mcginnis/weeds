#
# T R E A T M E N T
#
# Generate a treatment plan given the classified blob list
#


import numpy as np
import pandas as pd
import cv2 as cv
import logging

import constants

# Size of the cells in treatment grid
GRID_SIZE = 40

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
        self.log = logging.getLogger(__name__)

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

    def overlayTreatmentGrid(self):
        """
        Draw a grid on the current image. The cell size is statically defined
        """
        (maxRows, maxColumns, depth) = self._image.shape
        for i in range(0,maxRows, GRID_SIZE):
            cv.line(self._image,(0,i), (maxColumns,i),constants.COLOR_TREATMENT_GRID, 1, cv.LINE_AA)
        for j in range(0,maxColumns,GRID_SIZE):
            cv.line(self._image,(j,0), (j,maxRows),constants.COLOR_TREATMENT_GRID, 1, cv.LINE_AA)

    def _gridNumber(self, center: ()) -> ():
        """
        The grid location given the center coordinates.
        :param center: The center
        :return: a tuple indicating the grid row and column
        """
        (centerX, centerY) = center
        gridRow = int((self._maxRows - centerY) / GRID_SIZE) + 1
        #gridColumn = int((self._maxColumns - centerX) / GRID_SIZE)
        gridColumn = int(centerX / GRID_SIZE)
        return gridRow, gridColumn

    def _gridToUpperLeftCoordinates(self, location: ()) -> ():
        """
        Get the coordinates of the upper left corner corresponding to the cell in the treatment matrix
        :param location: The location in the treatment matrix
        :return: The corresponding location in the image
        """
        (gridRow, gridColumn) = location
        upperLeftColumn = gridColumn * GRID_SIZE
        upperLeftRow = self._maxRows - (gridRow * GRID_SIZE)
        return upperLeftColumn, upperLeftRow

    def _hasVegetationInGrid(self, location: ()) -> bool:
        """
        Determines if the given grid position is sufficiently vegetated

        :param location: A tuple of the grid position
        :return: True if vegetated, False otherwise
        """
        #(gridRow, gridColumn) = location
        (upperLeftColumn, upperLeftRow) = self._gridToUpperLeftCoordinates(location)
        gridContents = self._binary[upperLeftRow:upperLeftRow + GRID_SIZE, upperLeftColumn:upperLeftColumn+GRID_SIZE]
        #cv.imwrite("grid-" + str(upperLeftColumn) + "-" + str(upperLeftRow) + ".jpg", gridContents)
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
                print("Undesired vegetation at xy = ({:},{:}) w = {:} h = {:}".format(x, y, w, h))
                (gridRow, gridColumn) = self._gridNumber(center)
                print("Center " + str(center) +
                      " is at grid " + str((gridRow, gridColumn)) +
                      " " + str(self._gridToUpperLeftCoordinates((gridRow, gridColumn))))
                #self._treatmentGrid[gridRow][gridColumn] = center
                #self._treatmentGrid[gridRow][gridColumn] = center
                cv.rectangle(self._image,
                             (gridColumn * GRID_SIZE, self._maxRows - (gridRow * GRID_SIZE)),
                             ((gridColumn * GRID_SIZE) + GRID_SIZE, (self._maxRows - (gridRow * GRID_SIZE)) + GRID_SIZE),
                             constants.COLOR_TREATMENT_WEED,
                            4)
                # Find all the cells within the bounding box of the undesirable vegetation
                for i in range(x,x+w, GRID_SIZE):
                    for j in range(y, y+h, GRID_SIZE):
                        (gridRow, gridColumn) = self._gridNumber((i, j))
                        #print("Mark to be treated: " + str(gridNumber))
                        if self._hasVegetationInGrid((gridRow, gridColumn)):
                            cv.rectangle(self._image,
                                         (gridColumn * GRID_SIZE, self._maxRows - (gridRow * GRID_SIZE)),
                                         ((gridColumn * GRID_SIZE) + GRID_SIZE, (self._maxRows - (gridRow * GRID_SIZE)) + GRID_SIZE),
                                         constants.COLOR_TREATMENT_WEED,
                                        4)
                            self._treatmentGrid[gridRow][gridColumn] = constants.TYPE_UNDESIRED

        return self._treatmentGrid


