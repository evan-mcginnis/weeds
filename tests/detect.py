import cv2 as cv
from ImageManipulation import ImageManipulation

img = cv.imread("boxes.jpg")

#blurred = cv.pyrMeanShiftFiltering(img,51,91)
manipulated = ImageManipulation(img)

contours, hierarchy, blobs, largest = manipulated.findBlobs(1)
manipulated.drawBoxes(blobs)

#cv.namedWindow("display", cv.WINDOW_NORMAL)
#cv.imshow("display", img)
#cv.waitKey()
cv.imwrite("detected.jpg", manipulated.image)
print(hierarchy)
