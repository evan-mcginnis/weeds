import cv2

#stitcher = cv2.createStitcher(False)
stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
foo = cv2.imread("right.jpg")
bar = cv2.imread("left.jpg")
status, pano = stitcher.stitch([foo,bar])
cv2.imwrite("stitched.jpg", pano)
